# 📖 Documentación Completa del Framework `PYMBBO`

Bienvenido a la documentación oficial de **`PYMBBO`**, un framework modular, intuitivo y de alto rendimiento diseñado en Python para acelerar el desarrollo, entrenamiento, evaluación y despliegue de redes neuronales y modelos de Inteligencia Artificial.

---

## 🚀 Visión General de la Arquitectura

`PYMBBO` está estructurado bajo el principio de **desacoplamiento total y escalabilidad por plugins**. El desarrollador solo necesita interactuar con el cliente central (`BaseModel` / `build_model`), mientras que la ingesta de datos, los componentes matemáticos, las métricas de prueba y las arquitecturas de red pueden expandirse independientemente sin alterar el código fuente del framework.

```
c:\Users\gerso\Pictures\Framework IA\
├── pyproject.toml                     # Configuración de empaquetado PyPI
├── setup.py                           # Script de instalación para pip
├── README.md                          # Guía rápida para PyPI
├── DOCS.md                            # Documentación técnica completa (este archivo)
├── LICENSE                            # Licencia MIT
├── pymbbo/                            # Paquete raíz de la librería
│   ├── __init__.py                    # Exportación central de API y auto-descubrimiento
│   ├── config.py                      # Módulo 1: Gestión de Hiperparámetros (Config / Hyperparameters)
│   ├── data/                          # Módulo 2: Ingesta, Procesamiento y Preparación de Datos
│   │   ├── __init__.py
│   │   ├── dataset.py                 # Dataset, load_dataset, split(), transform()
│   │   └── batcher.py                 # DataCollator y Batcher
│   ├── models/                        # Módulo 3 & 7: Construcción, Layers y Persistencia
│   │   ├── __init__.py
│   │   ├── base.py                    # BaseModel (cliente principal, summary, freeze, fit, predict)
│   │   ├── factory.py                 # build_model (patrón fábrica de modelos)
│   │   ├── layers.py                  # Dense, Conv2D, Dropout, BatchNorm, Flatten
│   │   ├── registry.py                # Sistema de Registro y Auto-Descubrimiento de Arquitecturas
│   │   └── persistence.py             # save_model, load_model, export_model (ONNX / TorchScript)
│   ├── architectures/                 # 📂 CARPETA PLUG-AND-PLAY DE ARQUITECTURAS
│   │   ├── __init__.py
│   │   ├── base_arch.py               # BaseArchitecture (Clase abstracta obligatoria)
│   │   ├── mlp/                       # Plugin 1: Perceptrón Multicapa (MLP)
│   │   │   ├── __init__.py
│   │   │   └── model.py
│   │   ├── cnn/                       # Plugin 2: Red Convencional 2D (CNN)
│   │   │   ├── __init__.py
│   │   │   └── model.py
│   │   └── transformer/               # Plugin 3: Decoder Transformer / LLM para Pruebas de Tokens
│   │       ├── __init__.py
│   │       └── model.py
│   ├── engine/                        # Módulo 4 & 5: Bucle de Entrenamiento e Inyección de Componentes
│   │   ├── __init__.py
│   │   ├── trainer.py                 # fit(), evaluate(), predict(), get_metrics()
│   │   └── callbacks.py               # EarlyStopping, ModelCheckpoint, LRScheduler
│   └── metrics/                       # Módulo 6: Métricas Estándar y Pruebas Especializadas de Benchmarking
│       ├── __init__.py
│       ├── standard.py                # Accuracy, F1, Precision, Recall, MAE, MSE, Perplexity
│       └── benchmark.py               # token_scaling_benchmark y compare_models
└── tests/                             # Suite de pruebas automatizadas
    └── test_pymbbo.py                 # Pruebas unitarias de cobertura 100%
```

---

## 🛠️ Explicación Detallada Módulo por Módulo

### 1. Configuración de Hiperparámetros y Experimentos (`pymbbo/config.py`)
Administra los parámetros centralizados del experimento (`learning_rate`, `batch_size`, `epochs`, `optimizer`, `loss_function`, `seed`, etc.).

#### Características:
- Acceso por atributos: `config.learning_rate`.
- Acceso tipo diccionario: `config["batch_size"]`.
- Persistencia JSON: `config.save("config.json")` y `Config.load("config.json")`.

```python
from pymbbo import Config

config = Config(
    learning_rate=0.001,
    batch_size=64,
    epochs=20,
    optimizer="adam",
    loss_function="cross_entropy"
)

# Guardar en disco para reproducibilidad exacta
config.save("experimento_01.json")

# Cargar configuración previa
config_cargada = Config.load("experimento_01.json")
```

---

### 2. Ingesta, Procesamiento y Preparación de Datos (`pymbbo/data/`)
Conecta fuentes de datos crudas (arreglos de NumPy, tensores PyTorch, tuplas `(X, Y)`, archivos CSV) con el modelo.

#### Funciones y Métodos Clave:
- `load_dataset(source, target_column=None)`: Carga datasets desde múltiples fuentes.
- `dataset.split(train=0.8, val=0.1, test=0.1)`: Divide en conjuntos de entrenamiento, validación y prueba.
- `dataset.transform(preprocess_fn)`: Aplica transformaciones o aumentación de datos en pipeline.
- `Batcher(dataset, batch_size=32, shuffle=True)`: Empaqueta los elementos en lotes homogéneos.

```python
import numpy as np
from pymbbo import load_dataset, Batcher

# Cargar desde arreglos NumPy o archivos CSV
X = np.random.randn(1000, 20)
Y = np.random.randint(0, 2, (1000, 1))

dataset = load_dataset((X, Y))

# Transformación de datos
dataset.transform(lambda x, y: (x * 1.5, y))

# División automática (80% Train, 10% Val, 10% Test)
train_ds, val_ds, test_ds = dataset.split(train=0.8, val=0.1, test=0.1)

# Empaquetado en lotes para GPU/CPU
train_batcher = Batcher(train_ds, batch_size=32)
```

---

### 3. Construcción del Modelo y Sistema Plug-and-Play de Arquitecturas (`pymbbo/models/` y `pymbbo/architectures/`)

`PYMBBO` permite construir modelos de dos formas:
1. **Modo Secuencial**: Añadiendo capas una a una con `add_layer`.
2. **Modo Arquitectura**: Utilizando arquitecturas predefinidas o creadas por la comunidad.

#### resumen de la Arquitectura (`summary()`):
Muestra una desglose con los nombres de capa, formas de salida, parámetros entrenables y parámetros congelados.

#### Congelamiento para Fine-Tuning (`freeze_layers()` / `unfreeze()`):
Permite congelar capas específicas o todo el modelo para Transfer Learning.

```python
from pymbbo import build_model

# 1. Construcción Secuencial
model_seq = build_model("sequential")
model_seq.add_layer("dense", units=64, activation="relu", in_features=20)
model_seq.add_layer("dropout", rate=0.2)
model_seq.add_layer("dense", units=1, activation="sigmoid")

# 2. Resumen visual del modelo
model_seq.summary()

# 3. Congelar capas para Fine-Tuning
model_seq.freeze_layers("all")
```

---

### 🔌 Guía para Crear y Agregar Nuevas Arquitecturas

Esta es la funcionalidad estrella de **`PYMBBO`**. Para agregar una nueva arquitectura al framework sin modificar el código interno:

1. Dirígete a la carpeta `pymbbo/architectures/`.
2. Crea una subcarpeta con el nombre de tu arquitectura (ej. `pymbbo/architectures/mi_red_personalizada/`).
3. Crea un archivo `model.py` (o cualquier archivo `.py`).
4. Importa `BaseArchitecture` y el decorador `@register_architecture`.
5. Hereda de `BaseArchitecture` e implementa `__init__` y `forward`.

#### Ejemplo Completo:

Crea el archivo `pymbbo/architectures/mi_red_personalizada/model.py`:

```python
import torch
import torch.nn as nn
from pymbbo.architectures.base_arch import BaseArchitecture
from pymbbo.models.registry import register_architecture

@register_architecture("mi_red")
class MiRedPersonalizada(BaseArchitecture):
    ARCH_NAME = "mi_red"

    def __init__(self, input_dim: int = 20, hidden_dim: int = 128, output_dim: int = 2, **kwargs):
        super().__init__(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, **kwargs)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.fc1(x))
        return self.fc2(out)
```

¡Listo! Al importar `pymbbo`, el marco detectará automáticamente `mi_red` y podrás instanciarla inmediatamente desde el cliente:

```python
from pymbbo import build_model

model = build_model("mi_red", input_dim=20, hidden_dim=64, output_dim=1)
model.summary()
```

---

### 4 & 5. Compilación, Entrenamiento y Callbacks (`pymbbo/engine/`)

Asocia la red matemática con optimizadores (`adam`, `sgd`, `adamw`), funciones de pérdida (`cross_entropy`, `mse`, `mae`, `bce`) y métricas.

#### Callbacks Incluidos:
- `EarlyStopping(patience=3)`: Detiene el entrenamiento si la pérdida de validación estanca.
- `ModelCheckpoint(filepath)`: Guarda la mejor versión del modelo en disco.
- `LRScheduler(factor=0.5, patience=2)`: Ajusta dinámicamente la tasa de aprendizaje.

```python
from pymbbo import build_model, EarlyStopping, ModelCheckpoint, LRScheduler

model = build_model("mlp", input_dim=20, hidden_units=[64, 32], output_dim=1)

# Compilación
model.compile(
    optimizer="adam",
    loss_function="bce",
    metrics=["accuracy"],
    learning_rate=0.001
)

# Entrenamiento completo con callbacks
callbacks = [
    EarlyStopping(patience=3, monitor="val_loss"),
    ModelCheckpoint("mejor_modelo.mbbo", save_best_only=True),
    LRScheduler(factor=0.5, patience=2)
]

historial = model.fit(
    train_data=train_ds,
    validation_data=val_ds,
    epochs=15,
    batch_size=32,
    callbacks=callbacks
)
```

---

### 6. Métricas Especializadas y Benchmarking (`pymbbo/metrics/`)

`PYMBBO` incluye métricas avanzadas y herramientas para evaluar y comparar rendimiento de modelos de manera intuitiva:

#### A. Pruebas Automáticas de Crecimiento de Tokens (`token_scaling_benchmark`)
Evalúa cómo escala un modelo de lenguaje/secuencias a medida que se incrementa el número de tokens (ej. de `min_tokens=500` a `max_tokens=100,000`). Calcula:
- Tiempo total de generación (s).
- Velocidad de procesamiento (tokens/segundo).
- Latencia por token (ms).
- Costo estimado de cómputo ($).

```python
from pymbbo import build_model, token_scaling_benchmark

model_llm = build_model("transformer", vocab_size=5000, d_model=256)

reporte_tokens = token_scaling_benchmark(
    model=model_llm,
    vocab_size=5000,
    min_tokens=500,
    max_tokens=10000,
    steps=5,
    cost_per_million_tokens=0.002
)
```

#### B. Comparación Simultánea de Modelos (`compare_models`)
Ejecuta y compara múltiples modelos en paralelo bajo el mismo conjunto de prueba:

```python
from pymbbo import build_model, compare_models

model_a = build_model("mlp", input_dim=20, hidden_units=[32], output_dim=1)
model_b = build_model("mlp", input_dim=20, hidden_units=[128, 64], output_dim=1)

model_a.compile(optimizer="adam", loss_function="bce")
model_b.compile(optimizer="adam", loss_function="bce")

reporte_comparativo = compare_models(
    models={
        "Modelo_Ligero": model_a,
        "Modelo_Pesado": model_b
    },
    test_data=val_ds
)
```

---

### 7. Persistencia y Exportación Producción (`pymbbo/models/persistence.py`)

Guarda la arquitectura, pesos y configuración completa en un solo archivo `.mbbo`, y permite reconstruirlo sin necesidad de redefinir el código.

```python
from pymbbo import load_model

# 1. Guardar modelo completo
model.save("mi_modelo_final.mbbo")

# 2. Reconstruir modelo en memoria
modelo_restaurado = load_model("mi_modelo_final.mbbo")

# 3. Exportar a formato ONNX / TorchScript para producción
model.export("modelo_produccion.onnx", format="onnx")
```

---

## 🧪 Pruebas Automatizadas

`PYMBBO` incluye una suite completa de pruebas unitarias en `tests/test_pymbbo.py`. Para ejecutarlas:

```bash
python -m unittest discover tests
```

---

## 📦 Publicación en PyPI

Para empaquetar y publicar **`pymbbo`** en PyPI:

```bash
# 1. Instalar herramientas de empaquetado
pip install build twine

# 2. Generar distribuciones wheel y tar.gz
python -m build

# 3. Subir a PyPI
python -m twine upload dist/*
```
