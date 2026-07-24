# PYMBBO ⚡

[![PyPI version](https://img.shields.io/badge/PyPI-v0.1.0-blue.svg)](https://pypi.org/project/pymbbo/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)

**PYMBBO** es un framework modular, intuitivo y de alto rendimiento en Python diseñado para facilitar la experimentación, entrenamiento, evaluación y despliegue de redes neuronales, ofreciendo un sistema **Plug-and-Play de Arquitecturas** y herramientas avanzadas de **Benchmarking y Comparación de Modelos**.

---

## 🔥 Características Principales

1. **Gestión de Hiperparámetros (`Config` / `Hyperparameters`)**: Objeto central con acceso tipo atributo/diccionario y guardado/carga en formato JSON.
2. **Ingesta y Preparación de Datos (`load_dataset`, `Dataset`, `Batcher`)**: Carga desde CSV, matrices NumPy, tensores y carpetas, con división de conjuntos (`split`) y flujo de transformaciones (`transform`).
3. **Construcción y Ensamblado de Redes (`build_model`)**: Soporta modo secuencial (`add_layer`), resúmenes visuales (`summary()`), congelamiento de capas (`freeze_layers`) y fine-tuning.
4. **Sistema Plug-and-Play de Arquitecturas (`pymbbo/architectures/`)**: Agrega nuevas arquitecturas creando una simple subcarpeta. El framework la descubre y registra automáticamente.
5. **Entrenamiento Avanzado (`fit`)**: Motor de entrenamiento con soporte para Callbacks (`EarlyStopping`, `ModelCheckpoint`, `LRScheduler`).
6. **Métricas Especializadas y Benchmarking (`token_scaling_benchmark`, `compare_models`)**: Pruebas automáticas de velocidad de tokens, rendimiento (tokens/seg), latencia, costo estimado y comparación simultánea lado a lado de modelos.
7. **Persistencia y Exportación (`save`, `load_model`, `export`)**: Exportación directa a `.mbbo`, `ONNX` y `TorchScript`.

---

## ⚡ Instalación Rápida

```bash
pip install pymbbo
```

O desde el código fuente:

```bash
git clone https://github.com/pymbbo/pymbbo.git
cd pymbbo
pip install -e .
```

---

## 💡 Ejemplo Rápido de Uso

```python
import numpy as np
from pymbbo import Config, load_dataset, build_model, EarlyStopping, ModelCheckpoint

# 1. Configurar Hiperparámetros
config = Config(learning_rate=0.001, batch_size=32, epochs=10)

# 2. Cargar y Preparar Datos
X = np.random.randn(500, 10).astype(np.float32)
Y = np.random.randint(0, 2, (500, 1)).astype(np.float32)
dataset = load_dataset((X, Y))
train_ds, val_ds, test_ds = dataset.split(train=0.8, val=0.1, test=0.1)

# 3. Construir el Modelo (Usando arquitectura incorporada 'mlp')
model = build_model("mlp", input_dim=10, hidden_units=[64, 32], output_dim=1)
model.summary()

# 4. Compilar y Entrenar
model.compile(optimizer="adam", loss_function="bce", metrics=["accuracy"])

callbacks = [
    EarlyStopping(patience=3),
    ModelCheckpoint("best_model.mbbo")
]

model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=callbacks)

# 5. Inferencia y Evaluación
predicciones = model.predict(test_ds)
reporte = model.evaluate(test_ds)
print("Resultado Evaluación:", reporte)

# 6. Guardar y Exportar
model.save("modelo_final.mbbo")
model.export("modelo.onnx", format="onnx")
```

---

## 📖 Documentación Completa

Para acceder al manual completo de desarrollador, explicación detallada de cada módulo y la guía paso a paso para crear plugins de arquitectura, consulta [DOCS.md](DOCS.md).

---

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Consulta el archivo [LICENSE](LICENSE) para más detalles.
