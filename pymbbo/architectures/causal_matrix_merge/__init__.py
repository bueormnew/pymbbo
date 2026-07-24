"""
Causal Matrix Merge v1 — Arquitectura de memoria matricial causal para pymbbo.

Esta arquitectura NO es un transformer. Utiliza una memoria matricial de tamaño
fijo que comprime contexto infinito mediante una regla de actualización afín.
No hay mecanismo de atención sobre el historial de tokens — cada token actualiza
una matriz fija y lee de ella, logrando complejidad O(1) por token generado.

Componentes principales:
    - CausalMatrixMergeConfig: Dataclass inmutable con todos los hiperparámetros.
    - MergeState: Estado inmutable de memoria (slots + checkpoints + normalizador).
    - CausalMatrixMerge: Bloque core de merge matricial causal (una capa).
    - CausalMatrixMergeModel: Modelo de lenguaje completo (N capas apiladas).

Ejemplo de uso completo
-----------------------

.. code-block:: python

    import torch
    from pymbbo.models.factory import build_model
    from pymbbo.models.persistence import save_model, load_model

    # ─── 1. Instanciar el modelo vía build_model ───────────────────────────
    model = build_model(
        "causal_matrix_merge",
        vocab_size=5000,
        model_dim=256,
        state_dim=128,
        num_slots=8,
        num_layers=4,
        num_checkpoints=4,
        checkpoint_stride=16,
        write_rank=4,
        dropout=0.1,
        use_residual_gate=True,
        max_context=2048,
    )

    # Ver resumen de parámetros
    model.summary()

    # ─── 2. Compilar y entrenar ────────────────────────────────────────────
    model.compile(
        optimizer="adamw",
        loss_function="cross_entropy",
        learning_rate=3e-4,
    )

    # Datos de ejemplo: secuencias de token IDs [batch, seq_len]
    train_tokens = torch.randint(0, 5000, (64, 128))
    # Target: siguiente token (shift a la izquierda)
    train_targets = torch.randint(0, 5000, (64, 128))

    # Entrenamiento manual (loop simple)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(5):
        optimizer.zero_grad()
        logits = model(train_tokens)               # [64, 128, 5000]
        loss = loss_fn(
            logits.view(-1, 5000),
            train_targets.view(-1),
        )
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1} — loss: {loss.item():.4f}")

    # ─── 3. Generación autoregresiva ──────────────────────────────────────
    model.eval()
    prompt = torch.tensor([[1, 42, 100, 7]])  # [1, 4] — un prompt de 4 tokens

    # Acceder a la arquitectura para usar generate()
    generated = model.architecture.generate(
        prompt_ids=prompt,
        max_new_tokens=50,
        temperature=0.8,
        top_k=40,
        top_p=0.95,
    )
    print(f"Tokens generados: {generated.shape}")  # [1, 54]

    # ─── 4. Guardar y cargar el modelo ────────────────────────────────────
    save_model(model, "mi_modelo_cmm.pt")

    # Reconstruir el modelo completo desde disco
    model_cargado = load_model("mi_modelo_cmm.pt")
    model_cargado.eval()

    # Verificar que genera igual
    generated2 = model_cargado.architecture.generate(
        prompt_ids=prompt,
        max_new_tokens=10,
        temperature=1.0,
    )
    print(f"Modelo cargado genera: {generated2.shape}")
"""

from .config import CausalMatrixMergeConfig
from .state import MergeState
from .merge import CausalMatrixMerge
from .model import CausalMatrixMergeModel

__all__ = [
    "CausalMatrixMergeConfig",
    "MergeState",
    "CausalMatrixMerge",
    "CausalMatrixMergeModel",
]
