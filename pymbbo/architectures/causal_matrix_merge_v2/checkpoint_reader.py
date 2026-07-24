"""Selección aprendida de checkpoints para Causal Matrix Merge v2.

Este módulo implementa el CheckpointSelector, que reemplaza los pesos fijos
linspace de v1 por atención aprendida query-dependent sobre el banco de
checkpoints. Esto permite al modelo recuperar información lejana relevante
de forma adaptativa según el contexto actual.

El módulo mantiene complejidad O(1) por token (solo depende de num_checkpoints
que es fijo) y es completamente diferenciable para entrenamiento por
backpropagation.

Algoritmo:
    1. Para cada checkpoint k: leer con atención sobre sus S slots
       usando key_proj y value_proj compartidos del merge.
    2. Computar summary de cada checkpoint (mean sobre slots).
    3. Proyectar summaries con checkpoint_key_proj.
    4. Computar relevancia: einsum('bd,bkd->bk', query, projected_summaries).
    5. Aplicar softmax → pesos aprendidos [B, K].
    6. Weighted sum de las lecturas → [B, model_dim].
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CheckpointSelector(nn.Module):
    """Selección aprendida de checkpoints, query-dependent.

    Reemplaza los pesos fijos linspace de v1 por atención aprendida
    sobre el banco de checkpoints. Los pesos de selección dependen del
    query actual, permitiendo al modelo asignar peso alto a cualquier
    checkpoint del banco independientemente de su antigüedad.

    Args:
        state_dim: Dimensión de cada slot de memoria (state_dim del config).
        num_checkpoints: Número de checkpoints en el banco (K).
    """

    def __init__(self, state_dim: int, num_checkpoints: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.num_checkpoints = num_checkpoints

        # Proyección aprendida para generar keys de checkpoint
        # Mapea el summary de cada checkpoint a un espacio de keys
        # para computar relevancia contra el query actual.
        self.checkpoint_key_proj = nn.Linear(state_dim, state_dim, bias=False)

    def forward(
        self,
        query: torch.Tensor,
        checkpoints: torch.Tensor,
        key_proj: nn.Linear,
        value_proj: nn.Linear,
    ) -> torch.Tensor:
        """Lectura ponderada aprendida del banco de checkpoints.

        Para cada checkpoint, lee con atención sobre sus slots usando las
        proyecciones compartidas del merge (key_proj, value_proj), computa
        la relevancia respecto al query actual, y retorna una suma ponderada
        de las lecturas.

        Cuando todos los checkpoints son cero (estado inicial), el softmax
        sobre scores de cero produce distribución uniforme, resultando en
        una lectura estable sin NaN ni desestabilización.

        Args:
            query: Tensor [B, state_dim] — query del token actual
                (ya proyectado por query_proj del merge).
            checkpoints: Tensor [B, K, S, state_dim] — banco de checkpoints.
            key_proj: Módulo Linear compartido del merge para proyectar keys
                de los slots (state_dim → state_dim).
            value_proj: Módulo Linear compartido del merge para proyectar values
                de los slots (state_dim → model_dim).

        Returns:
            ctx_checkpoints: Tensor [B, model_dim] — lectura ponderada de
                todos los checkpoints del banco.
        """
        B, K, S, _ = checkpoints.shape
        scale = math.sqrt(self.state_dim)

        # === Paso 1: Leer cada checkpoint con atención sobre sus slots ===
        # Reshape para procesar todos los checkpoints en batch:
        # [B, K, S, state_dim] → [B*K, S, state_dim]
        ckpt_flat = checkpoints.reshape(B * K, S, self.state_dim)

        # Keys y values de los slots de cada checkpoint
        ckpt_keys = key_proj(ckpt_flat)       # [B*K, S, state_dim]
        ckpt_vals = value_proj(ckpt_flat)     # [B*K, S, model_dim]

        # Expandir query para cada checkpoint: [B, state_dim] → [B*K, state_dim]
        query_expanded = query.unsqueeze(1).expand(B, K, self.state_dim)
        query_flat = query_expanded.reshape(B * K, self.state_dim)

        # Atención sobre slots de cada checkpoint
        attn_scores = torch.einsum("bd,bsd->bs", query_flat, ckpt_keys) / scale
        attn_weights = F.softmax(attn_scores, dim=-1).unsqueeze(-1)  # [B*K, S, 1]

        # Lectura ponderada de cada checkpoint: [B*K, model_dim]
        ckpt_reads = (attn_weights * ckpt_vals).sum(dim=1)

        # Reshape back: [B*K, model_dim] → [B, K, model_dim]
        model_dim = ckpt_vals.shape[-1]
        ckpt_reads = ckpt_reads.view(B, K, model_dim)

        # === Paso 2: Computar summary de cada checkpoint ===
        # Mean sobre slots como representación resumida: [B, K, state_dim]
        ckpt_summaries = checkpoints.mean(dim=2)  # [B, K, state_dim]

        # === Paso 3: Proyectar summaries con checkpoint_key_proj ===
        ckpt_projected = self.checkpoint_key_proj(ckpt_summaries)  # [B, K, state_dim]

        # === Paso 4: Computar relevancia query vs projected summaries ===
        # einsum('bd,bkd->bk'): similitud entre query y cada checkpoint
        relevance_scores = torch.einsum(
            "bd,bkd->bk", query, ckpt_projected
        ) / scale  # [B, K]

        # === Paso 5: softmax → pesos aprendidos ===
        # Cuando checkpoints son cero → summaries cero → projected cero →
        # scores cero → softmax produce 1/K uniforme → estable
        checkpoint_weights = F.softmax(relevance_scores, dim=-1)  # [B, K]

        # === Paso 6: Weighted sum de las lecturas ===
        # [B, K, 1] * [B, K, model_dim] → sum → [B, model_dim]
        ctx_checkpoints = (
            checkpoint_weights.unsqueeze(-1) * ckpt_reads
        ).sum(dim=1)  # [B, model_dim]

        return ctx_checkpoints
