"""Capa completa CausalMatrixMergeLayerV2 — pre-norm merge + residual + pre-norm MLP + residual.

Implementa la estructura de una capa individual del modelo CMM v2:
    1. RMSNorm → CausalMatrixMergeV2 → + Residual
    2. RMSNorm → SwiGLUMLP → + Residual

La arquitectura pre-norm normaliza antes de cada sub-bloque y aplica
conexiones residuales sumando la entrada de cada sub-bloque a su salida.
Esto favorece la estabilidad del entrenamiento en redes profundas.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .config import CausalMatrixMergeV2Config
from .merge_v2 import CausalMatrixMergeV2
from .mlp import SwiGLUMLP
from pymbbo.architectures.causal_matrix_merge.state import MergeState


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization con escala aprendida.

    Normaliza el tensor de entrada por su RMS (raíz cuadrada de la media
    de los cuadrados) y aplica una escala aprendida por dimensión.

    A diferencia de LayerNorm, no centra (no resta la media), lo cual
    reduce el costo computacional y preserva la magnitud relativa.

    Args:
        dim: Dimensión de la última axis del tensor de entrada.
        eps: Epsilon para estabilidad numérica (evitar división por cero).
    """

    def __init__(self, dim: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Aplica RMSNorm al tensor de entrada.

        Args:
            x: Tensor de forma arbitraria donde la última dimensión es `dim`.

        Returns:
            Tensor normalizado con la misma forma que la entrada.
        """
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True).clamp_min(self.eps))
        return x * rms * self.scale


class CausalMatrixMergeLayerV2(nn.Module):
    """Capa completa del modelo CMM v2: pre-norm merge + residual + pre-norm MLP + residual.

    Cada capa procesa tokens siguiendo la secuencia:
        1. Sub-bloque Merge: RMSNorm(x) → CausalMatrixMergeV2 → + x (residual)
        2. Sub-bloque MLP: RMSNorm(x) → SwiGLUMLP → + x (residual)

    La arquitectura pre-norm aplica normalización antes de cada sub-bloque,
    lo cual mejora la estabilidad del entrenamiento y permite redes más profundas.

    Args:
        config: CausalMatrixMergeV2Config con todos los hiperparámetros.
    """

    def __init__(self, config: CausalMatrixMergeV2Config) -> None:
        super().__init__()
        self.config = config

        # Sub-bloque 1: Merge con pre-norm
        self.merge_norm = RMSNorm(config.model_dim)
        self.merge = CausalMatrixMergeV2(config)

        # Sub-bloque 2: MLP con pre-norm
        self.mlp_norm = RMSNorm(config.model_dim)
        self.mlp = SwiGLUMLP(config.model_dim, config.ffn_mult, config.dropout)

    def forward(
        self, x: torch.Tensor, state: MergeState
    ) -> Tuple[torch.Tensor, MergeState]:
        """Procesa un token individual a través de la capa completa.

        Aplica la estructura pre-norm con residuales:
            1. residual + merge(norm(x))
            2. residual + mlp(norm(x))

        Args:
            x: Tensor [B, model_dim] — embedding del token actual.
            state: MergeState — estado de memoria previo (batched).

        Returns:
            Tuple de (output [B, model_dim], nuevo MergeState).
        """
        # Sub-bloque 1: Merge
        residual = x
        x_normed = self.merge_norm(x)
        x_merged, state = self.merge(x_normed, state)
        x = residual + x_merged

        # Sub-bloque 2: MLP
        residual = x
        x_normed = self.mlp_norm(x)
        x_mlp = self.mlp(x_normed)
        x = residual + x_mlp

        return x, state

    def forward_sequence(
        self, x: torch.Tensor, state: Optional[MergeState] = None
    ) -> Tuple[torch.Tensor, MergeState]:
        """Procesa secuencia completa en PARALELO (no token-a-token).

        Usa el parallel scan del merge block para procesar toda la secuencia
        de forma eficiente en GPU. ~20-50x mas rapido que el loop secuencial.

        Args:
            x: Tensor [B, T, model_dim] — secuencia de embeddings.
            state: Estado inicial. Si None, se crea uno nuevo.

        Returns:
            Tuple de (outputs [B, T, model_dim], estado final MergeState).
        """
        B, T, _ = x.shape

        if state is None:
            state = self.merge.init_state(B, device=x.device, dtype=x.dtype)

        # Sub-bloque 1: Merge (paralelo via scan)
        residual = x
        x_normed = self.merge_norm(x)  # [B, T, model_dim] - RMSNorm opera sobre ultima dim
        x_merged, state = self.merge.forward_sequence(x_normed, state)
        x = residual + x_merged

        # Sub-bloque 2: MLP (ya es paralelo, opera sobre ultima dim)
        residual = x
        x_normed = self.mlp_norm(x)
        x_mlp = self.mlp(x_normed)
        x = residual + x_mlp

        return x, state
