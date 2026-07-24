"""Paquete Causal Matrix Merge v2.

Evolución de CMM v1 con cinco mejoras: SwiGLU MLP, Sparse Routing,
Expressive Write, Learned Checkpoint Selection y Adaptive Merge.

Coexiste con v1 sin conflictos mediante registro independiente como
'causal_matrix_merge_v2'.
"""

from .config import CausalMatrixMergeV2Config
from .mlp import SwiGLUMLP
from .checkpoint_reader import CheckpointSelector
from .merge_v2 import WriteMLP, CausalMatrixMergeV2
from .layer import CausalMatrixMergeLayerV2, RMSNorm
from .model import CausalMatrixMergeModelV2  # This import activates @register_architecture

__all__ = [
    "CausalMatrixMergeV2Config",
    "SwiGLUMLP",
    "CheckpointSelector",
    "WriteMLP",
    "CausalMatrixMergeV2",
    "CausalMatrixMergeLayerV2",
    "RMSNorm",
    "CausalMatrixMergeModelV2",
]
