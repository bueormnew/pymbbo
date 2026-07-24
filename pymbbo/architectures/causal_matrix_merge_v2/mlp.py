"""Bloque MLP con activación SwiGLU para Causal Matrix Merge v2.

Implementa el Sub_Bloque_MLP de cada capa CMM v2. La activación SwiGLU
incrementa la capacidad representacional sin dependencia de la longitud
de secuencia (O(1) por token).

Fórmula:
    SwiGLU(x) = (x @ W_gate ⊙ silu(x @ W_up)) @ W_down

La dimensión interna se controla con el multiplicador ffn_mult:
    hidden_dim = int(model_dim * ffn_mult)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUMLP(nn.Module):
    """Bloque MLP con activación SwiGLU.

    SwiGLU(x) = (x @ W_gate ⊙ silu(x @ W_up)) @ W_down

    Todas las proyecciones lineales son sin bias. Dropout configurable
    se aplica a la salida durante entrenamiento.

    Args:
        model_dim: Dimensión del modelo (entrada y salida).
        ffn_mult: Factor multiplicador para la dimensión interna.
            hidden_dim = int(model_dim * ffn_mult).
        dropout: Probabilidad de dropout aplicada a la salida.
    """

    def __init__(self, model_dim: int, ffn_mult: float, dropout: float = 0.1) -> None:
        super().__init__()
        hidden_dim = int(model_dim * ffn_mult)

        # Proyecciones sin bias
        self.w_gate = nn.Linear(model_dim, hidden_dim, bias=False)
        self.w_up = nn.Linear(model_dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, model_dim, bias=False)

        # Dropout configurable
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Aplica SwiGLU MLP al tensor de entrada.

        Soporta entradas de forma [B, model_dim] y [B, T, model_dim].
        Las capas Linear de PyTorch operan sobre la última dimensión,
        por lo que no se requiere reshape explícito.

        Args:
            x: Tensor de forma [B, model_dim] o [B, T, model_dim].

        Returns:
            Tensor de la misma forma que la entrada.
        """
        gate = self.w_gate(x)           # [*, hidden_dim]
        up = F.silu(self.w_up(x))       # [*, hidden_dim]
        hidden = gate * up              # [*, hidden_dim]
        out = self.w_down(hidden)       # [*, model_dim]
        return self.dropout(out)
