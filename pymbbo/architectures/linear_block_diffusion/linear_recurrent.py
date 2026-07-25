import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class SwiGLU(nn.Module):
    """SwiGLU Activation Function Feed-Forward Network."""
    def __init__(self, d_model: int, d_ff: Optional[int] = None):
        super().__init__()
        if d_ff is None:
            d_ff = int(8 / 3 * d_model)
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class LinearRecurrentLayer(nn.Module):
    """
    Attention-Free Linear Recurrent Layer O(N) complexity.
    Uses gated recurrence with a hidden state decay & linear input projection.
    
    Formula:
      g_t = sigmoid(W_g * x_t + b_g)
      h_t = g_t * h_{t-1} + (1 - g_t) * (W_in * x_t)
      y_t = W_out * h_t
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.w_in = nn.Linear(d_model, d_model)
        self.w_gate = nn.Linear(d_model, d_model)
        self.w_out = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, h_prev: Optional[torch.Tensor] = None, reverse: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            h_prev: Previous hidden state of shape (batch_size, d_model)
            reverse: If True, processes the sequence backwards (for bidirectional modeling)
        Returns:
            output: Processed sequence of shape (batch_size, seq_len, d_model)
            h_last: Last hidden state of shape (batch_size, d_model)
        """
        batch_size, seq_len, _ = x.shape
        if h_prev is None:
            h = torch.zeros(batch_size, self.d_model, device=x.device, dtype=x.dtype)
        else:
            h = h_prev

        proj_x = self.w_in(x)
        gate_x = torch.sigmoid(self.w_gate(x))

        outputs = []
        indices = range(seq_len - 1, -1, -1) if reverse else range(seq_len)

        # Pre-allocate output or collect steps
        hidden_states = [None] * seq_len
        for t in indices:
            g_t = gate_x[:, t, :]
            in_t = proj_x[:, t, :]
            h = g_t * h + (1.0 - g_t) * in_t
            hidden_states[t] = h

        stacked_h = torch.stack(hidden_states, dim=1) # (batch_size, seq_len, d_model)
        output = self.w_out(stacked_h)
        return output, h


class BidirectionalLinearRecurrentBlock(nn.Module):
    """
    Bidirectional Linear Recurrent Block with SwiGLU FFN & RMSNorm.
    Processes sequences in linear time O(N) bidirectionally inside diffusion blocks.
    """
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.fwd_recurrent = LinearRecurrentLayer(d_model)
        self.bwd_recurrent = LinearRecurrentLayer(d_model)
        self.combine_proj = nn.Linear(d_model * 2, d_model)

        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Tensor of shape (batch_size, seq_len, d_model)
        """
        normed_x = self.norm1(x)
        fwd_out, _ = self.fwd_recurrent(normed_x, reverse=False)
        bwd_out, _ = self.bwd_recurrent(normed_x, reverse=True)
        
        recurrent_out = self.combine_proj(torch.cat([fwd_out, bwd_out], dim=-1))
        x = x + self.dropout(recurrent_out)

        # FFN sub-layer
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x
