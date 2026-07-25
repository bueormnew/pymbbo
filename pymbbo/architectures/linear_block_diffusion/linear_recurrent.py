import sys
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


def _chunked_stable_scan(
    gate_x: torch.Tensor,
    proj_x: torch.Tensor,
    h_init: torch.Tensor,
    reverse: bool = False,
    chunk_size: int = 32
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Chunked Stable Parallel Linear Recurrent Scan.

    Replaces the O(T) Python-level sequential loop with O(T/chunk_size) iterations,
    where each chunk of `chunk_size` steps is solved via a closed-form vectorized
    cumsum formula — fully parallel on GPU, no per-token Python overhead within chunks.

    The recurrence h_t = g_t * h_{t-1} + (1 - g_t) * v_t admits the closed form:

        h_t = exp(S_t) * h_0 + sum_{j=1}^{t} exp(S_t - S_j) * v_j

    where S_t = cumsum(log g_k, k=1..t).  exp(S_t - S_j) = prod_{k=j+1}^t g_k ∈ (0,1],
    so it never overflows.  The cross-term h_0 * exp(S_t) is also safe because S_t ≤ 0.

    ⚠️ FLOAT32 FORCED INTERNALLY:
    With chunk_size=32 and typical gate≈0.5, exp(-S) reaches exp(22) ≈ 3.6e9.
    float32 max ≈ 3.4e38 → SAFE.  float16 max ≈ 65504 → OVERFLOW → NaN.
    We always compute in float32 and cast output back to the caller's dtype.

    Args:
        gate_x:     (B, T, D) sigmoid gate values in (0, 1)
        proj_x:     (B, T, D) projected input values
        h_init:     (B, D)    initial hidden state
        reverse:    If True, scan from right-to-left
        chunk_size: Number of steps to vectorise at once (default 32)

    Returns:
        hidden_states: (B, T, D)  — same dtype as inputs
        h_last:        (B, D)     — same dtype as inputs
    """
    orig_dtype = gate_x.dtype

    # Always compute in float32 — see docstring for why
    gate_x = gate_x.float()
    proj_x = proj_x.float()
    h      = h_init.float()

    B, T, D = gate_x.shape

    if reverse:
        gate_x = gate_x.flip(1)
        proj_x = proj_x.flip(1)

    hidden_states = torch.empty(B, T, D, dtype=torch.float32, device=gate_x.device)
    num_chunks = (T + chunk_size - 1) // chunk_size

    for c in range(num_chunks):
        s = c * chunk_size
        e = min(s + chunk_size, T)

        g = gate_x[:, s:e, :]               # (B, L, D)  gates ∈ (0,1)
        v = (1.0 - g) * proj_x[:, s:e, :]  # (B, L, D)  effective input

        # S_t = cumsum(log g_k) within chunk — always ≤ 0
        log_g = torch.log(g.clamp(min=1e-6))
        S     = torch.cumsum(log_g, dim=1)   # (B, L, D)

        exp_S     = torch.exp(S)             # (B, L, D)  ∈ (0, 1] — safe
        exp_neg_S = torch.exp(-S)            # (B, L, D)  ∈ [1, 3.6e9] — safe in fp32

        # h_0 contribution: exp(S_t) * h_running
        h_from_init = exp_S * h.unsqueeze(1)           # (B, L, D)

        # Input contribution: exp(S_t) * cumsum_j(v_j * exp(-S_j))
        cum_weighted   = torch.cumsum(v * exp_neg_S, dim=1)  # (B, L, D)
        h_from_inputs  = exp_S * cum_weighted                 # (B, L, D)

        chunk_out = h_from_init + h_from_inputs   # (B, L, D)
        hidden_states[:, s:e, :] = chunk_out
        h = chunk_out[:, -1, :]                   # carry hidden state forward

    if reverse:
        hidden_states = hidden_states.flip(1)

    # Cast back to original dtype (fp16 if AMP was active)
    return hidden_states.to(orig_dtype), h.to(orig_dtype)



class LinearRecurrentLayer(nn.Module):
    """
    Attention-Free Linear Recurrent Layer O(N) complexity.
    Uses chunked-parallel stable gated recurrence — no Python per-token loop.

    Formula:
      g_t = sigmoid(W_g * x_t + b_g)
      h_t = g_t * h_{t-1} + (1 - g_t) * (W_in * x_t)
      y_t = W_out * h_t
    """
    def __init__(self, d_model: int, chunk_size: int = 32):
        super().__init__()
        self.d_model = d_model
        self.chunk_size = chunk_size
        self.w_in = nn.Linear(d_model, d_model, bias=False)
        # Gate has bias initialized to +2.0 so initial g_t ≈ sigmoid(2.0) = 0.88 (long context memory)
        self.w_gate = nn.Linear(d_model, d_model, bias=True)
        nn.init.constant_(self.w_gate.bias, 2.0)
        self.w_out = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        h_prev: Optional[torch.Tensor] = None,
        reverse: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:      (batch_size, seq_len, d_model)
            h_prev: (batch_size, d_model) or None
            reverse: Process right-to-left (bidirectional backward pass)
        Returns:
            output: (batch_size, seq_len, d_model)
            h_last: (batch_size, d_model)
        """
        B, T, _ = x.shape
        h_init = (
            torch.zeros(B, self.d_model, device=x.device, dtype=x.dtype)
            if h_prev is None else h_prev
        )

        proj_x = self.w_in(x)
        gate_x = torch.sigmoid(self.w_gate(x))

        hidden_states, h_last = _chunked_stable_scan(
            gate_x, proj_x, h_init, reverse=reverse, chunk_size=self.chunk_size
        )
        output = self.w_out(hidden_states)
        return output, h_last


class BidirectionalLinearRecurrentBlock(nn.Module):
    """
    Bidirectional Linear Recurrent Block with SwiGLU FFN & RMSNorm.
    Processes sequences bidirectionally in O(N) time with chunked parallel scan.
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
            x: (batch_size, seq_len, d_model)
        Returns:
            (batch_size, seq_len, d_model)
        """
        normed_x = self.norm1(x)
        fwd_out, _ = self.fwd_recurrent(normed_x, reverse=False)
        bwd_out, _ = self.bwd_recurrent(normed_x, reverse=True)

        recurrent_out = self.combine_proj(torch.cat([fwd_out, bwd_out], dim=-1))
        x = x + self.dropout(recurrent_out)

        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x
