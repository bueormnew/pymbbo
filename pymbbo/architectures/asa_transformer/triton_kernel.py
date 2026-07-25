"""
ASA Pass-2 Triton kernel — fallback/reference backend.

NOTE: In asa_attention.py, Pass-2 already dispatches to FlashAttention-2
via F.scaled_dot_product_attention (SDPA) by reshaping to (B·H·N, 1, A).
This module is kept as:
  1. A reference / benchmark against SDPA.
  2. A future entry point for custom Triton kernels that outperform SDPA
     on sparse (variable-A) workloads.

The public API (asa_pass2_attention) is preserved for compatibility.
"""
import math
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Triton availability check
# ---------------------------------------------------------------------------
TRITON_AVAILABLE = False
try:
    if torch.cuda.is_available():
        import triton
        import triton.language as tl
        TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# ---------------------------------------------------------------------------
# Triton kernel (reference implementation, used only when explicitly enabled)
# ---------------------------------------------------------------------------
if TRITON_AVAILABLE:
    @triton.jit
    def _asa_pass2_kernel(
        Q_ptr, K_ptr, V_ptr, Out_ptr,
        stride_qb, stride_qh, stride_qn, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_ka, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_va, stride_vd,
        stride_ob, stride_oh, stride_on, stride_od,
        scale,
        B, H, N, A, D: tl.constexpr,
    ):
        """
        One Triton program per (batch, head, position) triple.
        Computes softmax-weighted sum of A gathered value vectors.
        """
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_n = tl.program_id(2)

        d_off = tl.arange(0, D)

        # Load query vector  (D,)
        q_base = pid_b * stride_qb + pid_h * stride_qh + pid_n * stride_qn
        q_vec  = tl.load(Q_ptr + q_base + d_off * stride_qd)

        # Compute QK scores for all A gathered keys
        MAX_A = 1024  # kernel constraint
        a_off    = tl.arange(0, MAX_A)
        mask_a   = a_off < A
        scores   = tl.full([MAX_A], -1e9, dtype=tl.float32)

        for a in range(A):
            k_base = (pid_b * stride_kb + pid_h * stride_kh
                      + pid_n * stride_kn + a * stride_ka)
            k_vec  = tl.load(K_ptr + k_base + d_off * stride_kd)
            dot    = tl.sum(q_vec * k_vec) * scale
            scores = tl.where(a_off == a, dot, scores)

        # Numerically stable softmax
        max_s  = tl.max(tl.where(mask_a, scores, -1e9), axis=0)
        exp_s  = tl.where(mask_a, tl.exp(scores - max_s), 0.0)
        sum_e  = tl.sum(exp_s, axis=0)
        w      = exp_s / (sum_e + 1e-6)

        # Weighted sum of value vectors
        out_vec = tl.zeros([D], dtype=tl.float32)
        for a in range(A):
            w_a    = tl.sum(tl.where(a_off == a, w, 0.0))
            v_base = (pid_b * stride_vb + pid_h * stride_vh
                      + pid_n * stride_vn + a * stride_va)
            v_vec  = tl.load(V_ptr + v_base + d_off * stride_vd)
            out_vec += w_a * v_vec

        o_base = pid_b * stride_ob + pid_h * stride_oh + pid_n * stride_on
        tl.store(Out_ptr + o_base + d_off * stride_od, out_vec)


# ---------------------------------------------------------------------------
# PyTorch vectorised fallback
# ---------------------------------------------------------------------------
def _pytorch_pass2(
    q:         torch.Tensor,   # (B, H, N, D)
    k_gathered: torch.Tensor,  # (B, H, N, A, D)
    v_gathered: torch.Tensor,  # (B, H, N, A, D)
    scale: float,
) -> torch.Tensor:             # (B, H, N, D)
    """Vectorised fallback using element-wise multiply + reduce."""
    scores  = (q.unsqueeze(-2) * k_gathered).sum(-1) * scale  # (B,H,N,A)
    weights = F.softmax(scores, dim=-1)
    return (weights.unsqueeze(-1) * v_gathered).sum(-2)


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------
def asa_pass2_attention(
    q:         torch.Tensor,
    k_gathered: torch.Tensor,
    v_gathered: torch.Tensor,
    scale: float = None,
    force_triton: bool = False,
) -> torch.Tensor:
    """
    Pass-2 sparse attention dispatcher.

    Priority:
      1. Triton kernel  — only when explicitly requested via force_triton=True
                          and CUDA + Triton are available.
      2. SDPA reshape   — FlashAttention-2 backend via F.scaled_dot_product_attention.
                          Reshapes (B,H,N,A,D) → (B·H·N, 1, A) per-position flash.
      3. PyTorch vector — CPU or when SDPA unavailable.

    NOTE: asa_attention.py calls _pass2_attention() directly (which already
    follows priorities 2 and 3).  This function is exposed for external use
    or explicit Triton benchmarking.
    """
    B, H, N, D = q.shape
    A = k_gathered.shape[-2]
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    # Option 1: Triton (explicit request only)
    if force_triton and TRITON_AVAILABLE and q.is_cuda and D in (16, 32, 64, 128) and A <= 1024:
        out  = torch.empty_like(q)
        grid = (B, H, N)
        _asa_pass2_kernel[grid](
            q, k_gathered, v_gathered, out,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k_gathered.stride(0), k_gathered.stride(1), k_gathered.stride(2),
            k_gathered.stride(3), k_gathered.stride(4),
            v_gathered.stride(0), v_gathered.stride(1), v_gathered.stride(2),
            v_gathered.stride(3), v_gathered.stride(4),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            scale, B, H, N, A, D,
        )
        return out

    # Option 2: SDPA (FlashAttention-2) — reshape to (B·H·N, 1, A)
    if hasattr(F, "scaled_dot_product_attention") and q.is_cuda:
        q_s = q.reshape(B * H * N, 1, D)
        k_s = k_gathered.reshape(B * H * N, A, D)
        v_s = v_gathered.reshape(B * H * N, A, D)
        return F.scaled_dot_product_attention(q_s, k_s, v_s).reshape(B, H, N, D)

    # Option 3: CPU / no-SDPA vectorised fallback
    return _pytorch_pass2(q, k_gathered, v_gathered, scale)
