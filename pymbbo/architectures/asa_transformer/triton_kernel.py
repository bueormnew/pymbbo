import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Check Triton and CUDA availability
TRITON_AVAILABLE = False
try:
    if torch.cuda.is_available():
        import triton
        import triton.language as tl
        TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:
    @triton.jit
    def _asa_pass2_kernel(
        Q_ptr, K_gathered_ptr, V_gathered_ptr, Out_ptr,
        stride_qb, stride_qh, stride_qn, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_ka, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_va, stride_vd,
        stride_ob, stride_oh, stride_on, stride_od,
        scale,
        B, H, N, A, D: tl.constexpr
    ):
        """
        Triton kernel for fused Pass-2 sparse attention over gathered K and V tokens.
        Each thread block processes a single (batch, head, position) tuple.
        """
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_n = tl.program_id(2)

        # Offsets
        d_offsets = tl.arange(0, D)
        q_ptr = Q_ptr + pid_b * stride_qb + pid_h * stride_qh + pid_n * stride_qn + d_offsets * stride_qd
        q_vec = tl.load(q_ptr)

        # We will compute dot products against A gathered keys, find max for numerical stability, compute exp, sum, and weighted V
        # Load gathered keys: shape (A, D)
        a_offsets = tl.arange(0, 1024) # Process up to 1024 tokens block size
        mask_a = a_offsets < A

        # Compute scores: score_a = sum_d (q_d * k_{a, d}) * scale
        # We process in loops or vector registers
        max_score = -1e9
        
        # Pass 1: Dot products & Max finding
        scores = tl.zeros([1024], dtype=tl.float32) + (-1e9)
        for a_idx in range(A):
            k_ptr = K_gathered_ptr + pid_b * stride_kb + pid_h * stride_kh + pid_n * stride_kn + a_idx * stride_ka + d_offsets * stride_kd
            k_vec = tl.load(k_ptr)
            dot = tl.sum(q_vec * k_vec) * scale
            scores = tl.where(a_offsets == a_idx, dot, scores)

        # Compute Max & Exp & Sum
        max_score = tl.max(tl.where(mask_a, scores, -1e9), axis=0)
        exp_scores = tl.exp(scores - max_score)
        exp_scores = tl.where(mask_a, exp_scores, 0.0)
        sum_exp = tl.sum(exp_scores, axis=0)
        weights = exp_scores / (sum_exp + 1e-6)

        # Pass 2: Weighted sum over V
        out_vec = tl.zeros([D], dtype=tl.float32)
        for a_idx in range(A):
            w = tl.sum(tl.where(a_offsets == a_idx, weights, 0.0))
            v_ptr = V_gathered_ptr + pid_b * stride_vb + pid_h * stride_vh + pid_n * stride_vn + a_idx * stride_va + d_offsets * stride_vd
            v_vec = tl.load(v_ptr)
            out_vec += w * v_vec

        out_ptr = Out_ptr + pid_b * stride_ob + pid_h * stride_oh + pid_n * stride_on + d_offsets * stride_od
        tl.store(out_ptr, out_vec)


def pytorch_gather_attention(
    q: torch.Tensor,
    k_gathered: torch.Tensor,
    v_gathered: torch.Tensor,
    scale: float
) -> torch.Tensor:
    """
    High-performance PyTorch vectorized implementation of Pass 2 renormalized attention.

    Args:
        q: Query tensor of shape (batch, heads, seq_len, head_dim)
        k_gathered: Gathered Key tensor of shape (batch, heads, seq_len, num_selected, head_dim)
        v_gathered: Gathered Value tensor of shape (batch, heads, seq_len, num_selected, head_dim)
        scale: Scaling factor 1.0 / sqrt(head_dim)

    Returns:
        output: Attention output tensor of shape (batch, heads, seq_len, head_dim)
    """
    # q: (B, H, N, D) -> (B, H, N, 1, D)
    # k_gathered: (B, H, N, A, D)
    # scores: (B, H, N, A)
    scores = (q.unsqueeze(-2) * k_gathered).sum(dim=-1) * scale
    weights = F.softmax(scores, dim=-1) # (B, H, N, A)
    
    # output: (B, H, N, D)
    output = (weights.unsqueeze(-1) * v_gathered).sum(dim=-2)
    return output


def asa_pass2_attention(
    q: torch.Tensor,
    k_gathered: torch.Tensor,
    v_gathered: torch.Tensor,
    scale: float = None
) -> torch.Tensor:
    """
    Pass 2 Renormalized Sparse Attention dispatcher.
    Uses Triton GPU kernel when running on CUDA with Triton available and head_dim <= 128,
    otherwise uses optimized PyTorch vectorized implementation.
    """
    B, H, N, D = q.shape
    A = k_gathered.shape[-2]
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    if (
        TRITON_AVAILABLE 
        and q.is_cuda 
        and D in [16, 32, 64, 128] 
        and A <= 1024
    ):
        out = torch.empty_like(q)
        grid = (B, H, N)
        _asa_pass2_kernel[grid](
            q, k_gathered, v_gathered, out,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k_gathered.stride(0), k_gathered.stride(1), k_gathered.stride(2), k_gathered.stride(3), k_gathered.stride(4),
            v_gathered.stride(0), v_gathered.stride(1), v_gathered.stride(2), v_gathered.stride(3), v_gathered.stride(4),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            scale,
            B, H, N, A, D
        )
        return out

    return pytorch_gather_attention(q, k_gathered, v_gathered, scale)
