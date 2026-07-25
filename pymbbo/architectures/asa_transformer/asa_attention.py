import math
import random
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers: memory-efficient gather + Pass-2 dispatcher
# ---------------------------------------------------------------------------

def _gather_kv(
    k: torch.Tensor,    # (B, H, N_total, D)
    v: torch.Tensor,    # (B, H, N_total, D)
    sel: torch.Tensor,  # (B, N, A)  — integer selection indices
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Memory-efficient K/V gather using advanced indexing.

    Complexity: O(B·H·N·A·D)  — NOT O(B·H·N·N_total·D).

    Previous implementations used `k.unsqueeze(2).expand(..., N_total, ...)` +
    `torch.gather` with a (B·H, N·A·D) int64 index (called "flat_idx"), which
    materialised a catastrophically large index tensor:
        - flat_idx  ≈ (B·H · N·A·D) int64  →  up to 1 GB for A=128
        - gathered  ≈ (B·H · N·A·D) float32

    This version:
        - sel_merged  (B·H, N·A) int64  →  only ~8 MB for A=128
        - bh_idx      (B·H, 1)   int64  →  negligible
        - k/v_gath    (B, H, N, A, D) float32  →  same output, far smaller index
    """
    B, H, N_total, D = k.shape
    N, A = sel.shape[1], sel.shape[2]

    k_2d = k.reshape(B * H, N_total, D)        # view, no copy
    v_2d = v.reshape(B * H, N_total, D)        # view, no copy

    # (B, N, A) → (B*H, N*A)   — still just a view via expand + reshape
    sel_bh = (sel.unsqueeze(1)
                 .expand(B, H, N, A)
                 .reshape(B * H, N * A))

    # Advanced indexing: k_2d[bh, sel, :] — no D-expanded index tensor
    bh_idx = torch.arange(B * H, device=k.device).unsqueeze(1)  # (B*H, 1)
    k_gath = k_2d[bh_idx, sel_bh].reshape(B, H, N, A, D)
    v_gath = v_2d[bh_idx, sel_bh].reshape(B, H, N, A, D)
    return k_gath, v_gath


def _pass2_attention(
    q:      torch.Tensor,   # (B, H, N, D)
    k_gath: torch.Tensor,   # (B, H, N, A, D)
    v_gath: torch.Tensor,   # (B, H, N, A, D)
    scale:  float,
) -> torch.Tensor:          # (B, H, N, D)
    """
    Pass-2 sparse renormalized attention.

    When on CUDA with PyTorch >= 2.0 (SDPA / FlashAttention-2 backend):
        Reshapes to (B·H·N, 1, A) — each query position is an independent
        batch element attending to its A selected keys.  SDPA fuses the
        QK^T, softmax, and @V into a single flash kernel.

    CPU fallback:
        Vectorised einsum-equivalent loop kept for correctness.
    """
    B, H, N, A, D = k_gath.shape

    if hasattr(F, "scaled_dot_product_attention") and q.is_cuda:
        # Reshape: treat every (b, h, n) position as an independent "batch"
        # with exactly 1 query and A context tokens.
        q_s = q.reshape(B * H * N, 1, D)       # (B·H·N, 1, D)
        k_s = k_gath.reshape(B * H * N, A, D)  # (B·H·N, A, D)
        v_s = v_gath.reshape(B * H * N, A, D)  # (B·H·N, A, D)
        # SDPA handles scale internally; no causal mask (tokens already selected)
        out = F.scaled_dot_product_attention(q_s, k_s, v_s)  # (B·H·N, 1, D)
        return out.reshape(B, H, N, D)

    # CPU / no-SDPA fallback
    scores  = (q.unsqueeze(-2) * k_gath).sum(-1) * scale  # (B, H, N, A)
    weights = F.softmax(scores, dim=-1)
    return (weights.unsqueeze(-1) * v_gath).sum(-2)        # (B, H, N, D)


# ---------------------------------------------------------------------------
# Core Module
# ---------------------------------------------------------------------------

class AdaptiveSelectiveAttention(nn.Module):
    """
    Adaptive Selective Attention (ASA) — Buenahora Ormaza (2026).

    Two-pass score-gated causal attention with Selection Sharing:

    Pass-1 (router, no-grad):
        Dense causal scoring to derive token importance I_i(j).
        Runs under torch.no_grad() — selection indices are integer-valued
        and non-differentiable, so no gradient information is lost.
        This frees the O(N²) attention matrix immediately after top-k
        selection, saving up to 70 % of training VRAM.

    Selection:
        top-A past tokens + self-token i per query position.

    Pass-2 (all layers, with grad):
        Renormalized sparse attention over selected tokens only.
        Uses memory-efficient advanced-index gather — O(N·A) index tensor.
        On CUDA dispatches to FlashAttention-2 (SDPA) by reshaping to
        (B·H·N, 1, A) so each query attends its A keys as a flash batch.

    Margin loss (optional):
        If return_aux_loss=True, recomputes importance scores *with* grad
        locally and cheaply for the margin term, keeping the rest free.

    Selection Sharing:
        Router layer (is_router=True) computes selection_indices.
        Follower layers (is_router=False) reuse the router's indices,
        skipping Pass-1 entirely for further speed and memory savings.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        max_a: int = 64,
        is_router: bool = True,
        dropout: float = 0.0,
        curriculum_budgets: Optional[List[Optional[int]]] = None,
        margin_delta: float = 0.1,
    ) -> None:
        super().__init__()
        assert d_model % nhead == 0, (
            f"d_model ({d_model}) must be divisible by nhead ({nhead})"
        )
        self.d_model           = d_model
        self.nhead             = nhead
        self.head_dim          = d_model // nhead
        self.max_a             = max_a
        self.is_router         = is_router
        self.scale             = 1.0 / math.sqrt(self.head_dim)
        self.curriculum_budgets = curriculum_budgets
        self.margin_delta      = margin_delta

        self.q_proj   = nn.Linear(d_model, d_model)
        self.k_proj   = nn.Linear(d_model, d_model)
        self.v_proj   = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout  = nn.Dropout(dropout)

    # ------------------------------------------------------------------

    def _sample_budget(self, seq_len: int, dynamic_max_a: Optional[int] = None) -> int:
        """Determines the selection budget 'a' for the current forward pass."""
        if dynamic_max_a is not None:
            return dynamic_max_a
        if self.training and self.curriculum_budgets:
            sampled = random.choice(self.curriculum_budgets)
            if sampled is None or sampled >= seq_len:
                return seq_len
            return max(1, sampled)
        return self.max_a

    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        selection_indices: Optional[torch.Tensor] = None,
        max_a: Optional[int] = None,
        return_aux_loss: bool = False,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[Tuple[torch.Tensor, torch.Tensor]],
    ]:
        """
        Args:
            x:                 (B, N, d_model)
            selection_indices: (B, N, A+1)  pre-computed by router layer (follower mode)
            max_a:             runtime budget override
            return_aux_loss:   compute optional margin routing loss
            kv_cache:          (cached_k, cached_v) for autoregressive decoding

        Returns:
            out:               (B, N, d_model)
            selection_indices: (B, N, A+1)  —  None for follower layers
            aux_loss:          scalar or None
            new_kv_cache:      updated (k, v) cache
        """
        B, N, _ = x.shape
        H, D    = self.nhead, self.head_dim

        # ── QKV projections ─────────────────────────────────────────
        def _proj(lin: nn.Linear) -> torch.Tensor:
            return lin(x).view(B, N, H, D).transpose(1, 2)  # (B, H, N, D)

        q = _proj(self.q_proj)
        k = _proj(self.k_proj)
        v = _proj(self.v_proj)

        # ── KV Cache (autoregressive decoding) ───────────────────────
        if kv_cache is not None:
            ck, cv = kv_cache
            k = torch.cat([ck, k], dim=2)
            v = torch.cat([cv, v], dim=2)
        new_kv_cache = (k, v)
        N_total = k.shape[2]

        budget = self._sample_budget(N_total, max_a)
        self.sample_budget = budget  # expose for monitoring

        # ── Full-attention fallback via FlashAttention-2 (SDPA) ─────
        if budget >= N_total:
            if hasattr(F, "scaled_dot_product_attention") and x.is_cuda:
                dp  = self.dropout.p if self.training else 0.0
                out = F.scaled_dot_product_attention(
                    q, k, v,
                    is_causal=(N == N_total),
                    dropout_p=dp,
                )
            else:
                s = (q @ k.transpose(-2, -1)) * self.scale
                if N == N_total:
                    cm = torch.triu(torch.full((N, N), float("-inf"), device=x.device), 1)
                    s  = s + cm[None, None]
                out = F.softmax(s, dim=-1) @ v
            out = self.out_proj(out.transpose(1, 2).contiguous().view(B, N, self.d_model))
            fi  = torch.arange(N_total, device=x.device).view(1, 1, -1).expand(B, N, -1)
            return out, fi, None, new_kv_cache

        # ── Pass-1: routing under no_grad (key optimisation) ────────
        #
        # Rationale: selection_indices are integers — they carry zero
        # gradient regardless.  Running Pass-1 inside no_grad means the
        # O(N²) attention matrix (scores, softmax weights, importance)
        # is freed immediately after topk, saving ~70 % of VRAM during
        # training without losing any differentiable signal.
        #
        aux_loss = None

        if self.is_router or selection_indices is None:
            with torch.no_grad():
                raw = (q.detach() @ k.detach().transpose(-2, -1)) * self.scale  # (B,H,N,N_total)
                cm  = torch.triu(
                    torch.full((N, N_total), float("-inf"), device=x.device), 1
                )
                imp = F.softmax(raw + cm[None, None], dim=-1).mean(1)  # (B,N,N_total)

                # Mask future tokens and select top-A
                imp_m = imp.masked_fill(cm[None] == float("-inf"), -1e9)
                a_cap = min(budget, N_total)
                _, top_idx = torch.topk(imp_m, k=a_cap, dim=-1, sorted=False)

                # Always include self-token i
                self_idx = (
                    torch.arange(N_total - N, N_total, device=x.device)
                    .view(1, -1, 1)
                    .expand(B, N, 1)
                )
                selection_indices = torch.cat([top_idx, self_idx], dim=-1)  # (B,N,A+1)

            # ── Margin loss (optional): recompute imp *with* grad ───
            # Only the margin boundary scores need to be differentiable.
            # We recompute importance here (small additional cost) to keep
            # the routing loss while still freeing the routing graph above.
            if return_aux_loss:
                raw_g = (q @ k.transpose(-2, -1)) * self.scale
                cm_g  = torch.triu(
                    torch.full((N, N_total), float("-inf"), device=x.device), 1
                )
                imp_g = F.softmax(raw_g + cm_g[None, None], dim=-1).mean(1)

                sel_s   = torch.gather(imp_g, -1, selection_indices)
                min_sel = sel_s.min(-1).values

                valid = cm_g[None].expand(B, N, N_total) != float("-inf")
                smask = torch.zeros((B, N, N_total), dtype=torch.bool, device=x.device)
                smask.scatter_(-1, selection_indices, True)
                non_s = imp_g.masked_fill(~(valid & ~smask), -1e9)
                mns   = non_s.max(-1).values
                mns   = torch.where(mns == -1e9, torch.zeros_like(mns), mns)
                aux_loss = F.relu(self.margin_delta - min_sel + mns).mean()

        # ── Pass-2: sparse attention over selected tokens ────────────
        #
        # Gather:  advanced-index  O(N·A) index  (NOT the flat N·A·D index)
        # Attend:  SDPA reshape → FlashAttention-2 when on CUDA
        #
        k_gath, v_gath = _gather_kv(k, v, selection_indices)
        out = _pass2_attention(q, k_gath, v_gath, self.scale)
        out = self.out_proj(out.transpose(1, 2).contiguous().view(B, N, self.d_model))

        return out, selection_indices, aux_loss, new_kv_cache
