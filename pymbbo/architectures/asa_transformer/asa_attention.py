import math
import random
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Token-level Gather and Pass-2 Helpers
# ---------------------------------------------------------------------------

def _gather_kv_token(
    k: torch.Tensor,    # (B, H, N_total, D)
    v: torch.Tensor,    # (B, H, N_total, D)
    sel: torch.Tensor,  # (B, N, A)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Token-level memory-efficient gather via advanced indexing."""
    B, H, N_total, D = k.shape
    N, A = sel.shape[1], sel.shape[2]

    k_2d = k.reshape(B * H, N_total, D)
    v_2d = v.reshape(B * H, N_total, D)

    sel_bh = sel.unsqueeze(1).expand(B, H, N, A).reshape(B * H, N * A)
    bh_idx = torch.arange(B * H, device=k.device).unsqueeze(1)
    
    k_gath = k_2d[bh_idx, sel_bh].reshape(B, H, N, A, D)
    v_gath = v_2d[bh_idx, sel_bh].reshape(B, H, N, A, D)
    return k_gath, v_gath


def _pass2_token_attention(
    q: torch.Tensor, k_gath: torch.Tensor, v_gath: torch.Tensor, scale: float
) -> torch.Tensor:
    """Token-level Pass-2 attention."""
    B, H, N, A, D = k_gath.shape
    if hasattr(F, "scaled_dot_product_attention") and q.is_cuda:
        q_s = q.reshape(B * H * N, 1, D)
        k_s = k_gath.reshape(B * H * N, A, D)
        v_s = v_gath.reshape(B * H * N, A, D)
        out = F.scaled_dot_product_attention(q_s, k_s, v_s)
        return out.reshape(B, H, N, D)

    scores  = (q.unsqueeze(-2) * k_gath).sum(-1) * scale
    weights = F.softmax(scores, dim=-1)
    return (weights.unsqueeze(-1) * v_gath).sum(-2)


# ---------------------------------------------------------------------------
# Core Adaptive Selective Attention (ASA) Module — Block & Token Capable
# ---------------------------------------------------------------------------

class AdaptiveSelectiveAttention(nn.Module):
    """
    Adaptive Selective Attention (ASA) — Buenahora Ormaza (2026).

    Supports both Block-level ASA (block_size > 1) and Token-level ASA (block_size = 1).

    Block-level ASA (Default block_size=16):
        - Pass 1 (Router): Computes block-level importance scores I_{block_i}(block_j).
          Reduces routing sequence length from N to N / C (e.g., 1024 -> 64 blocks).
        - Selection: Selects top-A_blocks key blocks + self block per query block.
        - Pass 2 (All Layers): Block-sparse attention via SDPA.
          Eliminates per-token gather overhead, achieves 16x VRAM reduction and
          runs GPU Tensor Cores at full speed.
        - Selection Sharing: Followers reuse block_selection from the router layer.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        max_a: int = 64,
        is_router: bool = True,
        block_size: int = 16,
        dropout: float = 0.0,
        curriculum_budgets: Optional[List[Optional[int]]] = None,
        margin_delta: float = 0.1,
    ) -> None:
        super().__init__()
        assert d_model % nhead == 0, f"d_model ({d_model}) must be divisible by nhead ({nhead})"
        
        self.d_model            = d_model
        self.nhead              = nhead
        self.head_dim           = d_model // nhead
        self.max_a              = max_a
        self.is_router          = is_router
        self.block_size         = block_size
        self.scale              = 1.0 / math.sqrt(self.head_dim)
        self.curriculum_budgets  = curriculum_budgets
        self.margin_delta       = margin_delta

        self.q_proj   = nn.Linear(d_model, d_model)
        self.k_proj   = nn.Linear(d_model, d_model)
        self.v_proj   = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout  = nn.Dropout(dropout)

    def _sample_budget(self, seq_len: int, dynamic_max_a: Optional[int] = None) -> int:
        if dynamic_max_a is not None:
            return dynamic_max_a
        if self.training and self.curriculum_budgets:
            sampled = random.choice(self.curriculum_budgets)
            if sampled is None or sampled >= seq_len:
                return seq_len
            return max(1, sampled)
        return self.max_a

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
        B, N, _ = x.shape
        H, D    = self.nhead, self.head_dim

        q = self.q_proj(x).view(B, N, H, D).transpose(1, 2)
        k = self.k_proj(x).view(B, N, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, N, H, D).transpose(1, 2)

        if kv_cache is not None:
            ck, cv = kv_cache
            k = torch.cat([ck, k], dim=2)
            v = torch.cat([cv, v], dim=2)
        new_kv_cache = (k, v)
        N_total = k.shape[2]

        budget = self._sample_budget(N_total, max_a)
        self.sample_budget = budget

        # ── Full-attention fallback ──────────────────────────────────
        if budget >= N_total:
            if hasattr(F, "scaled_dot_product_attention") and not return_aux_loss and x.is_cuda:
                dp  = self.dropout.p if self.training else 0.0
                out = F.scaled_dot_product_attention(
                    q, k, v, is_causal=(N == N_total), dropout_p=dp
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

        aux_loss = None
        C = self.block_size

        # ── BLOCK-ASA PATH (if C > 1 and sequence divides cleanly) ──
        if C > 1 and (N % C == 0) and (N_total % C == 0):
            N_blocks       = N // C
            N_total_blocks = N_total // C
            A_blocks       = max(1, budget // C)

            # 1. Pass-1 Router: Block-level Scoring (no_grad)
            if self.is_router or selection_indices is None:
                with torch.no_grad():
                    q_blk_m = q.reshape(B, H, N_blocks, C, D).mean(dim=3)
                    k_blk_m = k.reshape(B, H, N_total_blocks, C, D).mean(dim=3)

                    raw_b = (q_blk_m @ k_blk_m.transpose(-2, -1)) * self.scale
                    cm_b  = torch.triu(
                        torch.full((N_blocks, N_total_blocks), float("-inf"), device=x.device),
                        diagonal=N_total_blocks - N_blocks + 1,
                    )
                    imp_b  = F.softmax(raw_b + cm_b[None, None], dim=-1).mean(1)
                    imp_bm = imp_b.masked_fill(cm_b[None] == float("-inf"), -1e9)

                    a_cap = min(A_blocks, N_total_blocks)
                    _, top_b = torch.topk(imp_bm, k=a_cap, dim=-1, sorted=False)

                    self_b = (
                        torch.arange(N_total_blocks - N_blocks, N_total_blocks, device=x.device)
                        .view(1, -1, 1)
                        .expand(B, N_blocks, 1)
                    )
                    selection_indices = torch.cat([top_b, self_b], dim=-1) # (B, N_blocks, a_cap + 1)

                # Recompute block margin loss with grad if requested
                if return_aux_loss:
                    q_blk_mg = q.reshape(B, H, N_blocks, C, D).mean(dim=3)
                    k_blk_mg = k.reshape(B, H, N_total_blocks, C, D).mean(dim=3)
                    cm_bg  = torch.triu(
                        torch.full((N_blocks, N_total_blocks), float("-inf"), device=x.device),
                        diagonal=N_total_blocks - N_blocks + 1,
                    )
                    imp_bg = F.softmax((q_blk_mg @ k_blk_mg.transpose(-2, -1)) * self.scale + cm_bg[None, None], dim=-1).mean(1)
                    sel_sb = torch.gather(imp_bg, -1, selection_indices)
                    min_selb = sel_sb.min(-1).values
                    validb = cm_bg[None].expand(B, N_blocks, N_total_blocks) != float("-inf")
                    smaskb = torch.zeros((B, N_blocks, N_total_blocks), dtype=torch.bool, device=x.device)
                    smaskb.scatter_(-1, selection_indices, True)
                    non_sb = imp_bg.masked_fill(~(validb & ~smaskb), -1e9)
                    mnsb   = non_sb.max(-1).values
                    mnsb   = torch.where(mnsb == -1e9, torch.zeros_like(mnsb), mnsb)
                    aux_loss = F.relu(self.margin_delta - min_selb + mnsb).mean()

            # 2. Pass-2: Block-sparse attention via SDPA
            num_sel_b = selection_indices.shape[-1]
            k_blk = k.reshape(B, H, N_total_blocks, C, D)
            v_blk = v.reshape(B, H, N_total_blocks, C, D)

            sel_bh = selection_indices.unsqueeze(1).expand(B, H, N_blocks, num_sel_b).reshape(B * H, N_blocks * num_sel_b)
            bh_idx = torch.arange(B * H, device=x.device).unsqueeze(1)

            k_2d = k_blk.reshape(B * H, N_total_blocks, C * D)
            v_2d = v_blk.reshape(B * H, N_total_blocks, C * D)

            k_gath_b = k_2d[bh_idx, sel_bh].reshape(B, H, N_blocks, num_sel_b, C, D)
            v_gath_b = v_2d[bh_idx, sel_bh].reshape(B, H, N_blocks, num_sel_b, C, D)

            k_ctx = k_gath_b.reshape(B * H * N_blocks, num_sel_b * C, D)
            v_ctx = v_gath_b.reshape(B * H * N_blocks, num_sel_b * C, D)
            q_ctx = q.reshape(B * H * N_blocks, C, D)

            if hasattr(F, "scaled_dot_product_attention") and x.is_cuda:
                out_ctx = F.scaled_dot_product_attention(q_ctx, k_ctx, v_ctx)
            else:
                s_b = (q_ctx @ k_ctx.transpose(-2, -1)) * self.scale
                out_ctx = F.softmax(s_b, dim=-1) @ v_ctx

            out = out_ctx.reshape(B, H, N, D)
            out = self.out_proj(out.transpose(1, 2).contiguous().view(B, N, self.d_model))
            return out, selection_indices, aux_loss, new_kv_cache

        # ── TOKEN-ASA FALLBACK PATH (for unaligned sequence lengths) ──
        if self.is_router or selection_indices is None:
            with torch.no_grad():
                raw = (q.detach() @ k.detach().transpose(-2, -1)) * self.scale
                cm  = torch.triu(torch.full((N, N_total), float("-inf"), device=x.device), 1)
                imp = F.softmax(raw + cm[None, None], dim=-1).mean(1)
                imp_m = imp.masked_fill(cm[None] == float("-inf"), -1e9)

                a_cap = min(budget, N_total)
                _, top_idx = torch.topk(imp_m, k=a_cap, dim=-1, sorted=False)
                self_idx  = (
                    torch.arange(N_total - N, N_total, device=x.device)
                    .view(1, -1, 1)
                    .expand(B, N, 1)
                )
                selection_indices = torch.cat([top_idx, self_idx], dim=-1)

            if return_aux_loss:
                cm_g  = torch.triu(torch.full((N, N_total), float("-inf"), device=x.device), 1)
                imp_g = F.softmax((q @ k.transpose(-2, -1)) * self.scale + cm_g[None, None], dim=-1).mean(1)
                sel_s = torch.gather(imp_g, -1, selection_indices)
                min_s = sel_s.min(-1).values
                valid = cm_g[None].expand(B, N, N_total) != float("-inf")
                smask = torch.zeros((B, N, N_total), dtype=torch.bool, device=x.device)
                smask.scatter_(-1, selection_indices, True)
                non_s = imp_g.masked_fill(~(valid & ~smask), -1e9)
                mns   = non_s.max(-1).values
                mns   = torch.where(mns == -1e9, torch.zeros_like(mns), mns)
                aux_loss = F.relu(self.margin_delta - min_s + mns).mean()

        k_gath, v_gath = _gather_kv_token(k, v, selection_indices)
        out = _pass2_token_attention(q, k_gath, v_gath, self.scale)
        out = self.out_proj(out.transpose(1, 2).contiguous().view(B, N, self.d_model))
        return out, selection_indices, aux_loss, new_kv_cache
