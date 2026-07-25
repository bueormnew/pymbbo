import math
import random
from typing import Optional, Tuple, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from pymbbo.architectures.asa_transformer.triton_kernel import asa_pass2_attention


class AdaptiveSelectiveAttention(nn.Module):
    """
    Adaptive Selective Attention (ASA) Module based on Buenahora Ormaza (2026).
    
    Implements a two-pass score-gated attention mechanism:
    1. Pass 1 (Router): Computes full causal attention to derive per-token importance score I_i(j).
    2. Selection: Selects top-a past tokens (max_a) + self token i for each position i.
    3. Pass 2: Re-computes clean, renormalized attention restricted only to the selected tokens.
    4. Selection Sharing: Router layer computes selection T; Follower layers reuse T with their own Q,K,V.
    """
    def __init__(
        self,
        d_model: int,
        nhead: int,
        max_a: int = 64,
        is_router: bool = True,
        dropout: float = 0.0,
        curriculum_budgets: Optional[List[Optional[int]]] = None,
        margin_delta: float = 0.1
    ):
        super().__init__()
        assert d_model % nhead == 0, f"d_model ({d_model}) must be divisible by nhead ({nhead})"
        
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.max_a = max_a
        self.is_router = is_router
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.curriculum_budgets = curriculum_budgets
        self.margin_delta = margin_delta
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)

    def _sample_budget(self, seq_len: int, dynamic_max_a: Optional[int] = None) -> int:
        """Determines the selection budget 'a' for current forward pass."""
        if dynamic_max_a is not None:
            return dynamic_max_a
            
        if self.training and self.curriculum_budgets is not None and len(self.curriculum_budgets) > 0:
            sampled = random.choice(self.curriculum_budgets)
            if sampled is None or sampled >= seq_len:
                return seq_len # Full attention
            return max(1, sampled)
            
        return self.max_a

    def forward(
        self,
        x: torch.Tensor,
        selection_indices: Optional[torch.Tensor] = None,
        max_a: Optional[int] = None,
        return_aux_loss: bool = False,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass for ASA.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            selection_indices: Precomputed selection indices tensor of shape (batch, seq_len, num_selected) if follower layer
            max_a: Optional dynamic selection budget override
            return_aux_loss: Whether to compute optional margin loss (router layer only)
            kv_cache: Optional tuple of (cached_k, cached_v) for fast autoregressive decoding

        Returns:
            output: Attention output tensor of shape (batch, seq_len, d_model)
            selection_indices: Derived selection indices if router layer, else passed selection_indices
            aux_loss: Margin loss scalar tensor (if router layer & return_aux_loss=True), else None
            new_kv_cache: Updated (k, v) cache tuple if kv_cache was provided
        """
        B, N, _ = x.shape
        
        # Linear projections
        q = self.q_proj(x).view(B, N, self.nhead, self.head_dim).transpose(1, 2) # (B, H, N, D)
        k = self.k_proj(x).view(B, N, self.nhead, self.head_dim).transpose(1, 2) # (B, H, N, D)
        v = self.v_proj(x).view(B, N, self.nhead, self.head_dim).transpose(1, 2) # (B, H, N, D)

        # Fast Autoregressive Decoding with KV Cache
        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            k = torch.cat([cached_k, k], dim=2) # (B, H, N_total, D)
            v = torch.cat([cached_v, v], dim=2) # (B, H, N_total, D)
        new_kv_cache = (k, v)

        N_total = k.shape[2] # Current total sequence length (including cached keys)
        budget = self.sample_budget = self._sample_budget(N_total, max_a)

        # Fallback to full causal attention if budget covers full context or sequence is short
        if budget >= N_total:
            # Full causal attention fallback using FlashAttention (PyTorch F.scaled_dot_product_attention)
            if hasattr(F, "scaled_dot_product_attention") and not return_aux_loss and x.is_cuda:
                dropout_p = self.dropout.p if self.training else 0.0
                is_causal = (N == N_total)
                attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, dropout_p=dropout_p)
                out = attn_out.transpose(1, 2).contiguous().view(B, N, self.d_model)
            else:
                attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale # (B, H, N, N_total)
                if N == N_total:
                    causal_mask = torch.triu(torch.full((N, N), float('-inf'), device=x.device), diagonal=1)
                    attn_scores = attn_scores + causal_mask.unsqueeze(0).unsqueeze(0)
                attn_weights = F.softmax(attn_scores, dim=-1)
                attn_weights = self.dropout(attn_weights)
                out = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(B, N, self.d_model)

            out = self.out_proj(out)
            full_indices = torch.arange(N_total, device=x.device).unsqueeze(0).unsqueeze(0).expand(B, N, N_total)
            return out, full_indices, None, new_kv_cache

        aux_loss = None

        # -------------------------------------------------------------
        # Router Layer: Pass 1 Scoring & Selection Derivation
        # -------------------------------------------------------------
        if self.is_router or selection_indices is None:
            # 1. Full Causal Attention Scoring Pass
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale # (B, H, N, N_total)
            causal_mask = torch.triu(torch.full((N, N_total), float('-inf'), device=x.device), diagonal=1)
            scores_masked = scores + causal_mask.unsqueeze(0).unsqueeze(0)
            
            # 2. Compute importance scores I_i(j) = mean over heads of softmax(scores)
            pass1_weights = F.softmax(scores_masked, dim=-1) # (B, H, N, N_total)
            importance_scores = pass1_weights.mean(dim=1) # (B, N, N_total)
            
            # Mask out invalid future tokens in importance scores for top-k selection
            masked_importance = importance_scores.masked_fill(
                causal_mask.unsqueeze(0) == float('-inf'), -1e9
            )
            
            # 3. Top-a token selection
            # Budget 'a': select top 'a' tokens. Always include self-token i.
            # Number of selected tokens per query = min(budget, N_total)
            a_cap = min(budget, N_total)
            _, top_indices = torch.topk(masked_importance, k=a_cap, dim=-1, sorted=False) # (B, N, a_cap)
            
            # Always append/ensure self-token index i for each position i
            self_indices = torch.arange(N_total - N, N_total, device=x.device).unsqueeze(0).unsqueeze(-1).expand(B, N, 1)
            
            # Combine top_indices and self_indices, removing duplicates or using set union pattern
            selection_indices = torch.cat([top_indices, self_indices], dim=-1) # (B, N, a_cap + 1)

            # Compute Margin Loss if requested (Eq. 11 in paper)
            if return_aux_loss:
                # Margin loss = max(0, delta - min_{j in T} I_i(j) + max_{j not in T} I_i(j))
                # Gather scores of selected vs non-selected
                selected_scores = torch.gather(importance_scores, dim=-1, index=selection_indices)
                min_selected = selected_scores.min(dim=-1).values # (B, N)
                
                # Create mask for non-selected valid past tokens
                valid_mask = (causal_mask.unsqueeze(0).expand(B, N, N_total) != float('-inf'))
                sel_mask = torch.zeros((B, N, N_total), dtype=torch.bool, device=x.device)
                sel_mask.scatter_(dim=-1, index=selection_indices, value=True)
                non_selected_mask = valid_mask & (~sel_mask)
                
                non_sel_scores = importance_scores.masked_fill(~non_selected_mask, -1e9)
                max_non_selected = non_sel_scores.max(dim=-1).values
                max_non_selected = torch.where(max_non_selected == -1e9, torch.tensor(0.0, device=x.device), max_non_selected)
                
                margin_terms = F.relu(self.margin_delta - min_selected + max_non_selected)
                aux_loss = margin_terms.mean()

        # -------------------------------------------------------------
        # Pass 2: Renormalized Sparse Attention over Selected Tokens
        # -------------------------------------------------------------
        # Memory-efficient flatten-index gather:
        # Avoids materializing the catastrophic (B, H, N, N_total, D) intermediate tensor.
        # Instead flattens the N_total*D dimension and builds a flat index, keeping
        # memory cost at O(B * H * N * A * D) instead of O(B * H * N * N_total * D).
        # -------------------------------------------------------------
        num_selected = selection_indices.shape[-1]
        D = self.head_dim

        # k, v: (B, H, N_total, D)
        # Flatten to (B*H, N_total*D) for flat indexing
        k_flat = k.reshape(B * self.nhead, N_total * D)
        v_flat = v.reshape(B * self.nhead, N_total * D)

        # selection_indices: (B, N, num_selected)
        # Expand head dim:  (B*H, N, num_selected)
        sel_bh = selection_indices.unsqueeze(1).expand(B, self.nhead, N, num_selected) \
                                  .reshape(B * self.nhead, N, num_selected)

        # Build flat indices: sel[bh, n, a] * D + d for each d in [0..D)
        # Shape: (B*H, N, num_selected, D) → reshape to (B*H, N*num_selected*D)
        d_offset = torch.arange(D, device=x.device).view(1, 1, 1, D)
        flat_idx = (sel_bh.unsqueeze(-1) * D + d_offset).reshape(B * self.nhead, N * num_selected * D)

        # Gather and reshape to (B, H, N, num_selected, D)
        k_gathered = torch.gather(k_flat, 1, flat_idx).reshape(B, self.nhead, N, num_selected, D)
        v_gathered = torch.gather(v_flat, 1, flat_idx).reshape(B, self.nhead, N, num_selected, D)

        # Fused Pass-2 Attention computation
        attn_out = asa_pass2_attention(q, k_gathered, v_gathered, scale=self.scale)  # (B, H, N, D)

        # Reshape & project output
        out = attn_out.transpose(1, 2).contiguous().view(B, N, self.d_model)
        out = self.out_proj(out)
        
        return out, selection_indices, aux_loss, new_kv_cache
