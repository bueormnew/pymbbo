import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

from pymbbo.architectures.asa_transformer.asa_attention import AdaptiveSelectiveAttention


class FeedForward(nn.Module):
    """
    Standard MLP Feed-Forward Network with GELU activation.
    """
    def __init__(self, d_model: int, dim_feedforward: int = None, dropout: float = 0.0):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = d_model * 4
        self.fc1 = nn.Linear(d_model, dim_feedforward)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(self.act(self.fc1(x))))


class ASABlock(nn.Module):
    """
    Single Pre-LN Transformer Decoder Block using Adaptive Selective Attention.
    """
    def __init__(
        self,
        d_model: int,
        nhead: int,
        max_a: int = 64,
        is_router: bool = True,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.0,
        curriculum_budgets: Optional[List[Optional[int]]] = None,
        margin_delta: float = 0.1
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = AdaptiveSelectiveAttention(
            d_model=d_model,
            nhead=nhead,
            max_a=max_a,
            is_router=is_router,
            dropout=dropout,
            curriculum_budgets=curriculum_budgets,
            margin_delta=margin_delta
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model=d_model, dim_feedforward=dim_feedforward, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        selection_indices: Optional[torch.Tensor] = None,
        max_a: Optional[int] = None,
        return_aux_loss: bool = False,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        
        # Pre-LN Attention
        normed_x = self.ln1(x)
        attn_out, derived_indices, aux_loss, new_kv_cache = self.attn(
            normed_x,
            selection_indices=selection_indices,
            max_a=max_a,
            return_aux_loss=return_aux_loss,
            kv_cache=kv_cache
        )
        x = x + attn_out
        
        # Pre-LN FFN
        x = x + self.ffn(self.ln2(x))
        
        return x, derived_indices, aux_loss, new_kv_cache


class ASALayerGroup(nn.Module):
    """
    Group of 'g' consecutive Transformer layers implementing Selection Sharing (Section 4.6).
    - Layer 0: Router layer (computes Pass 1 scoring & derives selection T)
    - Layers 1..g-1: Follower layers (reuses selection T with their own Q,K,V)
    """
    def __init__(
        self,
        group_size: int,
        d_model: int,
        nhead: int,
        max_a: int = 64,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.0,
        curriculum_budgets: Optional[List[Optional[int]]] = None,
        margin_delta: float = 0.1
    ):
        super().__init__()
        self.group_size = group_size
        self.layers = nn.ModuleList()
        
        for idx in range(group_size):
            is_router = (idx == 0) # Only the first layer in the group is a Router layer
            self.layers.append(
                ASABlock(
                    d_model=d_model,
                    nhead=nhead,
                    max_a=max_a,
                    is_router=is_router,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    curriculum_budgets=curriculum_budgets,
                    margin_delta=margin_delta
                )
            )

    def forward(
        self,
        x: torch.Tensor,
        max_a: Optional[int] = None,
        return_aux_loss: bool = False,
        group_kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]]:
        
        shared_selection = None
        group_aux_loss = None
        new_group_kv_caches = []

        for layer_idx, layer in enumerate(self.layers):
            kv_c = group_kv_caches[layer_idx] if group_kv_caches is not None else None
            
            x, derived_sel, aux_l, new_kv_c = layer(
                x,
                selection_indices=shared_selection,
                max_a=max_a,
                return_aux_loss=return_aux_loss,
                kv_cache=kv_c
            )
            
            # Router layer produces shared_selection for all follower layers in group
            if layer_idx == 0:
                shared_selection = derived_sel
                
            if aux_l is not None:
                if group_aux_loss is None:
                    group_aux_loss = aux_l
                else:
                    group_aux_loss = group_aux_loss + aux_l
                    
            if new_kv_c is not None:
                new_group_kv_caches.append(new_kv_c)

        return x, group_aux_loss, new_group_kv_caches
