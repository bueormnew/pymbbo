import math
from typing import Dict, Any, Optional, List, Union, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from pymbbo.architectures.base_arch import BaseArchitecture
from pymbbo.models.registry import register_architecture
from pymbbo.architectures.asa_transformer.asa_block import ASALayerGroup


@register_architecture("asa_transformer")
@register_architecture("asa_gpt")
class ASATransformerArchitecture(BaseArchitecture):
    """
    Adaptive Selective Attention (ASA) Transformer / GPT Architecture Plugin for PYMBBO.
    Based on Buenahora Ormaza (2026).

    Combines score-gated two-pass attention with selection sharing across layer groups,
    variable-budget curriculum training, optional margin loss, and KV-cache autoregressive decoding.
    """
    ARCH_NAME = "asa_transformer"

    def __init__(
        self,
        vocab_size: int = 1000,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        max_seq_len: int = 2048,
        group_size: int = 2,
        max_a: int = 64,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.0,
        curriculum_budgets: Optional[List[Optional[int]]] = None,
        margin_loss_weight: float = 0.01,
        margin_delta: float = 0.1,
        **kwargs
    ):
        super().__init__(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            max_seq_len=max_seq_len,
            group_size=group_size,
            max_a=max_a,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            curriculum_budgets=curriculum_budgets,
            margin_loss_weight=margin_loss_weight,
            margin_delta=margin_delta,
            **kwargs
        )

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.group_size = group_size
        self.default_max_a = max_a
        self.margin_loss_weight = margin_loss_weight

        if curriculum_budgets is None:
            curriculum_budgets = [16, 32, 64, 128, 256, None]
        self.curriculum_budgets = curriculum_budgets

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        # Layer Groups
        self.groups = nn.ModuleList()
        num_groups = math.ceil(num_layers / group_size)
        remaining_layers = num_layers

        for _ in range(num_groups):
            g_size = min(group_size, remaining_layers)
            self.groups.append(
                ASALayerGroup(
                    group_size=g_size,
                    d_model=d_model,
                    nhead=nhead,
                    max_a=max_a,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    curriculum_budgets=curriculum_budgets,
                    margin_delta=margin_delta
                )
            )
            remaining_layers -= g_size

        self.final_ln = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)

        self.last_aux_loss = None

    def forward(
        self,
        x: torch.Tensor,
        max_a: Optional[int] = None,
        return_aux_loss: bool = False,
        kv_caches: Optional[List[List[Tuple[torch.Tensor, torch.Tensor]]]] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for ASA Transformer.

        Args:
            x: Input token IDs of shape (batch, seq_len)
            max_a: Dynamic inference or training selection budget (overrides default max_a)
            return_aux_loss: If True, returns (logits, aux_loss)
            kv_caches: List of KV cache pairs per group for autoregressive generation

        Returns:
            logits: Logits of shape (batch, seq_len, vocab_size)
            aux_loss: Optional auxiliary margin loss scalar tensor if requested
        """
        batch_size, seq_len = x.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length ({seq_len}) exceeds model max_seq_len ({self.max_seq_len})")

        # Determine start position offset if kv_caches provided
        if kv_caches is not None and len(kv_caches) > 0 and len(kv_caches[0]) > 0:
            past_len = kv_caches[0][0][0].shape[2]
        else:
            past_len = 0

        positions = torch.arange(past_len, past_len + seq_len, device=x.device).unsqueeze(0)
        h = self.token_embedding(x) + self.pos_embedding(positions)
        h = self.dropout(h)

        total_aux_loss = None
        new_kv_caches = []

        for idx, group in enumerate(self.groups):
            g_caches = kv_caches[idx] if kv_caches is not None else None
            h, g_aux_loss, new_g_caches = group(
                h,
                max_a=max_a,
                return_aux_loss=return_aux_loss,
                group_kv_caches=g_caches
            )

            if g_aux_loss is not None:
                if total_aux_loss is None:
                    total_aux_loss = g_aux_loss
                else:
                    total_aux_loss = total_aux_loss + g_aux_loss
            
            if new_g_caches:
                new_kv_caches.append(new_g_caches)

        h = self.final_ln(h)
        logits = self.fc_out(h)

        self.last_aux_loss = total_aux_loss

        if return_aux_loss:
            return logits, total_aux_loss

        return logits

    def generate(
        self,
        prompt_tokens: torch.Tensor,
        max_new_tokens: int = 100,
        max_a: Optional[int] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> torch.Tensor:
        """
        Fast autoregressive text generation using Adaptive Selective Attention with KV caching.

        Args:
            prompt_tokens: Initial prompt tensor of shape (batch, prompt_len)
            max_new_tokens: Maximum number of new tokens to generate
            max_a: Dynamic selection budget 'max_a' for inference
            temperature: Sampling temperature
            top_k: Optional top-k filtering parameter
            top_p: Optional top-p nucleus sampling parameter

        Returns:
            generated_tokens: Output token sequence including prompt of shape (batch, prompt_len + new_tokens)
        """
        self.eval()
        curr_tokens = prompt_tokens.clone()
        device = prompt_tokens.device
        
        # Initial forward pass over prompt to build KV cache
        kv_caches = None
        with torch.no_grad():
            positions = torch.arange(0, curr_tokens.size(1), device=device).unsqueeze(0)
            h = self.token_embedding(curr_tokens) + self.pos_embedding(positions)
            
            kv_caches = []
            for group in self.groups:
                h, _, g_caches = group(h, max_a=max_a, group_kv_caches=None)
                kv_caches.append(g_caches)

            logits = self.fc_out(self.final_ln(h))
            next_logits = logits[:, -1, :] / max(temperature, 1e-5)
            
            if top_k is not None:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, [-1]]] = -float('Inf')
                
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            curr_tokens = torch.cat([curr_tokens, next_token], dim=1)

            # Generate step by step using KV cache
            for _ in range(max_new_tokens - 1):
                if curr_tokens.size(1) >= self.max_seq_len:
                    break

                step_token = curr_tokens[:, -1:]
                past_len = curr_tokens.size(1) - 1
                pos = torch.tensor([[past_len]], device=device)
                
                h_step = self.token_embedding(step_token) + self.pos_embedding(pos)
                
                new_kv_caches = []
                for idx, group in enumerate(self.groups):
                    h_step, _, g_caches = group(h_step, max_a=max_a, group_kv_caches=kv_caches[idx])
                    new_kv_caches.append(g_caches)
                
                kv_caches = new_kv_caches
                logits_step = self.fc_out(self.final_ln(h_step))
                next_logits = logits_step[:, -1, :] / max(temperature, 1e-5)
                
                if top_k is not None:
                    v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                    next_logits[next_logits < v[:, [-1]]] = -float('Inf')

                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
                curr_tokens = torch.cat([curr_tokens, next_token], dim=1)

        return curr_tokens
