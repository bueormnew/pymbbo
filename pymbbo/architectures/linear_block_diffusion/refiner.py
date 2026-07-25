import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .linear_recurrent import BidirectionalLinearRecurrentBlock, RMSNorm, SwiGLU

class SinusoidalTimestepEmbedding(nn.Module):
    """Sinusoidal Timestep Embedding for Diffusion Step Conditioning."""
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model)
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: Tensor of shape (batch_size,) or (batch_size, 1) with step values or float ratio
        Returns:
            Embedding of shape (batch_size, d_model)
        """
        if timesteps.dim() == 2:
            timesteps = timesteps.squeeze(-1)
            
        half_dim = self.d_model // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * -emb)
        emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.d_model % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return self.mlp(emb)


class LinearContextCrossFusion(nn.Module):
    """
    Linear O(N) Context Fusion between prompt context memory H_prompt and target block X_block.
    Replaces cross-attention with a linear projection & gated memory integration.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.prompt_proj = nn.Linear(d_model, d_model)
        self.target_proj = nn.Linear(d_model, d_model)
        self.gate_proj = nn.Linear(d_model * 2, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, block_x: torch.Tensor, prompt_h: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            block_x: Target block tensor of shape (batch_size, block_len, d_model)
            prompt_h: Global prompt summary or representation of shape (batch_size, d_model) or (batch_size, prompt_len, d_model)
        Returns:
            Fused tensor of shape (batch_size, block_len, d_model)
        """
        if prompt_h is None:
            return block_x

        if prompt_h.dim() == 3:
            # Pool prompt representations over sequence length linearly
            prompt_ctx = prompt_h.mean(dim=1) # (batch_size, d_model)
        else:
            prompt_ctx = prompt_h

        # Broadcast prompt context across target sequence length
        prompt_expanded = prompt_ctx.unsqueeze(1).expand_as(block_x) # (batch_size, block_len, d_model)
        
        gate = torch.sigmoid(self.gate_proj(torch.cat([block_x, prompt_expanded], dim=-1)))
        fused = gate * self.target_proj(block_x) + (1.0 - gate) * self.prompt_proj(prompt_expanded)
        return block_x + self.out_proj(fused)


class DiffusionBlockRefiner(nn.Module):
    """
    Diffusion Block Refiner Module.
    Combines timestep conditioning, prompt cross-fusion, and stacked 
    Bidirectional Linear Recurrent Blocks to perform O(N) diffusion passes.
    """
    def __init__(self, d_model: int, num_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.time_embedder = SinusoidalTimestepEmbedding(d_model)
        self.context_fusion = LinearContextCrossFusion(d_model)

        self.layers = nn.ModuleList([
            BidirectionalLinearRecurrentBlock(d_model, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(d_model)

    def forward(self, block_embeddings: torch.Tensor, timesteps: torch.Tensor, prompt_context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            block_embeddings: Tensor of shape (batch_size, seq_len, d_model)
            timesteps: Tensor of shape (batch_size,) with diffusion step indices or normalized ratios
            prompt_context: Optional prompt context representation (batch_size, prompt_len, d_model) or (batch_size, d_model)
        Returns:
            Refined embeddings of shape (batch_size, seq_len, d_model)
        """
        # Inject timestep embedding into input sequence
        time_emb = self.time_embedder(timesteps).unsqueeze(1) # (batch_size, 1, d_model)
        x = block_embeddings + time_emb

        # Fuse prompt memory context
        x = self.context_fusion(x, prompt_context)

        # Pass through stacked Bidirectional Linear Recurrent Layers
        for layer in self.layers:
            x = layer(x)

        return self.final_norm(x)
