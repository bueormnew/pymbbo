import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, Union

from pymbbo.architectures.base_arch import BaseArchitecture
from pymbbo.models.registry import register_architecture
from .linear_recurrent import BidirectionalLinearRecurrentBlock, RMSNorm
from .refiner import DiffusionBlockRefiner

@register_architecture("linear_block_diffusion")
class LinearBlockDiffusionArchitecture(BaseArchitecture):
    """
    Linear Recurrent Overlapping Block Diffusion (LinearBlockDiffusion) Architecture.
    
    Combines:
    1. Attention-Free O(N) Recurrent Backbone
    2. Progressive Block Diffusion Refinement (K passes with [MASK] unmasking)
    3. Autoregressive Overlapping Block Windowing (e.g. 50% overlap sliding window)
    4. Full compatibility with standard autoregressive text datasets (Cross-Entropy Loss)
    5. Fully configurable and dynamically overridable hyperparameters (at init and generate())
    6. Exposure-Bias Mitigation via Scheduled Noise Injection & Dynamic Overlap Training
    """
    ARCH_NAME = "linear_block_diffusion"

    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 256,
        num_layers: int = 4,
        block_size: int = 512,
        overlap_ratio: float = 0.5,
        num_diffusion_steps: int = 8,
        chunk_denoise_size: int = 64,
        mask_token_id: Optional[int] = None,
        pad_token_id: int = 0,
        eos_token_id: Optional[int] = 2,
        dropout: float = 0.1,
        max_seq_len: int = 8192,
        noise_injection_prob: float = 0.15,
        randomize_overlap_training: bool = True,
        **kwargs
    ):
        super().__init__(
            vocab_size=vocab_size,
            d_model=d_model,
            num_layers=num_layers,
            block_size=block_size,
            overlap_ratio=overlap_ratio,
            num_diffusion_steps=num_diffusion_steps,
            chunk_denoise_size=chunk_denoise_size,
            mask_token_id=mask_token_id,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            dropout=dropout,
            max_seq_len=max_seq_len,
            noise_injection_prob=noise_injection_prob,
            randomize_overlap_training=randomize_overlap_training,
            **kwargs
        )
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.block_size = block_size
        self.overlap_ratio = overlap_ratio
        self.num_diffusion_steps = num_diffusion_steps
        self.chunk_denoise_size = chunk_denoise_size
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.mask_token_id = mask_token_id if mask_token_id is not None else (vocab_size - 1)
        self.noise_injection_prob = noise_injection_prob
        self.randomize_overlap_training = randomize_overlap_training

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        # Prompt Encoder (Attention-free linear recurrent blocks)
        self.prompt_encoder = nn.ModuleList([
            BidirectionalLinearRecurrentBlock(d_model, dropout=dropout)
            for _ in range(2)
        ])
        self.prompt_norm = RMSNorm(d_model)

        # Diffusion Block Refiner Stack
        self.refiner = DiffusionBlockRefiner(d_model=d_model, num_layers=num_layers, dropout=dropout)

        # Output LM Head (Tied weights with token embeddings)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def encode_prompt(self, prompt_ids: torch.Tensor) -> torch.Tensor:
        """Encodes input prompt tokens of variable length into persistent memory context."""
        batch_size, seq_len = prompt_ids.shape
        pos = torch.arange(seq_len, device=prompt_ids.device).unsqueeze(0)
        x = self.token_embedding(prompt_ids) + self.pos_embedding(pos)

        for layer in self.prompt_encoder:
            x = layer(x)
        return self.prompt_norm(x)

    def forward(
        self,
        x: torch.Tensor,
        target_ids: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
        noise_injection_prob: Optional[float] = None,
        **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for training or evaluation.
        
        If target_ids is provided (Standard Dataset Training):
            1. Samples a diffusion timestep k in [1, num_diffusion_steps]
            2. Corrupts target_ids according to diffusion noise schedule with [MASK] tokens
            3. Applies Noise Injection (Exposure-Bias Mitigation)
            4. Refines the noisy sequence through DiffusionBlockRefiner
            5. Computes Cross-Entropy loss against target_ids
        """
        prompt_ids = x
        batch_size, prompt_len = prompt_ids.shape
        prompt_ctx = self.encode_prompt(prompt_ids)

        if target_ids is None:
            # Simple forward pass over input
            pos = torch.arange(prompt_len, device=prompt_ids.device).unsqueeze(0)
            emb = self.token_embedding(prompt_ids) + self.pos_embedding(pos)
            t_steps = torch.ones(batch_size, device=prompt_ids.device) * self.num_diffusion_steps
            refined = self.refiner(emb, t_steps, prompt_ctx)
            logits = self.lm_head(refined)
            return logits

        # Standard Dataset Training Mode
        target_batch, target_len = target_ids.shape
        
        # Sample random timestep k per batch item
        if timestep is None:
            t_k = torch.randint(1, self.num_diffusion_steps + 1, (target_batch,), device=target_ids.device)
        else:
            t_k = timestep

        # Masking schedule: noise ratio alpha_k = 1 - (k / K)
        alpha_k = 1.0 - (t_k.float() / float(self.num_diffusion_steps)) # ratio of masked tokens
        p_noise = noise_injection_prob if noise_injection_prob is not None else self.noise_injection_prob

        # Create noisy target sequence
        target_noisy = target_ids.clone()
        for i in range(target_batch):
            num_to_mask = int(target_len * alpha_k[i].item())
            if num_to_mask > 0:
                mask_indices = torch.randperm(target_len, device=target_ids.device)[:num_to_mask]
                target_noisy[i, mask_indices] = self.mask_token_id

            # Exposure Bias Mitigation: Inject random token corruption on a fraction of unmasked tokens
            if p_noise > 0 and self.training:
                unmasked_indices = (target_noisy[i] != self.mask_token_id).nonzero(as_tuple=True)[0]
                if len(unmasked_indices) > 0:
                    num_corrupt = int(len(unmasked_indices) * p_noise)
                    if num_corrupt > 0:
                        corrupt_subset = unmasked_indices[torch.randperm(len(unmasked_indices), device=target_ids.device)[:num_corrupt]]
                        rand_tokens = torch.randint(1, self.vocab_size - 1, (num_corrupt,), device=target_ids.device)
                        target_noisy[i, corrupt_subset] = rand_tokens

        # Embed noisy target
        pos = torch.arange(target_len, device=target_ids.device).unsqueeze(0)
        target_emb = self.token_embedding(target_noisy) + self.pos_embedding(pos)

        # Refine target sequence
        refined_emb = self.refiner(target_emb, t_k, prompt_ctx)
        logits = self.lm_head(refined_emb) # (batch_size, target_len, vocab_size)

        # Compute Cross-Entropy Loss over original target tokens
        loss = F.cross_entropy(
            logits.view(-1, self.vocab_size),
            target_ids.view(-1),
            ignore_index=self.pad_token_id
        )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 512,
        block_size: Optional[int] = None,
        overlap_ratio: Optional[float] = None,
        num_diffusion_steps: Optional[int] = None,
        chunk_denoise_size: Optional[int] = None,
        temperature: float = 1.0,
        top_k: int = 50,
        eos_token_id: Optional[int] = None
    ) -> torch.Tensor:
        """
        Progressive Overlapping Block Diffusion Inference Generator.
        
        Args:
            prompt_ids: Prompt token tensor of shape (batch_size, prompt_len)
            max_new_tokens: Maximum number of tokens to generate
            block_size: Block size override (default: self.block_size, e.g. 512)
            overlap_ratio: Overlap ratio override (default: self.overlap_ratio, e.g. 0.5)
            num_diffusion_steps: Diffusion passes per block override (default: self.num_diffusion_steps, e.g. 8)
            chunk_denoise_size: Sub-chunk progressive unmask size (default: self.chunk_denoise_size, e.g. 64)
            temperature: Sampling temperature
            top_k: Top-k logits filtering
            eos_token_id: Optional End-of-Sequence token ID for early stopping
        Returns:
            Generated sequence tensor of shape (batch_size, prompt_len + generated_length)
        """
        self.eval()
        B_size = block_size if block_size is not None else self.block_size
        ov_ratio = overlap_ratio if overlap_ratio is not None else self.overlap_ratio
        K_steps = num_diffusion_steps if num_diffusion_steps is not None else self.num_diffusion_steps
        chunk_size = chunk_denoise_size if chunk_denoise_size is not None else self.chunk_denoise_size
        stop_eos = eos_token_id if eos_token_id is not None else self.eos_token_id

        batch_size, prompt_len = prompt_ids.shape
        device = prompt_ids.device

        # 1. Encode variable-length prompt
        prompt_ctx = self.encode_prompt(prompt_ids)

        # Calculate stride and overlap prefix length
        overlap_len = int(B_size * ov_ratio)
        stride = B_size - overlap_len # new tokens produced per block shift

        generated_seq = prompt_ids.clone()
        tokens_generated = 0

        # Accumulated context for overlapping blocks
        refined_overlap_prefix = None
        eos_found = False

        while tokens_generated < max_new_tokens and not eos_found:
            # Determine suffix size for current block
            current_new_len = min(stride, max_new_tokens - tokens_generated)
            current_block_len = (overlap_len if refined_overlap_prefix is not None else 0) + current_new_len

            # Initialize block tokens with [MASK] for new suffix
            block_tokens = torch.full(
                (batch_size, current_block_len),
                fill_value=self.mask_token_id,
                dtype=torch.long,
                device=device
            )

            # Copy preserved overlap prefix if available
            if refined_overlap_prefix is not None:
                block_tokens[:, :overlap_len] = refined_overlap_prefix

            # 2. Progressive Diffusion Refinement over K passes
            for k in range(1, K_steps + 1):
                # Calculate active unmask boundary index (Left-to-Right progressive denoise)
                unmask_limit = (overlap_len if refined_overlap_prefix is not None else 0) + min(k * chunk_size, current_new_len)

                # Embed current block state
                pos = torch.arange(current_block_len, device=device).unsqueeze(0)
                emb = self.token_embedding(block_tokens) + self.pos_embedding(pos)

                t_k = torch.full((batch_size,), fill_value=k, device=device, dtype=torch.long)
                refined_emb = self.refiner(emb, t_k, prompt_ctx)
                logits = self.lm_head(refined_emb) # (batch_size, current_block_len, vocab_size)

                # Sample predictions for unmasked chunk range
                unmask_start = overlap_len if refined_overlap_prefix is not None else 0
                if unmask_limit > unmask_start:
                    sub_logits = logits[:, unmask_start:unmask_limit, :] / max(temperature, 1e-5)
                    
                    if top_k > 0:
                        v, _ = torch.topk(sub_logits, min(top_k, sub_logits.size(-1)))
                        sub_logits[sub_logits < v[:, :, [-1]]] = -float('Inf')

                    probs = F.softmax(sub_logits, dim=-1)
                    sampled_tokens = torch.multinomial(probs.view(-1, self.vocab_size), num_samples=1).view(batch_size, -1)
                    
                    # Update block tokens in active chunk range (Tokens once unmasked stay frozen for stability)
                    block_tokens[:, unmask_start:unmask_limit] = sampled_tokens

            # Extract newly refined tokens
            newly_refined_tokens = block_tokens[:, (overlap_len if refined_overlap_prefix is not None else 0):]

            # Check for EOS token in newly refined sequence
            if stop_eos is not None and (newly_refined_tokens == stop_eos).any():
                eos_found = True
                # Truncate at first EOS position
                eos_mask = (newly_refined_tokens == stop_eos)
                first_eos_idx = (eos_mask.cumsum(dim=1) == 1).nonzero(as_tuple=False)
                if len(first_eos_idx) > 0:
                    cut_off = first_eos_idx[0, 1].item() + 1
                    newly_refined_tokens = newly_refined_tokens[:, :cut_off]

            generated_seq = torch.cat([generated_seq, newly_refined_tokens], dim=1)
            tokens_generated += newly_refined_tokens.shape[1]

            # Update overlap prefix for next window shift
            if block_tokens.shape[1] >= overlap_len:
                refined_overlap_prefix = block_tokens[:, -overlap_len:].clone()

        return generated_seq
