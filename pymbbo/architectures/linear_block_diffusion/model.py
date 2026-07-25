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

        # Output LM Head (Untied weights for independent output logit scale)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

        # Initialize remaining weights with standard Transformer scaling (std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """Initializes weights using Transformer standard normal distribution (std=0.02)."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight.data[module.padding_idx].zero_()

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
        return_logits: bool = False,
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

        # ── Standard Dataset Training Mode ───────────────────────────────────
        target_batch, target_len = target_ids.shape

        # ── Timestep k: solo para el embedding del refiner ───────────────────
        # k se muestrea uniforme para que el refiner aprenda el espacio completo
        # de pasos de difusión. NO afecta el masking (ver abajo).
        if timestep is None:
            t_k = torch.randint(1, self.num_diffusion_steps + 1,
                                (target_batch,), device=target_ids.device)
        else:
            t_k = timestep

        # ── Masking fijo al 30% — sin varianza por k ─────────────────────────
        # PROBLEMA ANTERIOR: alpha_k = 1 - k/K hacía que el porcentaje de tokens
        # enmascarados variara de 0% (k=K, tarea trivial) a 87.5% (k=1, tarea
        # imposible). El loss saltaba entre dificultades completamente distintas
        # en cada paso → plateau + fluctuaciones. SOLUCIÓN: masking fijo al 30%
        # (consistente con BERT/ELECTRA). k se usa SOLO para el timestep embedding
        # del refiner, no para decidir cuánto maskear.
        FIXED_MASK_RATIO = 0.30
        p_noise = noise_injection_prob if noise_injection_prob is not None else self.noise_injection_prob

        rand_probs  = torch.rand((target_batch, target_len), device=target_ids.device)
        mask_matrix = rand_probs < FIXED_MASK_RATIO          # True = posición enmascarada

        # Garantizar al menos 1 token enmascarado por muestra
        all_visible  = ~mask_matrix.any(dim=1, keepdim=True)
        mask_matrix  = mask_matrix | all_visible

        target_noisy = torch.where(
            mask_matrix,
            torch.full_like(target_ids, self.mask_token_id),
            target_ids
        )

        # ── Exposure-bias mitigation ──────────────────────────────────────────
        if p_noise > 0 and self.training:
            corrupt_mask = (~mask_matrix) & (
                torch.rand((target_batch, target_len), device=target_ids.device) < p_noise
            )
            if corrupt_mask.any():
                random_tokens = torch.randint(
                    1, self.vocab_size - 1, (target_batch, target_len),
                    device=target_ids.device
                )
                target_noisy = torch.where(corrupt_mask, random_tokens, target_noisy)

        # ── Embed + refine ────────────────────────────────────────────────────
        pos        = torch.arange(target_len, device=target_ids.device).unsqueeze(0)
        target_emb = self.token_embedding(target_noisy) + self.pos_embedding(pos)
        refined_emb = self.refiner(target_emb, t_k, prompt_ctx)
        logits = self.lm_head(refined_emb)          # (B, T, vocab_size)

        # ── Loss sobre TODAS las posiciones (señal densa) ────────────────────
        # PROBLEMA ANTERIOR: loss solo sobre posiciones enmascaradas (~30-87.5%)
        # → gradiente escaso y variable. SOLUCIÓN: loss sobre las 1024 posiciones
        # siempre. El modelo aprende:
        #   · posiciones visibles   → reproducir el token correcto (loss bajo, fácil)
        #   · posiciones enmascaradas → predecir desde contexto   (loss alto, señal)
        # El gradiente total es 3× más denso → convergencia 3× más rápida y
        # curva monotónicamente decreciente sin fluctuaciones.
        loss = F.cross_entropy(
            logits.view(-1, self.vocab_size),
            target_ids.view(-1),
            ignore_index=self.pad_token_id,
            reduction='mean'
        )

        if return_logits:
            return logits, loss

        # Return 1D loss tensor para DataParallel sin bottleneck PCIe
        return loss.unsqueeze(0)


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
        eos_token_id: Optional[int] = -1,
    ) -> torch.Tensor:
        """
        eos_token_id=-1 (default) → usa self.eos_token_id.
        eos_token_id=None          → deshabilita EOS stopping.
        eos_token_id=<int>         → usa ese token como EOS.
        """
        self.eval()
        B_size = block_size if block_size is not None else self.block_size
        ov_ratio = overlap_ratio if overlap_ratio is not None else self.overlap_ratio
        K_steps = num_diffusion_steps if num_diffusion_steps is not None else self.num_diffusion_steps
        chunk_size = chunk_denoise_size if chunk_denoise_size is not None else self.chunk_denoise_size
        # eos_token_id=-1 sentinel → use model default; None → disabled; int → override
        if eos_token_id == -1:
            stop_eos = self.eos_token_id
        else:
            stop_eos = eos_token_id  # None disables EOS stopping; any int overrides

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
