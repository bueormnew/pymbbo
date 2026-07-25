"""
Linear Block Diffusion — Kaggle Dual T4 GPU Training & Evaluation Script
Architecture: LinearBlockDiffusionArchitecture (~41.4M Parameters)
Hardware: Google Kaggle Dual NVIDIA Tesla T4 GPUs (DataParallel + FP16 amp)
Tokenizer: HuggingFace gpt2 Tokenizer (vocab_size = 50257)
Dataset: HuggingFace Open-Source Text Corpus (wikitext-2-raw-v1, 1024 tokens)
"""

import os
import sys
import math
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Optional, List, Tuple, Dict, Any

from datasets import load_dataset
from transformers import AutoTokenizer

print("=" * 75)
print("🖥️ KAGGLE DUAL T4 GPU HARDWARE DIAGNOSTIC")
print("=" * 75)
print(f"PyTorch Version : {torch.__version__}")
print(f"CUDA Available  : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} GPU(s):")
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"  [GPU {i}] {props.name} | VRAM: {props.total_memory / 1e9:.2f} GB")
else:
    num_gpus = 0
    print("⚠️ WARNING: CUDA not detected. Ensure Kaggle GPU Accelerator is enabled.")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Architecture Definition ───────────────────────────────────────────
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: Optional[int] = None):
        super().__init__()
        if d_ff is None:
            d_ff = int(8 / 3 * d_model)
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class LinearRecurrentLayer(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.w_in = nn.Linear(d_model, d_model)
        self.w_gate = nn.Linear(d_model, d_model)
        self.w_out = nn.Linear(d_model, d_model)
    def forward(self, x, h_prev=None, reverse=False):
        batch_size, seq_len, _ = x.shape
        h = h_prev if h_prev is not None else torch.zeros(batch_size, self.d_model, device=x.device, dtype=x.dtype)
        proj_x = self.w_in(x)
        gate_x = torch.sigmoid(self.w_gate(x))
        hidden_states = [None] * seq_len
        indices = range(seq_len - 1, -1, -1) if reverse else range(seq_len)
        for t in indices:
            g_t = gate_x[:, t, :]
            h = g_t * h + (1.0 - g_t) * proj_x[:, t, :]
            hidden_states[t] = h
        return self.w_out(torch.stack(hidden_states, dim=1)), h

class BidirectionalLinearRecurrentBlock(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.fwd_recurrent = LinearRecurrentLayer(d_model)
        self.bwd_recurrent = LinearRecurrentLayer(d_model)
        self.combine_proj = nn.Linear(d_model * 2, d_model)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        normed_x = self.norm1(x)
        fwd_out, _ = self.fwd_recurrent(normed_x, reverse=False)
        bwd_out, _ = self.bwd_recurrent(normed_x, reverse=True)
        recurrent_out = self.combine_proj(torch.cat([fwd_out, bwd_out], dim=-1))
        x = x + self.dropout(recurrent_out)
        return x + self.dropout(self.ffn(self.norm2(x)))

class SinusoidalTimestepEmbedding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.mlp = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.SiLU(), nn.Linear(d_model * 2, d_model))
    def forward(self, timesteps):
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
    def __init__(self, d_model: int):
        super().__init__()
        self.prompt_proj = nn.Linear(d_model, d_model)
        self.target_proj = nn.Linear(d_model, d_model)
        self.gate_proj = nn.Linear(d_model * 2, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
    def forward(self, block_x, prompt_h):
        if prompt_h is None:
            return block_x
        prompt_ctx = prompt_h.mean(dim=1) if prompt_h.dim() == 3 else prompt_h
        prompt_expanded = prompt_ctx.unsqueeze(1).expand_as(block_x)
        gate = torch.sigmoid(self.gate_proj(torch.cat([block_x, prompt_expanded], dim=-1)))
        fused = gate * self.target_proj(block_x) + (1.0 - gate) * self.prompt_proj(prompt_expanded)
        return block_x + self.out_proj(fused)

class DiffusionBlockRefiner(nn.Module):
    def __init__(self, d_model: int, num_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.time_embedder = SinusoidalTimestepEmbedding(d_model)
        self.context_fusion = LinearContextCrossFusion(d_model)
        self.layers = nn.ModuleList([BidirectionalLinearRecurrentBlock(d_model, dropout=dropout) for _ in range(num_layers)])
        self.final_norm = RMSNorm(d_model)
    def forward(self, block_embeddings, timesteps, prompt_context=None):
        time_emb = self.time_embedder(timesteps).unsqueeze(1)
        x = block_embeddings + time_emb
        x = self.context_fusion(x, prompt_context)
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)

class LinearBlockDiffusionArchitecture(nn.Module):
    def __init__(self, vocab_size=50257, d_model=512, num_layers=6, block_size=512, overlap_ratio=0.5, num_diffusion_steps=8, chunk_denoise_size=64, mask_token_id=None, pad_token_id=50256, eos_token_id=50256, dropout=0.1, max_seq_len=8192, noise_injection_prob=0.15, **kwargs):
        super().__init__()
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

        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.prompt_encoder = nn.ModuleList([BidirectionalLinearRecurrentBlock(d_model, dropout=dropout) for _ in range(2)])
        self.prompt_norm = RMSNorm(d_model)
        self.refiner = DiffusionBlockRefiner(d_model=d_model, num_layers=num_layers, dropout=dropout)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def encode_prompt(self, prompt_ids):
        batch_size, seq_len = prompt_ids.shape
        pos = torch.arange(seq_len, device=prompt_ids.device).unsqueeze(0)
        x = self.token_embedding(prompt_ids) + self.pos_embedding(pos)
        for layer in self.prompt_encoder:
            x = layer(x)
        return self.prompt_norm(x)

    def forward(self, x, target_ids=None, timestep=None, noise_injection_prob=None, **kwargs):
        prompt_ids = x
        batch_size, prompt_len = prompt_ids.shape
        prompt_ctx = self.encode_prompt(prompt_ids)

        if target_ids is None:
            pos = torch.arange(prompt_len, device=prompt_ids.device).unsqueeze(0)
            emb = self.token_embedding(prompt_ids) + self.pos_embedding(pos)
            t_steps = torch.ones(batch_size, device=prompt_ids.device) * self.num_diffusion_steps
            refined = self.refiner(emb, t_steps, prompt_ctx)
            return self.lm_head(refined)

        target_batch, target_len = target_ids.shape
        t_k = torch.randint(1, self.num_diffusion_steps + 1, (target_batch,), device=target_ids.device) if timestep is None else timestep
        alpha_k = 1.0 - (t_k.float() / float(self.num_diffusion_steps))
        p_noise = noise_injection_prob if noise_injection_prob is not None else self.noise_injection_prob

        target_noisy = target_ids.clone()
        for i in range(target_batch):
            num_to_mask = int(target_len * alpha_k[i].item())
            if num_to_mask > 0:
                mask_indices = torch.randperm(target_len, device=target_ids.device)[:num_to_mask]
                target_noisy[i, mask_indices] = self.mask_token_id
            if p_noise > 0 and self.training:
                unmasked_indices = (target_noisy[i] != self.mask_token_id).nonzero(as_tuple=True)[0]
                if len(unmasked_indices) > 0:
                    num_corrupt = int(len(unmasked_indices) * p_noise)
                    if num_corrupt > 0:
                        corrupt_subset = unmasked_indices[torch.randperm(len(unmasked_indices), device=target_ids.device)[:num_corrupt]]
                        target_noisy[i, corrupt_subset] = torch.randint(1, self.vocab_size - 1, (num_corrupt,), device=target_ids.device)

        pos = torch.arange(target_len, device=target_ids.device).unsqueeze(0)
        target_emb = self.token_embedding(target_noisy) + self.pos_embedding(pos)
        refined_emb = self.refiner(target_emb, t_k, prompt_ctx)
        logits = self.lm_head(refined_emb)
        loss = F.cross_entropy(logits.view(-1, self.vocab_size), target_ids.view(-1), ignore_index=self.pad_token_id)
        return logits, loss

    @torch.no_grad()
    def generate(self, prompt_ids, max_new_tokens=512, block_size=None, overlap_ratio=None, num_diffusion_steps=None, chunk_denoise_size=None, temperature=1.0, top_k=50, eos_token_id=None):
        self.eval()
        B_size = block_size if block_size is not None else self.block_size
        ov_ratio = overlap_ratio if overlap_ratio is not None else self.overlap_ratio
        K_steps = num_diffusion_steps if num_diffusion_steps is not None else self.num_diffusion_steps
        chunk_size = chunk_denoise_size if chunk_denoise_size is not None else self.chunk_denoise_size
        stop_eos = eos_token_id if eos_token_id is not None else self.eos_token_id

        batch_size, prompt_len = prompt_ids.shape
        device = prompt_ids.device
        prompt_ctx = self.encode_prompt(prompt_ids)
        overlap_len = int(B_size * ov_ratio)
        stride = B_size - overlap_len

        generated_seq = prompt_ids.clone()
        tokens_generated = 0
        refined_overlap_prefix = None
        eos_found = False

        while tokens_generated < max_new_tokens and not eos_found:
            current_new_len = min(stride, max_new_tokens - tokens_generated)
            current_block_len = (overlap_len if refined_overlap_prefix is not None else 0) + current_new_len
            block_tokens = torch.full((batch_size, current_block_len), fill_value=self.mask_token_id, dtype=torch.long, device=device)
            if refined_overlap_prefix is not None:
                block_tokens[:, :overlap_len] = refined_overlap_prefix

            for k in range(1, K_steps + 1):
                unmask_limit = (overlap_len if refined_overlap_prefix is not None else 0) + min(k * chunk_size, current_new_len)
                pos = torch.arange(current_block_len, device=device).unsqueeze(0)
                emb = self.token_embedding(block_tokens) + self.pos_embedding(pos)
                t_k = torch.full((batch_size,), fill_value=k, device=device, dtype=torch.long)
                refined_emb = self.refiner(emb, t_k, prompt_ctx)
                logits = self.lm_head(refined_emb)

                unmask_start = overlap_len if refined_overlap_prefix is not None else 0
                if unmask_limit > unmask_start:
                    sub_logits = logits[:, unmask_start:unmask_limit, :] / max(temperature, 1e-5)
                    if top_k > 0:
                        v, _ = torch.topk(sub_logits, min(top_k, sub_logits.size(-1)))
                        sub_logits[sub_logits < v[:, :, [-1]]] = -float('Inf')
                    probs = F.softmax(sub_logits, dim=-1)
                    sampled_tokens = torch.multinomial(probs.view(-1, self.vocab_size), num_samples=1).view(batch_size, -1)
                    block_tokens[:, unmask_start:unmask_limit] = sampled_tokens

            newly_refined_tokens = block_tokens[:, (overlap_len if refined_overlap_prefix is not None else 0):]
            if stop_eos is not None and (newly_refined_tokens == stop_eos).any():
                eos_found = True
                eos_mask = (newly_refined_tokens == stop_eos)
                first_eos_idx = (eos_mask.cumsum(dim=1) == 1).nonzero(as_tuple=False)
                if len(first_eos_idx) > 0:
                    cut_off = first_eos_idx[0, 1].item() + 1
                    newly_refined_tokens = newly_refined_tokens[:, :cut_off]

            generated_seq = torch.cat([generated_seq, newly_refined_tokens], dim=1)
            tokens_generated += newly_refined_tokens.shape[1]
            if block_tokens.shape[1] >= overlap_len:
                refined_overlap_prefix = block_tokens[:, -overlap_len:].clone()

        return generated_seq

# ── Data & Tokenizer ──────────────────────────────────────────────────
print("=" * 75)
print("🤗 DOWNLOADING HUGGINGFACE DATASET & GPT-2 TOKENIZER")
print("=" * 75)

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
VOCAB_SIZE = tokenizer.vocab_size

raw_dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
train_texts = [text for text in raw_dataset["train"]["text"] if len(text.strip()) > 50]
val_texts = [text for text in raw_dataset["validation"]["text"] if len(text.strip()) > 50]

SEQ_LEN = 1024
PROMPT_LEN = 64

def prepare_huggingface_chunks(texts, num_samples=8000):
    prompts = []
    targets = []
    full_ids = []
    for text in texts:
        tokens = tokenizer.encode(text)
        full_ids.extend(tokens)
        if len(full_ids) >= (num_samples * 256):
            break
    stride = 256
    total_len = PROMPT_LEN + SEQ_LEN
    for i in range(0, len(full_ids) - total_len, stride):
        p = full_ids[i : i + PROMPT_LEN]
        t = full_ids[i + PROMPT_LEN : i + total_len]
        prompts.append(p)
        targets.append(t)
        if len(prompts) >= num_samples:
            break
    return torch.tensor(prompts, dtype=torch.long), torch.tensor(targets, dtype=torch.long)

train_p, train_t = prepare_huggingface_chunks(train_texts, num_samples=8000)
val_p, val_t = prepare_huggingface_chunks(val_texts, num_samples=1000)

class HFTextDataset(Dataset):
    def __init__(self, prompts, targets):
        self.prompts = prompts
        self.targets = targets
    def __len__(self):
        return len(self.prompts)
    def __getitem__(self, idx):
        return self.prompts[idx], self.targets[idx]

train_loader = DataLoader(HFTextDataset(train_p, train_t), batch_size=8, shuffle=True, drop_last=True)
val_loader = DataLoader(HFTextDataset(val_p, val_t), batch_size=8, shuffle=False, drop_last=True)

# ── Model & Multi-GPU ─────────────────────────────────────────────────
print("=" * 75)
print("🏗️ BUILDING 41.4M PARAMETER MODEL FOR GOOGLE KAGGLE DUAL T4 GPUs")
print("=" * 75)

raw_model = LinearBlockDiffusionArchitecture(
    vocab_size=VOCAB_SIZE,
    d_model=512,
    num_layers=6,
    block_size=512,
    overlap_ratio=0.5,
    num_diffusion_steps=8,
    chunk_denoise_size=64,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    noise_injection_prob=0.15
)

raw_model = raw_model.to(DEVICE)
model = nn.DataParallel(raw_model) if num_gpus > 1 else raw_model

# ── Training & Benchmarking ───────────────────────────────────────────
print("=" * 75)
print("⚡ TRAINING & BENCHMARKING METRICS")
print("=" * 75)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01, betas=(0.9, 0.95))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-5)
scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

EPOCHS = 2
max_steps_per_epoch = 100

total_tokens_trained = 0
total_train_time = 0.0
step_count = 0

model.train()
for epoch in range(1, EPOCHS + 1):
    for step, (p_batch, t_batch) in enumerate(train_loader):
        if step >= max_steps_per_epoch:
            break
        step_start_time = time.perf_counter()
        p_batch = p_batch.to(DEVICE)
        t_batch = t_batch.to(DEVICE)
        batch_tokens = t_batch.numel()
        
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            logits, loss = model(p_batch, target_ids=t_batch)
            if isinstance(model, nn.DataParallel):
                loss = loss.mean()
                
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        step_duration = time.perf_counter() - step_start_time
        total_train_time += step_duration
        total_tokens_trained += batch_tokens
        step_count += 1
        
        tokens_per_sec = batch_tokens / step_duration
        ms_per_step = step_duration * 1000.0
        max_vram_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        
        if (step + 1) % 25 == 0 or step == 0:
            print(f"Step [{step+1:3d}/{max_steps_per_epoch}] | Loss: {loss.item():.4f} | PPL: {math.exp(min(loss.item(), 20.0)):7.2f} | Time: {ms_per_step:6.2f} ms/step | Speed: {tokens_per_sec:8.1f} tok/s | Peak VRAM: {max_vram_gb:.2f} GB")

avg_train_tokens_per_sec = total_tokens_trained / total_train_time
avg_ms_per_step = (total_train_time / step_count) * 1000.0

# ── Validation & Perplexity ───────────────────────────────────────────
model.eval()
val_loss_sum = 0.0
val_steps = 0
with torch.no_grad():
    for p_batch, t_batch in val_loader:
        p_batch, t_batch = p_batch.to(DEVICE), t_batch.to(DEVICE)
        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            logits, loss = model(p_batch, target_ids=t_batch)
            if isinstance(model, nn.DataParallel):
                loss = loss.mean()
        val_loss_sum += loss.item()
        val_steps += 1
        if val_steps >= 50:
            break

avg_val_loss = val_loss_sum / max(val_steps, 1)
val_perplexity = math.exp(min(avg_val_loss, 20.0))

# ── Inference & Generation Evaluation ─────────────────────────────────
eval_model = raw_model.to(DEVICE)
eval_model.eval()

prompt_text = "In a distant world, scientists discovered"
prompt_tokens = tokenizer.encode(prompt_text, return_tensors="pt").to(DEVICE)

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

infer_start_time = time.perf_counter()
generated_ids = eval_model.generate(prompt_tokens, max_new_tokens=512, block_size=512, overlap_ratio=0.5, num_diffusion_steps=8, chunk_denoise_size=64, temperature=0.8, top_k=40)
if torch.cuda.is_available():
    torch.cuda.synchronize()

infer_duration = time.perf_counter() - infer_start_time
generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
generated_token_count = generated_ids.shape[1] - prompt_tokens.shape[1]
infer_tokens_per_sec = generated_token_count / infer_duration
num_blocks = math.ceil(generated_token_count / 256)
ms_per_block = (infer_duration / num_blocks) * 1000.0
infer_peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0

print("=" * 75)
print("🏆 COMPREHENSIVE BENCHMARK REPORT -- LINEAR BLOCK DIFFUSION (41.4M PARAMS)")
print("=" * 75)
metrics_data = {
    "Métrica de Evaluación": [
        "Velocidad de Entrenamiento (Tokens/sec)",
        "Velocidad de Inferencia (Tokens/sec)",
        "Uso Máximo VRAM GPU (Entrenamiento)",
        "Uso Máximo VRAM GPU (Inferencia)",
        "Tiempo por Paso de Entrenamiento (ms/step)",
        "Tiempo por Bloque Generado (ms/block)",
        "Pérdida de Validación (Validation Loss)",
        "Perplejidad de Validación (PPL)"
    ],
    "Valor Medido": [
        f"{avg_train_tokens_per_sec:,.2f} tok/s",
        f"{infer_tokens_per_sec:,.2f} tok/s",
        f"{max_vram_gb:.2f} GB",
        f"{infer_peak_vram_gb:.2f} GB",
        f"{avg_ms_per_step:.2f} ms/paso",
        f"{ms_per_block:.2f} ms/bloque",
        f"{avg_val_loss:.4f}",
        f"{val_perplexity:.2f}"
    ]
}
df_report = pd.DataFrame(metrics_data)
print(df_report.to_string(index=False))
