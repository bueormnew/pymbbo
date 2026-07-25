"""
Linear Block Diffusion — Kaggle Dual T4 GPU Training & Evaluation Script
Architecture: LinearBlockDiffusionArchitecture from pymbbo (~41.4M Parameters)
Hardware: Google Kaggle Dual NVIDIA Tesla T4 GPUs (DataParallel + FP16 amp)
Tokenizer: HuggingFace GPT2TokenizerFast (vocab_size = 50257)
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
from transformers import GPT2TokenizerFast

# Patch typing compatibility for Python 3.12 builtins in Kaggle
import typing, builtins
for _n in ['Tuple', 'List', 'Dict', 'Optional', 'Union', 'Any', 'Callable']:
    if not hasattr(builtins, _n):
        setattr(builtins, _n, getattr(typing, _n))

from pymbbo.architectures.linear_block_diffusion import LinearBlockDiffusionArchitecture

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

# ── Tokenizer & Dataset ───────────────────────────────────────────────
print("\n" + "=" * 75)
print("🤗 TOKENIZER & DATASET SETUP")
print("=" * 75)

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
VOCAB_SIZE = len(tokenizer)  # 50257
print(f"Loaded GPT-2 Tokenizer: vocab_size = {VOCAB_SIZE:,}")

raw_dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
train_texts = [t for t in raw_dataset["train"]["text"] if len(t.strip()) > 50]
val_texts = [t for t in raw_dataset["validation"]["text"] if len(t.strip()) > 50]
print(f"Filtered Dataset Samples: Train={len(train_texts):,} | Val={len(val_texts):,}")

SEQ_LEN = 1024
PROMPT_LEN = 64
TOTAL_LEN = PROMPT_LEN + SEQ_LEN

def tokenize_and_chunk(texts, max_samples=8000):
    all_ids = []
    for text in texts:
        all_ids.extend(tokenizer.encode(text))
        if len(all_ids) >= max_samples * 512:
            break
    prompts, targets = [], []
    stride = 256
    for i in range(0, len(all_ids) - TOTAL_LEN, stride):
        prompts.append(all_ids[i : i + PROMPT_LEN])
        targets.append(all_ids[i + PROMPT_LEN : i + TOTAL_LEN])
        if len(prompts) >= max_samples:
            break
    return torch.tensor(prompts, dtype=torch.long), torch.tensor(targets, dtype=torch.long)

print("\nTokenizing and chunking dataset into (64 prompt, 1024 target) sequence pairs...")
train_p, train_t = tokenize_and_chunk(train_texts, max_samples=8000)
val_p, val_t = tokenize_and_chunk(val_texts, max_samples=1000)

class TextPairDataset(Dataset):
    def __init__(self, prompts, targets):
        self.prompts = prompts
        self.targets = targets
    def __len__(self):
        return len(self.prompts)
    def __getitem__(self, idx):
        return self.prompts[idx], self.targets[idx]

BATCH_SIZE = 8
train_loader = DataLoader(TextPairDataset(train_p, train_t), batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(TextPairDataset(val_p, val_t), batch_size=BATCH_SIZE, shuffle=False, drop_last=True, num_workers=2, pin_memory=True)

print(f"DataLoaders Created: Train={len(train_p):,} | Val={len(val_p):,} | Batch Size={BATCH_SIZE}")

# ── Model Instantiation ───────────────────────────────────────────────
print("\n" + "=" * 75)
print("🏗️ CONSTRUCTING LinearBlockDiffusion MODEL")
print("=" * 75)

D_MODEL = 512
NUM_LAYERS = 6
BLOCK_SIZE = 512
OVERLAP_RATIO = 0.5
NUM_DIFFUSION_STEPS = 8
CHUNK_DENOISE_SIZE = 64

raw_model = LinearBlockDiffusionArchitecture(
    vocab_size=VOCAB_SIZE,
    d_model=D_MODEL,
    num_layers=NUM_LAYERS,
    block_size=BLOCK_SIZE,
    overlap_ratio=OVERLAP_RATIO,
    num_diffusion_steps=NUM_DIFFUSION_STEPS,
    chunk_denoise_size=CHUNK_DENOISE_SIZE,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    noise_injection_prob=0.15,
    dropout=0.1,
)

num_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
print(f"Trainable Parameters: {num_params / 1e6:.2f}M")

raw_model = raw_model.to(DEVICE)
if num_gpus > 1:
    print(f"🔗 Enabling DataParallel across {num_gpus} GPUs")
    model = nn.DataParallel(raw_model)
else:
    model = raw_model

# Warmup scan kernel before training loop
with torch.no_grad():
    _wp = torch.randint(0, 100, (1, 8)).to(DEVICE)
    _wt = torch.randint(0, 100, (1, 32)).to(DEVICE)
    raw_model(_wp, target_ids=_wt, return_logits=False)
    del _wp, _wt
if torch.cuda.is_available():
    torch.cuda.synchronize()

print(f"✅ Model successfully placed on {DEVICE}")

# ── Training Loop ─────────────────────────────────────────────────────
print("\n" + "=" * 75)
print("⚡ TRAINING LOOP — MEASURING TOKENS/S, VRAM, MS/STEP")
print("=" * 75)

EPOCHS = 3
LR = 3e-4
WARMUP_STEPS = 50
MAX_STEPS_PER_EPOCH = 200
NAN_PATIENCE = 5

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01, betas=(0.9, 0.95))
total_steps = EPOCHS * min(MAX_STEPS_PER_EPOCH, len(train_loader))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)
scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

total_tokens_trained = 0
total_train_time = 0.0
step_count = 0
global_step = 0
train_losses = []
peak_vram_train = 0.0
nan_streak = 0
aborted = False

model.train()
for epoch in range(1, EPOCHS + 1):
    if aborted: break
    epoch_loss, epoch_steps = 0.0, 0
    print(f"\n{'─'*75}  EPOCH {epoch}/{EPOCHS}")

    for step, (p_batch, t_batch) in enumerate(train_loader):
        if step >= MAX_STEPS_PER_EPOCH or aborted: break

        if global_step < WARMUP_STEPS:
            warmup_factor = (global_step + 1) / WARMUP_STEPS
            for pg in optimizer.param_groups:
                pg['lr'] = LR * warmup_factor

        t0 = time.perf_counter()
        p_batch = p_batch.to(DEVICE, non_blocking=True)
        t_batch = t_batch.to(DEVICE, non_blocking=True)
        batch_tokens = t_batch.numel()

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            loss = model(p_batch, target_ids=t_batch, return_logits=False)
            if isinstance(model, nn.DataParallel):
                loss = loss.mean()

        if not torch.isfinite(loss):
            nan_streak += 1
            print(f"  ⚠️ NaN loss at step {step+1} (streak {nan_streak}/{NAN_PATIENCE}) — skipping")
            if nan_streak >= NAN_PATIENCE:
                print("  🛑 Too many consecutive NaNs — aborting training")
                aborted = True
            optimizer.zero_grad(set_to_none=True)
            continue
        nan_streak = 0

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        if not torch.isfinite(grad_norm):
            print(f"  ⚠️ NaN/Inf gradient norm at step {step+1} — skipping optimizer step")
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            continue

        scaler.step(optimizer)
        scaler.update()
        if global_step >= WARMUP_STEPS:
            scheduler.step()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        total_train_time += dt
        total_tokens_trained += batch_tokens
        step_count += 1
        global_step += 1
        l = loss.item()
        epoch_loss += l
        epoch_steps += 1
        train_losses.append(l)

        if torch.cuda.is_available():
            peak_vram_train = max(peak_vram_train, torch.cuda.max_memory_allocated() / 1e9)

        if (step + 1) % 25 == 0 or step == 0:
            ppl = math.exp(min(l, 20.0))
            cur_lr = optimizer.param_groups[0]['lr']
            print(f"  [{step+1:3d}/{MAX_STEPS_PER_EPOCH}] "
                  f"Loss: {l:.4f} | PPL: {ppl:8.2f} | "
                  f"{dt*1000:6.1f} ms/step | {batch_tokens/dt:>7,.0f} tok/s | "
                  f"VRAM: {peak_vram_train:.2f} GB | LR: {cur_lr:.2e}")

    if epoch_steps > 0:
        avg = epoch_loss / epoch_steps
        print(f"  → Epoch {epoch} avg loss: {avg:.4f} | PPL: {math.exp(min(avg, 20)):.2f}")

if not aborted:
    avg_tok_s = total_tokens_trained / total_train_time
    avg_ms_per_step = (total_train_time / step_count) * 1000
    print(f"\n{'='*75}")
    print(f"✅ Training Finished: {step_count} steps | {avg_tok_s:,.0f} tok/s | {avg_ms_per_step:.1f} ms/step | VRAM: {peak_vram_train:.2f} GB")

# ── Validation Evaluation ─────────────────────────────────────────────
print("\n" + "=" * 75)
print("📉 VALIDATION EVALUATION")
print("=" * 75)

model.eval()
val_loss_sum, val_steps = 0.0, 0
MAX_VAL_STEPS = 50

with torch.no_grad():
    for p_batch, t_batch in val_loader:
        if val_steps >= MAX_VAL_STEPS: break
        p_batch = p_batch.to(DEVICE, non_blocking=True)
        t_batch = t_batch.to(DEVICE, non_blocking=True)
        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            loss = model(p_batch, target_ids=t_batch, return_logits=False)
            if isinstance(model, nn.DataParallel):
                loss = loss.mean()
        if torch.isfinite(loss):
            val_loss_sum += loss.item()
            val_steps += 1

avg_val_loss = val_loss_sum / max(val_steps, 1)
val_ppl = math.exp(min(avg_val_loss, 20.0))
print(f"  Validation Loss : {avg_val_loss:.4f}")
print(f"  Validation PPL  : {val_ppl:.2f}")
print(f"  Evaluated Steps : {val_steps}")

# ── Generation Benchmark ──────────────────────────────────────────────
print("\n" + "=" * 75)
print("🎯 INFERENCE GENERATION BENCHMARK")
print("=" * 75)

eval_model = raw_model
eval_model.eval()

test_prompts = [
    "In a distant world, scientists discovered",
    "The history of artificial intelligence began",
    "Once upon a time in a small village",
]

MAX_NEW_TOKENS = 256
all_infer_times, all_infer_tokens = [], []

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

for i, prompt_text in enumerate(test_prompts):
    prompt_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(DEVICE)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t0 = time.perf_counter()

    generated_ids = eval_model.generate(
        prompt_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        block_size=512,
        overlap_ratio=0.5,
        num_diffusion_steps=8,
        chunk_denoise_size=64,
        temperature=0.8,
        top_k=40,
        eos_token_id=None,
    )

    if torch.cuda.is_available(): torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    gen_count = generated_ids.shape[1] - prompt_ids.shape[1]
    all_infer_times.append(dt)
    all_infer_tokens.append(gen_count)
    decoded = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    print(f"\n{'─'*75}")
    print(f"  Sample {i+1} | Prompt: \"{prompt_text}\"")
    print(f"  {gen_count} tokens | {dt:.2f}s | {gen_count/dt:.1f} tok/s")
    print(f"{'─'*75}")
    print(decoded[:600])

total_infer_tok = sum(all_infer_tokens)
total_infer_time = sum(all_infer_times)
avg_infer_tok_s = total_infer_tok / total_infer_time
total_blocks = sum(math.ceil(t / 256) for t in all_infer_tokens)
avg_ms_per_block = (total_infer_time / total_blocks) * 1000
infer_peak_vram = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0

print(f"\n{'='*75}")
print(f"📊 Inference Summary: {avg_infer_tok_s:.1f} tok/s | {avg_ms_per_block:.1f} ms/block | Peak VRAM: {infer_peak_vram:.2f} GB")

# ── Final Performance Report ──────────────────────────────────────────
print("\n" + "=" * 75)
print("🏆 FINAL PERFORMANCE REPORT — LINEAR BLOCK DIFFUSION")
print("=" * 75)

df_config = pd.DataFrame({
    "Parameter": ["Params", "d_model", "Layers", "Vocab Size", "Block Size", "Diffusion Steps K", "Epochs", "Batch Size", "Learning Rate"],
    "Value": [f"{num_params / 1e6:.2f}M", str(D_MODEL), str(NUM_LAYERS), f"{VOCAB_SIZE:,}", str(BLOCK_SIZE), str(NUM_DIFFUSION_STEPS), str(EPOCHS), str(BATCH_SIZE), str(LR)]
})
print("\n📋 Configuration")
print(df_config.to_string(index=False))

_last_loss = train_losses[-1] if train_losses else float('nan')
df_perf = pd.DataFrame({
    "Metric": [
        "⚡ Train tok/s", "⚡ Training ms/step", "💾 Peak VRAM (train)",
        "📉 Final Train Loss", "📉 Final Train PPL",
        "✅ Validation Loss", "✅ Validation PPL",
        "🚀 Inference tok/s", "🚀 Inference ms/block", "💾 Peak VRAM (infer)",
    ],
    "Value": [
        f"{avg_tok_s:,.0f} tok/s" if not aborted else "N/A",
        f"{avg_ms_per_step:.1f} ms" if not aborted else "N/A",
        f"{peak_vram_train:.2f} GB",
        f"{_last_loss:.4f}", f"{math.exp(min(_last_loss, 20)):.2f}",
        f"{avg_val_loss:.4f}", f"{val_ppl:.2f}",
        f"{avg_infer_tok_s:.1f} tok/s", f"{avg_ms_per_block:.1f} ms",
        f"{infer_peak_vram:.2f} GB",
    ]
})
print("\n📊 Performance & Evaluation Metrics")
print(df_perf.to_string(index=False))

print(f"\n{'='*75}")
print("✅ Kaggle Benchmark Complete.")
print("=" * 75)
