import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import tracemalloc
import numpy as np
import torch
import torch.nn.functional as F

from pymbbo import build_model, Dataset


# =====================================================================
# 1. Synthetic Scaled Structured Sequence Generator (Seq Len = 1024)
# =====================================================================
def generate_scaled_synthetic_data(num_samples: int = 200, seq_len: int = 1024):
    """
    Generates synthetic token sequences with explicit long-range dependencies across 1024 tokens:
    Rule 1: If prompt starts with [1, 1, 1, 1] ('E's), tokens at pos 250..255 MUST be [2, 2, 2, 2, 2], and pos 750..755 MUST be [2, 2, 2, 2, 2].
    Rule 2: If prompt starts with [6, 6, 6, 6] ('X's), tokens at pos 250..255 MUST be [7, 7, 7, 7, 7], and pos 750..755 MUST be [7, 7, 7, 7, 7].
    Rule 3: If prompt starts with [3, 3, 3, 3] ('B's), tokens at pos 250..255 MUST be [8, 8, 8, 8, 8], and pos 750..755 MUST be [8, 8, 8, 8, 8].
    Filler tokens are sampled from {4, 5, 9, 10, 11}.
    """
    vocab_size = 16
    filler_tokens = [4, 5, 9, 10, 11]
    data = np.random.choice(filler_tokens, size=(num_samples, seq_len)).astype(np.int64)

    for i in range(num_samples):
        rule_choice = i % 3
        if rule_choice == 0:
            data[i, :4] = [1, 1, 1, 1]
            data[i, 250:255] = [2, 2, 2, 2, 2]
            data[i, 750:755] = [2, 2, 2, 2, 2]
        elif rule_choice == 1:
            data[i, :4] = [6, 6, 6, 6]
            data[i, 250:255] = [7, 7, 7, 7, 7]
            data[i, 750:755] = [7, 7, 7, 7, 7]
        else:
            data[i, :4] = [3, 3, 3, 3]
            data[i, 250:255] = [8, 8, 8, 8, 8]
            data[i, 750:755] = [8, 8, 8, 8, 8]

    X = data[:, :-1]
    Y = data[:, 1:]
    return X, Y, vocab_size


# =====================================================================
# 2. Scaled Benchmark Runner
# =====================================================================
def run_scaled_benchmark():
    print("=" * 95)
    print(" BENCHMARK ESCALADO (1024 TOKENS, ~500K PARÁMETROS): DENSE VS ASA (max_a=32, 128) ")
    print("=" * 95)

    seq_len = 1024
    X_train, Y_train, vocab_size = generate_scaled_synthetic_data(num_samples=150, seq_len=seq_len)
    X_test, Y_test, _ = generate_scaled_synthetic_data(num_samples=30, seq_len=seq_len)

    train_ds = Dataset(X_train, Y_train)

    # Architectural specs for ~500K parameters
    d_model = 128
    nhead = 4
    num_layers = 3
    max_seq_len = 1024

    configs = [
        {"name": "Standard Dense GPT",  "arch": "transformer",     "max_a": None},
        {"name": "ASA-GPT (max_a=32)",  "arch": "asa_transformer", "max_a": 32},
        {"name": "ASA-GPT (max_a=128)", "arch": "asa_transformer", "max_a": 128},
    ]

    results = []

    for idx, cfg in enumerate(configs, 1):
        name = cfg["name"]
        arch = cfg["arch"]
        max_a_val = cfg["max_a"]

        print(f"\n[{idx}/3] Entrenando y Evaluando: {name} ...")

        if arch == "transformer":
            model = build_model(
                arch,
                vocab_size=vocab_size,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                max_seq_len=max_seq_len
            )
        else:
            model = build_model(
                arch,
                vocab_size=vocab_size,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                max_seq_len=max_seq_len,
                group_size=2,
                max_a=max_a_val
            )

        model.compile(optimizer="adam", loss_function="cross_entropy", learning_rate=0.003)
        num_params = sum(p.numel() for p in model.parameters())

        # Fit for 2 quick epochs
        model.fit(train_ds, epochs=2, batch_size=16)

        # Perplexity Evaluation
        model.eval()
        with torch.no_grad():
            test_x = torch.from_numpy(X_test)
            test_y = torch.from_numpy(Y_test)
            if arch == "asa_transformer":
                logits = model(test_x, max_a=max_a_val)
            else:
                logits = model(test_x)
            loss = F.cross_entropy(logits.view(-1, vocab_size), test_y.reshape(-1))
            perplexity = torch.exp(loss).item()

        # Generation Speed, Consistency, and Memory Measurement
        tracemalloc.start()
        
        prompt = torch.tensor([[1, 1, 1, 1]], dtype=torch.int64)
        gen_tokens_count = 60
        num_runs = 4
        run_times = []

        for _ in range(num_runs):
            t0 = time.perf_counter()
            if arch == "asa_transformer":
                _ = model.architecture.generate(prompt, max_new_tokens=gen_tokens_count, max_a=max_a_val)
            else:
                _ = model.architecture.generate(prompt, max_new_tokens=gen_tokens_count)
            t1 = time.perf_counter()
            run_times.append(t1 - t0)

        current_ram, peak_ram = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        avg_gen_time = float(np.mean(run_times))
        std_gen_time = float(np.std(run_times))
        tokens_per_sec = gen_tokens_count / avg_gen_time
        peak_ram_mb = peak_ram / (1024 * 1024)

        results.append({
            "name": name,
            "params": num_params,
            "perplexity": perplexity,
            "gen_time_s": avg_gen_time,
            "std_time_s": std_gen_time,
            "tokens_sec": tokens_per_sec,
            "peak_ram_mb": peak_ram_mb
        })

    # Print Formatted Table
    print("\n" + "=" * 95)
    print(f"{'MODELO':<22} | {'PARÁMETROS':<11} | {'PERPLEXITY':<11} | {'TIEMPO (s)':<11} | {'CONSIST. (std)':<14} | {'TOK/SEC':<10} | {'RAM PICO (MB)':<12}")
    print("=" * 95)
    for r in results:
        print(f"{r['name']:<22} | {r['params']:<11,} | {r['perplexity']:<11.4f} | {r['gen_time_s']:<11.4f} | {r['std_time_s']:<14.5f} | {r['tokens_sec']:<10.2f} | {r['peak_ram_mb']:<12.3f}")
    print("=" * 95)


if __name__ == "__main__":
    run_scaled_benchmark()
