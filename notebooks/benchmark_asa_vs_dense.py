import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import tracemalloc
import numpy as np
import torch
import torch.nn.functional as F

from pymbbo import build_model, Dataset, load_dataset


# =====================================================================
# 1. Synthetic Structured Sequence Generator with Long-Range Rules
# =====================================================================
def generate_synthetic_structured_data(num_samples: int = 400, seq_len: int = 64):
    """
    Generates synthetic token sequences with explicit long-range dependencies:
    Rule 1: If prompt starts with [1, 1, 1] ('E's), tokens at pos 20..24 MUST be [2, 2, 2, 2, 2] ('A's).
    Rule 2: If prompt starts with [6, 6, 6] ('X's), tokens at pos 20..24 MUST be [7, 7, 7, 7, 7] ('Y's).
    Rule 3: If prompt starts with [3, 3, 3] ('B's), tokens at pos 20..24 MUST be [8, 8, 8, 8, 8] ('Z's).
    Filler tokens are randomly sampled from {4, 5, 9}.
    """
    vocab_size = 12
    filler_tokens = [4, 5, 9]
    data = np.random.choice(filler_tokens, size=(num_samples, seq_len)).astype(np.int64)

    for i in range(num_samples):
        rule_choice = i % 3
        if rule_choice == 0:
            # Rule 1: 'E' -> 'A'
            data[i, :3] = [1, 1, 1]
            data[i, 20:25] = [2, 2, 2, 2, 2]
            data[i, 40:43] = [2, 2, 2]
        elif rule_choice == 1:
            # Rule 2: 'X' -> 'Y'
            data[i, :3] = [6, 6, 6]
            data[i, 20:25] = [7, 7, 7, 7, 7]
            data[i, 40:43] = [7, 7, 7]
        else:
            # Rule 3: 'B' -> 'Z'
            data[i, :3] = [3, 3, 3]
            data[i, 20:25] = [8, 8, 8, 8, 8]
            data[i, 40:43] = [8, 8, 8]

    # Input X: tokens 0..N-2, Target Y: tokens 1..N-1
    X = data[:, :-1]
    Y = data[:, 1:]
    return X, Y, vocab_size


# =====================================================================
# 2. Benchmark Runner
# =====================================================================
def run_benchmark():
    print("=" * 80)
    print(" BENCHMARK REAL: TRANSFORMER STANDARD VS ASA TRANSFORMERS (ULTRA-SMALL) ")
    print("=" * 80)

    # 1. Dataset Generation
    seq_len = 64
    X_train, Y_train, vocab_size = generate_synthetic_structured_data(num_samples=300, seq_len=seq_len)
    X_test, Y_test, _ = generate_synthetic_structured_data(num_samples=60, seq_len=seq_len)
    
    train_ds = Dataset(X_train, Y_train)
    test_ds = Dataset(X_test, Y_test)

    # 2. Define Ultra-Small Models (< 300K parameters)
    d_model = 64
    nhead = 4
    num_layers = 4
    max_seq_len = 128

    configs = [
        {"name": "Standard Dense GPT", "arch": "transformer", "max_a": None},
        {"name": "ASA-GPT (max_a=4)",  "arch": "asa_transformer", "max_a": 4},
        {"name": "ASA-GPT (max_a=8)",  "arch": "asa_transformer", "max_a": 8},
        {"name": "ASA-GPT (max_a=16)", "arch": "asa_transformer", "max_a": 16},
    ]

    results = []

    for cfg in configs:
        name = cfg["name"]
        arch = cfg["arch"]
        max_a_val = cfg["max_a"]

        print(f"\n[1/4] Entrenando y Evaluando: {name} ...")
        
        # Build Model
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
                max_a=max_a_val or 16
            )

        model.compile(optimizer="adam", loss_function="cross_entropy", learning_rate=0.003)
        
        # Count Parameters
        num_params = sum(p.numel() for p in model.parameters())

        # Fit for 3 quick epochs
        model.fit(train_ds, epochs=3, batch_size=32)

        # Measure Perplexity on Test Set
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

        # Measure Generation Speed, Consistency, and RAM Consumption
        tracemalloc.start()
        
        prompt = torch.tensor([[1, 1, 1]], dtype=torch.int64) # Start with 'E's
        gen_tokens_count = 50
        num_runs = 5
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

    # Print Formatted Table Report
    print("\n" + "=" * 95)
    print(f"{'MODELO':<22} | {'PARÁMETROS':<11} | {'PERPLEXITY':<11} | {'TIEMPO (s)':<11} | {'CONSIST. (std)':<14} | {'TOK/SEC':<10} | {'RAM PICO (MB)':<12}")
    print("=" * 95)
    for r in results:
        print(f"{r['name']:<22} | {r['params']:<11,} | {r['perplexity']:<11.4f} | {r['gen_time_s']:<11.4f} | {r['std_time_s']:<14.5f} | {r['tokens_sec']:<10.2f} | {r['peak_ram_mb']:<12.3f}")
    print("=" * 95)


if __name__ == "__main__":
    run_benchmark()
