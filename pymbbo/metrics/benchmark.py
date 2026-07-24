import time
import math
import torch
from typing import Dict, List, Any, Union, Optional
from pymbbo.models.base import BaseModel

def token_scaling_benchmark(
    model: BaseModel,
    prompt_tokens: Optional[torch.Tensor] = None,
    vocab_size: int = 1000,
    min_tokens: int = 100,
    max_tokens: int = 1000,
    steps: int = 4,
    cost_per_million_tokens: float = 0.002,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Executes automatic token growth/scaling benchmarking.
    Measures total time, token throughput (tokens/sec), per-token latency (ms),
    and estimated inference cost across token sequence lengths.
    """
    model.eval()
    model.to(device)

    # Generate test tokens scale
    step_sizes = []
    if steps == 1:
        step_sizes = [max_tokens]
    else:
        step_sizes = [int(min_tokens + i * (max_tokens - min_tokens) / (steps - 1)) for i in range(steps)]

    if prompt_tokens is None:
        prompt_tokens = torch.randint(0, vocab_size, (1, 10), device=device)
    else:
        prompt_tokens = prompt_tokens.to(device)

    results = {
        "token_counts": [],
        "execution_times_sec": [],
        "tokens_per_second": [],
        "latency_per_token_ms": [],
        "estimated_costs_usd": []
    }

    print("=" * 75)
    print(f"{'PYMBBO Token Scaling Benchmark':^75}")
    print("=" * 75)
    print(f"{'Target Tokens':<15} | {'Time (s)':<12} | {'Tokens/sec':<15} | {'Latency/Token':<15} | {'Est. Cost ($)':<12}")
    print("-" * 75)

    for n_tokens in step_sizes:
        start_time = time.perf_counter()
        
        # Perform auto-regressive or scaled token generation simulation
        with torch.no_grad():
            if hasattr(model.architecture, "generate"):
                _ = model.architecture.generate(prompt_tokens, max_new_tokens=n_tokens)
            else:
                # Simulates token scaling for general sequence models
                seq = torch.randint(0, vocab_size, (1, n_tokens), device=device)
                _ = model(seq)

        elapsed = time.perf_counter() - start_time
        tps = n_tokens / max(elapsed, 1e-6)
        latency_ms = (elapsed / n_tokens) * 1000
        cost = (n_tokens / 1_000_000) * cost_per_million_tokens

        results["token_counts"].append(n_tokens)
        results["execution_times_sec"].append(round(elapsed, 4))
        results["tokens_per_second"].append(round(tps, 2))
        results["latency_per_token_ms"].append(round(latency_ms, 3))
        results["estimated_costs_usd"].append(round(cost, 6))

        n_str = f"{n_tokens:,}"
        print(f"{n_str:<15} | {elapsed:<12.4f} | {tps:<15.2f} | {latency_ms:<12.3f} ms | ${cost:<11.6f}")

    print("=" * 75)

    return results


def compare_models(
    models: Dict[str, BaseModel],
    test_data: Any,
    batch_size: int = 32,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Executes simultaneous benchmarking and side-by-side comparison of 2 or more models.
    Compares parameter count, inference latency, throughput, loss, and metric scores.
    """
    comparison_report = {}

    print("=" * 80)
    print(f"{'PYMBBO Simultaneous Model Benchmarking & Comparison':^80}")
    print("=" * 80)
    print(f"{'Model Name':<20} | {'Params':<12} | {'Latency (s)':<12} | {'Samples/sec':<15} | {'Loss':<10}")
    print("-" * 80)

    for name, model in models.items():
        model.eval()
        model.to(device)

        total_params = sum(p.numel() for p in model.parameters())

        # Measure evaluation performance
        start_time = time.perf_counter()
        eval_res = model.evaluate(test_data, batch_size=batch_size) if hasattr(model, "evaluate") and model.is_compiled else {}
        elapsed = time.perf_counter() - start_time

        num_samples = len(test_data) if hasattr(test_data, "__len__") else 100
        sps = num_samples / max(elapsed, 1e-6)
        loss_val = eval_res.get("loss", 0.0)

        comparison_report[name] = {
            "total_parameters": total_params,
            "inference_time_sec": round(elapsed, 4),
            "samples_per_second": round(sps, 2),
            "metrics": eval_res
        }

        print(f"{name:<20} | {total_params:<12,} | {elapsed:<12.4f} | {sps:<15.2f} | {loss_val:<10.4f}")

    print("=" * 80)
    return comparison_report
