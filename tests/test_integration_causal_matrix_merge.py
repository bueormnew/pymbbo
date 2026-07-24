"""
Integration test for Causal Matrix Merge architecture with pymbbo ecosystem.

Validates requirements 5.1-5.5, 7.2:
  - build_model("causal_matrix_merge", **kwargs) works correctly
  - discover_architectures() includes "causal_matrix_merge"
  - BaseModel.compile() works with optimizer/loss
  - One training step works (forward + loss + backward)
  - Logits are compatible with nn.CrossEntropyLoss
  - save_model / load_model round-trip works
  - generate() produces tokens of expected shape
"""

import os
import sys
import tempfile

import torch
import torch.nn as nn

# Ensure project root is on sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# ─── Model configuration shared across tests ─────────────────────────────────
MODEL_KWARGS = dict(
    vocab_size=500,
    model_dim=64,
    state_dim=32,
    num_slots=4,
    num_layers=2,
    num_checkpoints=2,
    checkpoint_stride=8,
    write_rank=2,
    dropout=0.0,
    use_residual_gate=True,
    max_context=128,
)


def _build_test_model():
    """Helper to build a test model instance."""
    from pymbbo.models.factory import build_model
    return build_model("causal_matrix_merge", **MODEL_KWARGS)


def test_discover_architectures():
    """Test that discover_architectures() includes 'causal_matrix_merge'."""
    from pymbbo.models.registry import discover_architectures

    registry = discover_architectures()
    assert "causal_matrix_merge" in registry, (
        f"'causal_matrix_merge' not found in registry. Available: {list(registry.keys())}"
    )


def test_build_model():
    """Test that build_model instantiates correctly and returns a BaseModel."""
    from pymbbo.models.factory import build_model
    from pymbbo.models.base import BaseModel

    model = _build_test_model()

    assert isinstance(model, BaseModel), f"Expected BaseModel, got {type(model)}"
    assert model.architecture is not None, "architecture attribute should not be None"
    assert hasattr(model.architecture, "ARCH_NAME"), "architecture missing ARCH_NAME"
    assert model.architecture.ARCH_NAME == "causal_matrix_merge"


def test_compile():
    """Test that BaseModel.compile() works with optimizer and loss."""
    model = _build_test_model()
    model.compile(
        optimizer="adamw",
        loss_function="cross_entropy",
        learning_rate=3e-4,
    )
    assert model.is_compiled, "Model should be marked as compiled"
    assert model.optimizer is not None, "Optimizer should not be None after compile"
    assert model.loss_fn is not None, "Loss function should not be None after compile"


def test_training_step():
    """Test one full training step: forward + loss + backward."""
    model = _build_test_model()
    model.compile(optimizer="adamw", loss_function="cross_entropy", learning_rate=3e-4)
    model.train()

    batch_size = 4
    seq_len = 16
    vocab_size = 500

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    target_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Forward pass
    logits = model(input_ids)
    assert logits.shape == (batch_size, seq_len, vocab_size), (
        f"Expected logits shape {(batch_size, seq_len, vocab_size)}, got {logits.shape}"
    )

    # Compute loss
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(logits.view(-1, vocab_size), target_ids.view(-1))
    assert loss.dim() == 0, "Loss should be a scalar"
    assert not torch.isnan(loss), "Loss should not be NaN"
    assert not torch.isinf(loss), "Loss should not be Inf"

    # Backward pass
    model.optimizer.zero_grad()
    loss.backward()

    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters() if p.requires_grad)
    assert has_grad, "At least some parameters should have non-zero gradients"

    model.optimizer.step()


def test_logits_crossentropy_compatible():
    """Test that logits are reshapeable for nn.CrossEntropyLoss."""
    model = _build_test_model()
    model.eval()

    batch_size = 8
    seq_len = 32
    vocab_size = 500

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    target_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        logits = model(input_ids)

    flat_logits = logits.view(-1, vocab_size)
    assert flat_logits.shape == (batch_size * seq_len, vocab_size), (
        f"Expected flat shape {(batch_size * seq_len, vocab_size)}, got {flat_logits.shape}"
    )

    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(flat_logits, target_ids.view(-1))
    assert not torch.isnan(loss), "Loss should not be NaN"


def test_save_load_roundtrip():
    """Test save_model / load_model round-trip preserves state_dict."""
    from pymbbo.models.persistence import save_model, load_model

    model = _build_test_model()
    model.eval()

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        filepath = f.name

    try:
        save_model(model, filepath)
        assert os.path.exists(filepath), "Saved file should exist"

        loaded_model = load_model(filepath)
        assert loaded_model is not None, "Loaded model should not be None"
        assert loaded_model.architecture is not None, "Loaded model architecture should not be None"

        # Compare state_dicts
        original_sd = model.state_dict()
        loaded_sd = loaded_model.state_dict()

        assert set(original_sd.keys()) == set(loaded_sd.keys()), "State dict keys mismatch"

        for key in original_sd:
            assert torch.allclose(original_sd[key], loaded_sd[key], atol=1e-6), (
                f"State dict mismatch for key '{key}'"
            )

        # Verify loaded model can do inference
        input_ids = torch.randint(0, 500, (2, 8))
        with torch.no_grad():
            logits_orig = model(input_ids)
            logits_loaded = loaded_model(input_ids)

        assert torch.allclose(logits_orig, logits_loaded, atol=1e-5), (
            "Inference results differ between original and loaded model"
        )
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


def test_generate():
    """Test that generate() produces tokens of expected shape."""
    model = _build_test_model()
    model.eval()

    batch_size = 2
    prompt_len = 4
    max_new_tokens = 10
    vocab_size = 500

    prompt = torch.randint(0, vocab_size, (batch_size, prompt_len))

    generated = model.architecture.generate(
        prompt_ids=prompt,
        max_new_tokens=max_new_tokens,
        temperature=1.0,
        top_k=50,
    )

    expected_total_len = prompt_len + max_new_tokens
    assert generated.shape == (batch_size, expected_total_len), (
        f"Expected shape {(batch_size, expected_total_len)}, got {generated.shape}"
    )

    # Verify all generated tokens are valid token IDs
    assert (generated >= 0).all(), "All tokens should be >= 0"
    assert (generated < vocab_size).all(), f"All tokens should be < vocab_size ({vocab_size})"

    # Verify prompt is preserved at the start
    assert torch.equal(generated[:, :prompt_len], prompt), "Prompt tokens should be preserved"


def main():
    print("=" * 70)
    print("  INTEGRATION TEST: Causal Matrix Merge + pymbbo Ecosystem")
    print("=" * 70)

    tests = [
        ("discover_architectures() includes 'causal_matrix_merge'", test_discover_architectures),
        ("build_model('causal_matrix_merge', **kwargs) works", test_build_model),
        ("BaseModel.compile() works", test_compile),
        ("One training step (forward + loss + backward)", test_training_step),
        ("Logits compatible with nn.CrossEntropyLoss", test_logits_crossentropy_compatible),
        ("save_model / load_model round-trip", test_save_load_roundtrip),
        ("generate() produces tokens of expected shape", test_generate),
    ]

    for i, (desc, fn) in enumerate(tests, 1):
        print(f"\n[TEST {i}] {desc}...")
        fn()
        print(f"  ✓ PASSED")

    print("\n" + "=" * 70)
    print("  ALL 7 INTEGRATION TESTS PASSED ✓")
    print("=" * 70)


if __name__ == "__main__":
    main()
