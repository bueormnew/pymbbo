import pytest
import torch
from pymbbo.models.factory import build_model
from pymbbo.models.registry import discover_architectures
from pymbbo.architectures.linear_block_diffusion.model import LinearBlockDiffusionArchitecture

def test_architecture_registration():
    """Verify linear_block_diffusion architecture is discovered by PYMBBO registry."""
    registry = discover_architectures()
    assert "linear_block_diffusion" in registry
    assert registry["linear_block_diffusion"] == LinearBlockDiffusionArchitecture

def test_build_model_factory():
    """Verify build_model instantiates LinearBlockDiffusionArchitecture."""
    model = build_model(
        "linear_block_diffusion",
        vocab_size=1000,
        d_model=64,
        num_layers=2,
        block_size=128,
        overlap_ratio=0.5,
        num_diffusion_steps=4
    )
    assert model is not None
    assert isinstance(model.architecture, LinearBlockDiffusionArchitecture)

def test_variable_prompt_lengths_and_forward_shapes():
    """Test model forward pass with variable input prompt lengths (20, 100, 500 tokens)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arch = LinearBlockDiffusionArchitecture(
        vocab_size=1000,
        d_model=64,
        num_layers=2,
        block_size=128,
        num_diffusion_steps=4
    ).to(device)

    prompt_lengths = [20, 100, 500]
    for p_len in prompt_lengths:
        prompt_ids = torch.randint(1, 900, (2, p_len), device=device)
        logits = arch(prompt_ids)
        assert logits.shape == (2, p_len, 1000)

def test_standard_dataset_training_loss():
    """Test forward pass with target_ids (standard autoregressive dataset training mode)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arch = LinearBlockDiffusionArchitecture(
        vocab_size=1000,
        d_model=64,
        num_layers=2,
        block_size=128,
        num_diffusion_steps=8
    ).to(device)

    prompt_ids = torch.randint(1, 900, (4, 30), device=device)
    target_ids = torch.randint(1, 900, (4, 128), device=device)

    logits, loss = arch(prompt_ids, target_ids=target_ids, return_logits=True)
    assert logits.shape == (4, 128, 1000)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() > 0

    loss_only = arch(prompt_ids, target_ids=target_ids, return_logits=False)
    assert isinstance(loss_only, torch.Tensor)
    assert not torch.isnan(loss_only)

    # Backward pass check
    loss.backward()
    for param in arch.parameters():
        if param.requires_grad and param.grad is not None:
            assert not torch.isnan(param.grad).any()

def test_progressive_overlapping_block_generation():
    """Test progressive sub-chunk unmasking generation over sliding window blocks."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arch = LinearBlockDiffusionArchitecture(
        vocab_size=1000,
        d_model=64,
        num_layers=2,
        block_size=128,
        overlap_ratio=0.5, # 64 overlap, 64 stride
        num_diffusion_steps=4,
        chunk_denoise_size=32
    ).to(device)

    prompt_ids = torch.randint(1, 900, (2, 25), device=device)
    max_new = 150

    generated = arch.generate(
        prompt_ids,
        max_new_tokens=max_new,
        temperature=1.0,
        eos_token_id=None,  # Disable EOS stopping so length is deterministic
    )

    # Generated sequence should have shape (batch_size, prompt_len + max_new)
    assert generated.shape[0] == 2
    assert generated.shape[1] == 25 + max_new
    # Confirm prompt tokens remain at start
    assert torch.equal(generated[:, :25], prompt_ids)


def test_dynamic_hyperparameter_overrides_in_generate():
    """Test dynamic hyperparameter overrides during inference call to generate()."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arch = LinearBlockDiffusionArchitecture(
        vocab_size=1000,
        d_model=64,
        num_layers=2,
        block_size=512,
        overlap_ratio=0.5,
        num_diffusion_steps=8,
        chunk_denoise_size=64
    ).to(device)

    prompt_ids = torch.randint(1, 900, (1, 15), device=device)

    # Override block_size=64, overlap_ratio=0.25, num_diffusion_steps=2, chunk_denoise_size=16
    generated = arch.generate(
        prompt_ids,
        max_new_tokens=80,
        block_size=64,
        overlap_ratio=0.25,
        num_diffusion_steps=2,
        chunk_denoise_size=16,
        eos_token_id=None
    )

    assert generated.shape == (1, 15 + 80)
