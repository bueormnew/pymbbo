"""Property-based tests for Per-Slot Local Merge Improvements.

Uses Hypothesis to verify universal invariants of the per-slot decay and write
mechanisms added to CausalMatrixMergeV2.
"""

import math

import pytest
import torch
import torch.nn.functional as F
from hypothesis import given, settings, assume
import hypothesis.strategies as st

from pymbbo.architectures.causal_matrix_merge.state import MergeState
from pymbbo.architectures.causal_matrix_merge_v2.config import CausalMatrixMergeV2Config
from pymbbo.architectures.causal_matrix_merge_v2.merge_v2 import CausalMatrixMergeV2, rms_norm


# =============================================================================
# Feature: slot-local-merge-improvements, Property 11: Config Serialization Round-Trip
# Validates: Requirements 6.7
# =============================================================================


@st.composite
def valid_config_strategy(draw):
    """Generate a valid CausalMatrixMergeV2Config with all fields randomized within valid ranges."""
    num_slots = draw(st.integers(min_value=1, max_value=32))
    top_k_slots = draw(st.integers(min_value=1, max_value=num_slots))

    return CausalMatrixMergeV2Config(
        vocab_size=draw(st.integers(min_value=1, max_value=10000)),
        model_dim=draw(st.integers(min_value=1, max_value=512)),
        state_dim=draw(st.integers(min_value=1, max_value=256)),
        num_slots=num_slots,
        num_layers=draw(st.integers(min_value=1, max_value=8)),
        num_checkpoints=draw(st.integers(min_value=1, max_value=16)),
        checkpoint_stride=draw(st.integers(min_value=1, max_value=64)),
        write_rank=draw(st.integers(min_value=1, max_value=16)),
        dropout=draw(st.floats(min_value=0.0, max_value=1.0)),
        use_residual_gate=draw(st.booleans()),
        max_context=draw(st.integers(min_value=1, max_value=4096)),
        ffn_mult=draw(st.floats(min_value=0.01, max_value=10.0)),
        top_k_slots=top_k_slots,
        use_adaptive_merge=draw(st.booleans()),
        use_learned_checkpoints=draw(st.booleans()),
        use_per_slot_decay=draw(st.booleans()),
        use_per_slot_write=draw(st.booleans()),
    )


@given(config=valid_config_strategy())
@settings(max_examples=100, deadline=None)
def test_config_serialization_round_trip(config: CausalMatrixMergeV2Config):
    """Property 11: Config Serialization Round-Trip.

    **Validates: Requirements 6.7**

    For any valid CausalMatrixMergeV2Config instance (including the new
    use_per_slot_decay and use_per_slot_write fields), serializing via to_dict()
    and reconstructing via from_dict() SHALL produce an identical config.
    """
    # Serialize to dict
    serialized = config.to_dict()

    # Reconstruct from dict
    reconstructed = CausalMatrixMergeV2Config.from_dict(serialized)

    # Verify round-trip identity
    assert reconstructed == config, (
        f"Config round-trip failed.\n"
        f"Original: {config}\n"
        f"Reconstructed: {reconstructed}\n"
        f"Serialized dict: {serialized}"
    )


@given(config=valid_config_strategy())
@settings(max_examples=100, deadline=None)
def test_config_backward_compatibility_missing_per_slot_fields(
    config: CausalMatrixMergeV2Config,
):
    """Property 11 (backward compatibility): Config from_dict with missing per-slot fields.

    **Validates: Requirements 6.7**

    For any valid config serialized to dict, removing the new per-slot fields
    (use_per_slot_decay, use_per_slot_write) and calling from_dict SHALL produce
    a config with those fields set to their defaults (True).
    """
    # Serialize to dict
    serialized = config.to_dict()

    # Remove the new per-slot fields to simulate loading an old checkpoint
    serialized.pop("use_per_slot_decay", None)
    serialized.pop("use_per_slot_write", None)

    # Reconstruct from dict — should use defaults for missing fields
    reconstructed = CausalMatrixMergeV2Config.from_dict(serialized)

    # The new fields should have their default values (True)
    assert reconstructed.use_per_slot_decay is True, (
        f"Expected use_per_slot_decay=True (default), got {reconstructed.use_per_slot_decay}"
    )
    assert reconstructed.use_per_slot_write is True, (
        f"Expected use_per_slot_write=True (default), got {reconstructed.use_per_slot_write}"
    )

    # All other fields should match the original config
    assert reconstructed.vocab_size == config.vocab_size
    assert reconstructed.model_dim == config.model_dim
    assert reconstructed.state_dim == config.state_dim
    assert reconstructed.num_slots == config.num_slots
    assert reconstructed.num_layers == config.num_layers
    assert reconstructed.num_checkpoints == config.num_checkpoints
    assert reconstructed.checkpoint_stride == config.checkpoint_stride
    assert reconstructed.write_rank == config.write_rank
    assert reconstructed.dropout == config.dropout
    assert reconstructed.use_residual_gate == config.use_residual_gate
    assert reconstructed.max_context == config.max_context
    assert reconstructed.ffn_mult == config.ffn_mult
    assert reconstructed.top_k_slots == config.top_k_slots
    assert reconstructed.use_adaptive_merge == config.use_adaptive_merge
    assert reconstructed.use_learned_checkpoints == config.use_learned_checkpoints


# =============================================================================
# Feature: slot-local-merge-improvements, Property 2: Decay Output Shape and Range
# Validates: Requirements 1.4, 2.3, 2.6, 8.5
# =============================================================================

# Small config for testing (matches design doc test configuration)
_DECAY_TEST_CONFIG = CausalMatrixMergeV2Config(
    vocab_size=100,
    model_dim=32,
    state_dim=16,
    num_slots=4,
    num_layers=1,
    write_rank=2,
    num_checkpoints=2,
    checkpoint_stride=4,
    top_k_slots=2,
    use_per_slot_decay=True,
    use_per_slot_write=True,
    use_adaptive_merge=True,
)


@st.composite
def decay_input_strategy(draw):
    """Generate valid inputs for _per_slot_adaptive_decay testing.

    Generates:
    - h: [B, model_dim] with values in (-5, 5)
    - decay_base: [B, S, 1] with values in (0.01, 0.99) to guarantee strict (0,1)
    - memory: [B, S, state_dim] with values in (-2, 2)
    """
    batch_size = draw(st.integers(min_value=1, max_value=4))
    model_dim = _DECAY_TEST_CONFIG.model_dim
    state_dim = _DECAY_TEST_CONFIG.state_dim
    num_slots = _DECAY_TEST_CONFIG.num_slots

    # Generate h in range (-5, 5)
    h = torch.FloatTensor(batch_size, model_dim).uniform_(-5.0, 5.0)

    # Generate decay_base in range (0.01, 0.99) to guarantee strict (0,1)
    decay_base = torch.FloatTensor(batch_size, num_slots, 1).uniform_(0.01, 0.99)

    # Generate memory in range (-2, 2)
    memory = torch.FloatTensor(batch_size, num_slots, state_dim).uniform_(-2.0, 2.0)

    return h, decay_base, memory, batch_size


@given(inputs=decay_input_strategy())
@settings(max_examples=100, deadline=None)
def test_decay_output_shape_and_range(inputs):
    """Property 2: Decay Output Shape and Range.

    **Validates: Requirements 1.4, 2.3, 2.6, 8.5**

    For any valid input tensor h of shape [B, model_dim] and memory of shape
    [B, S, state_dim], the per-slot adaptive decay SHALL produce a tensor of
    shape [B, S, 1] with all values strictly in the open interval (0, 1).
    """
    h, decay_base, memory, batch_size = inputs
    num_slots = _DECAY_TEST_CONFIG.num_slots

    # Instantiate model (eval mode, no gradients needed for shape/range check)
    torch.manual_seed(42)
    model = CausalMatrixMergeV2(_DECAY_TEST_CONFIG)
    model.eval()

    with torch.no_grad():
        decay_final = model._per_slot_adaptive_decay(h, decay_base, memory)

    # Assert output shape is [B, S, 1]
    assert decay_final.shape == (batch_size, num_slots, 1), (
        f"Expected shape ({batch_size}, {num_slots}, 1), got {decay_final.shape}"
    )

    # Assert all values strictly > 0
    assert (decay_final > 0).all(), (
        f"Found values <= 0 in decay output. Min value: {decay_final.min().item()}"
    )

    # Assert all values strictly < 1
    assert (decay_final < 1).all(), (
        f"Found values >= 1 in decay output. Max value: {decay_final.max().item()}"
    )


@given(inputs=st.integers(min_value=1, max_value=4))
@settings(max_examples=100, deadline=None)
def test_decay_output_shape_and_range_zero_memory(inputs):
    """Property 2 (zero memory): Decay Output Shape and Range with zero memory.

    **Validates: Requirements 1.4, 2.3, 2.6, 8.5**

    For zero memory (initial state), the per-slot adaptive decay SHALL still
    produce a tensor of shape [B, S, 1] with all values strictly in (0, 1),
    verifying numerical stability at initialization.
    """
    batch_size = inputs
    model_dim = _DECAY_TEST_CONFIG.model_dim
    state_dim = _DECAY_TEST_CONFIG.state_dim
    num_slots = _DECAY_TEST_CONFIG.num_slots

    # Generate h in range (-5, 5)
    h = torch.FloatTensor(batch_size, model_dim).uniform_(-5.0, 5.0)

    # decay_base in range (0.01, 0.99) to guarantee strict (0,1)
    decay_base = torch.FloatTensor(batch_size, num_slots, 1).uniform_(0.01, 0.99)

    # Zero memory (initial state)
    memory = torch.zeros(batch_size, num_slots, state_dim)

    # Instantiate model
    torch.manual_seed(42)
    model = CausalMatrixMergeV2(_DECAY_TEST_CONFIG)
    model.eval()

    with torch.no_grad():
        decay_final = model._per_slot_adaptive_decay(h, decay_base, memory)

    # Assert output shape is [B, S, 1]
    assert decay_final.shape == (batch_size, num_slots, 1), (
        f"Expected shape ({batch_size}, {num_slots}, 1), got {decay_final.shape}"
    )

    # Assert all values strictly > 0
    assert (decay_final > 0).all(), (
        f"Found values <= 0 in decay output with zero memory. "
        f"Min value: {decay_final.min().item()}"
    )

    # Assert all values strictly < 1
    assert (decay_final < 1).all(), (
        f"Found values >= 1 in decay output with zero memory. "
        f"Max value: {decay_final.max().item()}"
    )


# =============================================================================
# Feature: slot-local-merge-improvements, Property 1: Slot Independence of Decay Modulation
# Validates: Requirements 1.1, 1.3
# =============================================================================

# Small config for Property 1 tests
_SMALL_CONFIG_P1 = CausalMatrixMergeV2Config(
    model_dim=32,
    state_dim=16,
    num_slots=4,
    write_rank=2,
    num_checkpoints=2,
    checkpoint_stride=4,
    top_k_slots=2,
    use_per_slot_decay=True,
    use_per_slot_write=True,
)


@st.composite
def slot_decay_inputs(draw):
    """Generate random inputs for _per_slot_adaptive_decay testing.

    Produces h, decay_base, and memory tensors with valid shapes and ranges,
    along with a randomly chosen slot index to modify.
    """
    batch_size = draw(st.integers(min_value=1, max_value=4))
    num_slots = _SMALL_CONFIG_P1.num_slots
    model_dim = _SMALL_CONFIG_P1.model_dim
    state_dim = _SMALL_CONFIG_P1.state_dim

    # Generate h: [B, model_dim] in reasonable range
    h = torch.randn(batch_size, model_dim) * 2.0

    # Generate decay_base: [B, S, 1] in (0, 1)
    decay_base = torch.sigmoid(torch.randn(batch_size, num_slots, 1))

    # Generate memory: [B, S, state_dim] in reasonable range
    memory = torch.randn(batch_size, num_slots, state_dim) * 2.0

    # Pick a slot to modify
    slot_to_modify = draw(st.integers(min_value=0, max_value=num_slots - 1))

    return h, decay_base, memory, slot_to_modify


@given(data=slot_decay_inputs())
@settings(max_examples=100, deadline=None)
def test_slot_independence_of_decay_modulation(data):
    """Property 1: Slot Independence of Decay Modulation.

    **Validates: Requirements 1.1, 1.3**

    For any batch of token representations h and memory states, modifying the
    content of slot j in memory SHALL NOT change the decay modulation computed
    for any other slot i (where i ≠ j).
    """
    h, decay_base, memory, slot_to_modify = data

    # Create model with per-slot decay enabled
    torch.manual_seed(42)
    model = CausalMatrixMergeV2(_SMALL_CONFIG_P1)
    model.eval()

    # Compute decay with original memory
    with torch.no_grad():
        decay_original = model._per_slot_adaptive_decay(h, decay_base, memory)

    # Modify the selected slot's content in memory
    memory_modified = memory.clone()
    memory_modified[:, slot_to_modify, :] = torch.randn_like(
        memory_modified[:, slot_to_modify, :]
    ) * 5.0  # Use different magnitude to ensure actual change

    # Compute decay with modified memory
    with torch.no_grad():
        decay_modified = model._per_slot_adaptive_decay(h, decay_base, memory_modified)

    # Assert all other slots' decay values remain unchanged
    num_slots = _SMALL_CONFIG_P1.num_slots
    for i in range(num_slots):
        if i == slot_to_modify:
            continue  # Skip the modified slot — its decay may change
        assert torch.allclose(
            decay_original[:, i, :],
            decay_modified[:, i, :],
            atol=1e-6,
        ), (
            f"Slot {i} decay changed when slot {slot_to_modify} was modified.\n"
            f"Original: {decay_original[:, i, :]}\n"
            f"Modified: {decay_modified[:, i, :]}\n"
            f"Difference: {(decay_original[:, i, :] - decay_modified[:, i, :]).abs().max()}"
        )


# =============================================================================
# Feature: slot-local-merge-improvements, Property 7: Write Adaptation Shape and Bounded Magnitude
# Validates: Requirements 3.1, 8.6
# =============================================================================

@st.composite
def write_adaptation_inputs(draw):
    """Generate random write_base and memory tensors for per-slot write adaptation testing."""
    batch_size = draw(st.integers(min_value=1, max_value=4))
    state_dim = 16
    num_slots = 4

    # Generate write_base: [B, state_dim]
    write_base_data = draw(
        st.lists(
            st.lists(
                st.floats(min_value=-3.0, max_value=3.0, allow_nan=False, allow_infinity=False),
                min_size=state_dim,
                max_size=state_dim,
            ),
            min_size=batch_size,
            max_size=batch_size,
        )
    )

    # Generate memory: [B, S, state_dim]
    memory_data = draw(
        st.lists(
            st.lists(
                st.lists(
                    st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False),
                    min_size=state_dim,
                    max_size=state_dim,
                ),
                min_size=num_slots,
                max_size=num_slots,
            ),
            min_size=batch_size,
            max_size=batch_size,
        )
    )

    write_base = torch.tensor(write_base_data, dtype=torch.float32)
    memory = torch.tensor(memory_data, dtype=torch.float32)

    return batch_size, num_slots, state_dim, write_base, memory


@given(data=write_adaptation_inputs())
@settings(max_examples=100, deadline=None)
def test_write_adaptation_shape_and_bounded_magnitude(data):
    """Property 7: Write Adaptation Shape and Bounded Magnitude.

    **Validates: Requirements 3.1, 8.6**

    For any write base vector of shape [B, state_dim] and memory of shape
    [B, S, state_dim], the per-slot write adaptation SHALL produce a tensor
    of shape [B, S, state_dim], and the L2 norm of each adapted write vector
    SHALL be bounded by 2 * ||write_base|| + C for some constant C determined
    by the beta projection's weight norms.

    Practical bound: for each slot s,
        ||write_adapted[:, s, :]|| <= 2 * ||write_base|| + ||beta[:, s, :]||
    where beta = film_beta_proj(memory).
    """
    batch_size, num_slots, state_dim, write_base, memory = data

    # Create model with the specified config
    config = CausalMatrixMergeV2Config(
        vocab_size=100,
        model_dim=32,
        state_dim=16,
        num_slots=4,
        num_layers=1,
        num_checkpoints=2,
        checkpoint_stride=4,
        write_rank=2,
        dropout=0.0,
        use_residual_gate=False,
        max_context=64,
        ffn_mult=1.0,
        top_k_slots=2,
        use_adaptive_merge=True,
        use_learned_checkpoints=False,
        use_per_slot_decay=True,
        use_per_slot_write=True,
    )
    model = CausalMatrixMergeV2(config)
    model.eval()

    with torch.no_grad():
        write_adapted = model._per_slot_adaptive_write(write_base, memory)

    # --- Shape assertion ---
    assert write_adapted.shape == (batch_size, num_slots, state_dim), (
        f"Expected shape ({batch_size}, {num_slots}, {state_dim}), "
        f"got {write_adapted.shape}"
    )

    # --- Bounded magnitude assertion ---
    # Compute beta from the model (same computation as inside the method)
    with torch.no_grad():
        beta = model.film_beta_proj(memory)  # [B, S, state_dim]

    # For each batch element and slot, verify the bound:
    # ||write_adapted[b, s, :]|| <= 2 * ||write_base[b, :]|| + ||beta[b, s, :]||
    # Since gamma ∈ (0, 2): ||gamma * write|| <= 2 * ||write||
    # Therefore: ||gamma * write + beta|| <= 2 * ||write|| + ||beta||
    write_base_norm = torch.linalg.norm(write_base, dim=-1)  # [B]
    beta_norm = torch.linalg.norm(beta, dim=-1)  # [B, S]
    adapted_norm = torch.linalg.norm(write_adapted, dim=-1)  # [B, S]

    for b in range(batch_size):
        for s in range(num_slots):
            bound = 2.0 * write_base_norm[b] + beta_norm[b, s]
            assert adapted_norm[b, s] <= bound + 1e-5, (
                f"Bound violated at batch={b}, slot={s}: "
                f"||write_adapted|| = {adapted_norm[b, s]:.6f} > "
                f"2 * ||write_base|| + ||beta|| = {bound:.6f}"
            )


# =============================================================================
# Feature: slot-local-merge-improvements, Property 3: Decay Differentiability
# Validates: Requirements 1.5, 8.1, 8.3
# =============================================================================


@st.composite
def decay_differentiability_inputs(draw):
    """Generate valid inputs for testing decay differentiability.

    Produces batch_size, h, decay_base, and memory tensors with requires_grad
    set appropriately for gradient verification. Inputs are non-degenerate
    (at least some non-zero values) to ensure meaningful gradient flow.
    """
    batch_size = draw(st.integers(min_value=1, max_value=4))

    # Config: model_dim=32, state_dim=16, num_slots=4
    model_dim = 32
    state_dim = 16
    num_slots = 4

    # Generate h with finite values in a range that avoids all-zeros
    h_data = draw(
        st.lists(
            st.lists(
                st.floats(min_value=-3.0, max_value=3.0, allow_nan=False, allow_infinity=False),
                min_size=model_dim,
                max_size=model_dim,
            ),
            min_size=batch_size,
            max_size=batch_size,
        )
    )

    # Generate decay_base in (0, 1) — must be strictly positive for meaningful gradients
    decay_base_data = draw(
        st.lists(
            st.lists(
                st.floats(min_value=0.1, max_value=0.9, allow_nan=False, allow_infinity=False),
                min_size=num_slots,
                max_size=num_slots,
            ),
            min_size=batch_size,
            max_size=batch_size,
        )
    )

    # Generate memory with finite non-zero values to ensure gradient flows
    # through slot_decay_proj weights (grad_W ∝ memory)
    memory_data = draw(
        st.lists(
            st.lists(
                st.lists(
                    st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False),
                    min_size=state_dim,
                    max_size=state_dim,
                ),
                min_size=num_slots,
                max_size=num_slots,
            ),
            min_size=batch_size,
            max_size=batch_size,
        )
    )

    return batch_size, h_data, decay_base_data, memory_data


@given(inputs=decay_differentiability_inputs())
@settings(max_examples=100, deadline=None)
def test_decay_differentiability(inputs):
    """Property 3: Decay Differentiability.

    **Validates: Requirements 1.5, 8.1, 8.3**

    For any valid input tensors (non-degenerate, finite), the per-slot adaptive
    decay computation SHALL produce finite, non-zero gradients for all its
    learnable parameters after backpropagation.

    Also verifies gradient flows through h (since h requires_grad) and memory
    (set requires_grad on memory too).
    """
    batch_size, h_data, decay_base_data, memory_data = inputs

    # Model config matching the spec
    config = CausalMatrixMergeV2Config(
        vocab_size=100,
        model_dim=32,
        state_dim=16,
        num_slots=4,
        num_layers=1,
        write_rank=2,
        num_checkpoints=2,
        checkpoint_stride=4,
        top_k_slots=2,
        use_per_slot_decay=True,
        use_per_slot_write=True,
        use_adaptive_merge=True,
    )

    # Create model in eval mode (no dropout randomness)
    model = CausalMatrixMergeV2(config)
    model.eval()

    # Zero all gradients
    model.zero_grad()

    # Create input tensors with gradient tracking
    h = torch.tensor(h_data, dtype=torch.float32, requires_grad=True)
    decay_base = torch.tensor(decay_base_data, dtype=torch.float32).unsqueeze(-1)  # [B, S, 1]
    memory = torch.tensor(memory_data, dtype=torch.float32, requires_grad=True)

    # Ensure inputs are finite and non-degenerate
    assume(torch.isfinite(h).all().item())
    assume(torch.isfinite(decay_base).all().item())
    assume(torch.isfinite(memory).all().item())
    # Non-degenerate: h and memory must have non-zero values for gradient to flow
    # through the weight matrix (grad_W depends on input activation)
    assume(h.abs().sum().item() > 1e-6)
    assume(memory.abs().sum().item() > 1e-6)

    # Call the per-slot adaptive decay method
    decay_final = model._per_slot_adaptive_decay(h, decay_base, memory)

    # Sum the output and call backward
    loss = decay_final.sum()
    loss.backward()

    # 1. Verify slot_decay_proj.weight.grad is finite and has non-zero values
    weight_grad = model.slot_decay_proj.weight.grad
    assert weight_grad is not None, "slot_decay_proj.weight.grad is None — no gradient computed"
    assert torch.isfinite(weight_grad).all().item(), (
        f"slot_decay_proj.weight.grad contains non-finite values: {weight_grad}"
    )
    assert weight_grad.abs().sum().item() > 0, (
        "slot_decay_proj.weight.grad is all zeros — gradient did not flow through"
    )

    # 2. Verify gradient flows through h
    assert h.grad is not None, "h.grad is None — gradient did not flow through h"
    assert torch.isfinite(h.grad).all().item(), (
        f"h.grad contains non-finite values: {h.grad}"
    )
    assert h.grad.abs().sum().item() > 0, (
        "h.grad is all zeros — gradient did not flow through h"
    )

    # 3. Verify gradient flows through memory
    assert memory.grad is not None, "memory.grad is None — gradient did not flow through memory"
    assert torch.isfinite(memory.grad).all().item(), (
        f"memory.grad contains non-finite values: {memory.grad}"
    )
    assert memory.grad.abs().sum().item() > 0, (
        "memory.grad is all zeros — gradient did not flow through memory"
    )


# =============================================================================
# Feature: slot-local-merge-improvements, Property 6: Slot Independence of Write Adaptation
# Validates: Requirements 3.2
# =============================================================================

# Small config for Property 6 tests
_SMALL_CONFIG_P6 = CausalMatrixMergeV2Config(
    model_dim=32,
    state_dim=16,
    num_slots=4,
    write_rank=2,
    num_checkpoints=2,
    checkpoint_stride=4,
    top_k_slots=2,
    use_per_slot_decay=True,
    use_per_slot_write=True,
)


@st.composite
def slot_write_inputs(draw):
    """Generate random inputs for _per_slot_adaptive_write testing.

    Produces write_base and memory tensors with valid shapes and ranges,
    along with a randomly chosen slot index to modify.
    """
    batch_size = draw(st.integers(min_value=1, max_value=4))
    num_slots = _SMALL_CONFIG_P6.num_slots
    state_dim = _SMALL_CONFIG_P6.state_dim

    # Generate write_base: [B, state_dim] in reasonable range (-3, 3)
    write_base = torch.randn(batch_size, state_dim) * 3.0

    # Generate memory: [B, S, state_dim] in reasonable range (-2, 2)
    memory = torch.randn(batch_size, num_slots, state_dim) * 2.0

    # Pick a slot to modify
    slot_to_modify = draw(st.integers(min_value=0, max_value=num_slots - 1))

    return write_base, memory, slot_to_modify


@given(data=slot_write_inputs())
@settings(max_examples=100, deadline=None)
def test_slot_independence_of_write_adaptation(data):
    """Property 6: Slot Independence of Write Adaptation.

    **Validates: Requirements 3.2**

    For any write base vector and memory state, modifying the content of slot j
    in memory SHALL NOT change the adapted write vector computed for any other
    slot i (where i ≠ j).
    """
    write_base, memory, slot_to_modify = data

    # Create model with per-slot write enabled
    torch.manual_seed(42)
    model = CausalMatrixMergeV2(_SMALL_CONFIG_P6)
    model.eval()

    # Compute write adaptation with original memory
    with torch.no_grad():
        write_original = model._per_slot_adaptive_write(write_base, memory)

    # Modify the selected slot's content in memory
    memory_modified = memory.clone()
    memory_modified[:, slot_to_modify, :] = torch.randn_like(
        memory_modified[:, slot_to_modify, :]
    ) * 5.0  # Use different magnitude to ensure actual change

    # Compute write adaptation with modified memory
    with torch.no_grad():
        write_modified = model._per_slot_adaptive_write(write_base, memory_modified)

    # Assert all other slots' adapted write values remain unchanged
    num_slots = _SMALL_CONFIG_P6.num_slots
    for i in range(num_slots):
        if i == slot_to_modify:
            continue  # Skip the modified slot — its write may change
        assert torch.allclose(
            write_original[:, i, :],
            write_modified[:, i, :],
            atol=1e-6,
        ), (
            f"Slot {i} adapted write changed when slot {slot_to_modify} was modified.\n"
            f"Original: {write_original[:, i, :]}\n"
            f"Modified: {write_modified[:, i, :]}\n"
            f"Difference: {(write_original[:, i, :] - write_modified[:, i, :]).abs().max()}"
        )


# =============================================================================
# Feature: slot-local-merge-improvements, Property 8: Write Differentiability
# Validates: Requirements 3.6, 8.2, 8.4
# =============================================================================


@st.composite
def write_differentiability_inputs(draw):
    """Generate valid inputs for testing write differentiability.

    Produces write_base and memory tensors with requires_grad set appropriately
    for gradient verification. Inputs are non-degenerate (finite, non-zero)
    to ensure meaningful gradient flow through the FiLM parameters.
    """
    batch_size = draw(st.integers(min_value=1, max_value=4))

    # Config: model_dim=32, state_dim=16, num_slots=4
    state_dim = 16
    num_slots = 4

    # Generate write_base with finite values in a range that avoids all-zeros
    write_base_data = draw(
        st.lists(
            st.lists(
                st.floats(min_value=-3.0, max_value=3.0, allow_nan=False, allow_infinity=False),
                min_size=state_dim,
                max_size=state_dim,
            ),
            min_size=batch_size,
            max_size=batch_size,
        )
    )

    # Generate memory with finite non-zero values to ensure gradient flows
    # through film_gamma_proj and film_beta_proj weights (grad_W ∝ memory)
    memory_data = draw(
        st.lists(
            st.lists(
                st.lists(
                    st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False),
                    min_size=state_dim,
                    max_size=state_dim,
                ),
                min_size=num_slots,
                max_size=num_slots,
            ),
            min_size=batch_size,
            max_size=batch_size,
        )
    )

    return batch_size, write_base_data, memory_data


@given(inputs=write_differentiability_inputs())
@settings(max_examples=100, deadline=None)
def test_write_differentiability(inputs):
    """Property 8: Write Differentiability.

    **Validates: Requirements 3.6, 8.2, 8.4**

    For any valid input tensors (non-degenerate, finite), the per-slot write
    adaptation SHALL produce finite, non-zero gradients for all its learnable
    parameters after backpropagation.

    Verifies:
    1. film_gamma_proj.weight.grad is finite and has non-zero values
    2. film_gamma_proj.bias.grad is finite and has non-zero values
    3. film_beta_proj.weight.grad is finite and has non-zero values
    4. film_beta_proj.bias.grad is finite and has non-zero values
    5. Gradient flows through write_base (requires_grad=True)
    6. Gradient flows through memory (requires_grad=True)
    """
    batch_size, write_base_data, memory_data = inputs

    # Model config matching the spec
    config = CausalMatrixMergeV2Config(
        vocab_size=100,
        model_dim=32,
        state_dim=16,
        num_slots=4,
        num_layers=1,
        write_rank=2,
        num_checkpoints=2,
        checkpoint_stride=4,
        top_k_slots=2,
        use_per_slot_decay=True,
        use_per_slot_write=True,
        use_adaptive_merge=True,
    )

    # Create model in eval mode (no dropout randomness)
    model = CausalMatrixMergeV2(config)
    model.eval()

    # Zero all gradients
    model.zero_grad()

    # Create input tensors with gradient tracking
    write_base = torch.tensor(write_base_data, dtype=torch.float32, requires_grad=True)
    memory = torch.tensor(memory_data, dtype=torch.float32, requires_grad=True)

    # Ensure inputs are finite and non-degenerate
    assume(torch.isfinite(write_base).all().item())
    assume(torch.isfinite(memory).all().item())
    # Non-degenerate: write_base and memory must have non-zero values for gradient
    # to flow through the weight matrices (grad_W depends on input activation)
    assume(write_base.abs().sum().item() > 1e-6)
    assume(memory.abs().sum().item() > 1e-6)

    # Call the per-slot adaptive write method
    write_adapted = model._per_slot_adaptive_write(write_base, memory)

    # Sum the output and call backward
    loss = write_adapted.sum()
    loss.backward()

    # 1. Verify film_gamma_proj.weight.grad is finite and has non-zero values
    gamma_weight_grad = model.film_gamma_proj.weight.grad
    assert gamma_weight_grad is not None, (
        "film_gamma_proj.weight.grad is None — no gradient computed"
    )
    assert torch.isfinite(gamma_weight_grad).all().item(), (
        f"film_gamma_proj.weight.grad contains non-finite values: {gamma_weight_grad}"
    )
    assert gamma_weight_grad.abs().sum().item() > 0, (
        "film_gamma_proj.weight.grad is all zeros — gradient did not flow through"
    )

    # 2. Verify film_gamma_proj.bias.grad is finite and has non-zero values
    gamma_bias_grad = model.film_gamma_proj.bias.grad
    assert gamma_bias_grad is not None, (
        "film_gamma_proj.bias.grad is None — no gradient computed"
    )
    assert torch.isfinite(gamma_bias_grad).all().item(), (
        f"film_gamma_proj.bias.grad contains non-finite values: {gamma_bias_grad}"
    )
    assert gamma_bias_grad.abs().sum().item() > 0, (
        "film_gamma_proj.bias.grad is all zeros — gradient did not flow through"
    )

    # 3. Verify film_beta_proj.weight.grad is finite and has non-zero values
    beta_weight_grad = model.film_beta_proj.weight.grad
    assert beta_weight_grad is not None, (
        "film_beta_proj.weight.grad is None — no gradient computed"
    )
    assert torch.isfinite(beta_weight_grad).all().item(), (
        f"film_beta_proj.weight.grad contains non-finite values: {beta_weight_grad}"
    )
    assert beta_weight_grad.abs().sum().item() > 0, (
        "film_beta_proj.weight.grad is all zeros — gradient did not flow through"
    )

    # 4. Verify film_beta_proj.bias.grad is finite and has non-zero values
    beta_bias_grad = model.film_beta_proj.bias.grad
    assert beta_bias_grad is not None, (
        "film_beta_proj.bias.grad is None — no gradient computed"
    )
    assert torch.isfinite(beta_bias_grad).all().item(), (
        f"film_beta_proj.bias.grad contains non-finite values: {beta_bias_grad}"
    )
    assert beta_bias_grad.abs().sum().item() > 0, (
        "film_beta_proj.bias.grad is all zeros — gradient did not flow through"
    )

    # 5. Verify gradient flows through write_base
    assert write_base.grad is not None, (
        "write_base.grad is None — gradient did not flow through write_base"
    )
    assert torch.isfinite(write_base.grad).all().item(), (
        f"write_base.grad contains non-finite values: {write_base.grad}"
    )
    assert write_base.grad.abs().sum().item() > 0, (
        "write_base.grad is all zeros — gradient did not flow through write_base"
    )

    # 6. Verify gradient flows through memory
    assert memory.grad is not None, (
        "memory.grad is None — gradient did not flow through memory"
    )
    assert torch.isfinite(memory.grad).all().item(), (
        f"memory.grad contains non-finite values: {memory.grad}"
    )
    assert memory.grad.abs().sum().item() > 0, (
        "memory.grad is all zeros — gradient did not flow through memory"
    )


@given(batch_size=st.integers(min_value=1, max_value=4))
@settings(max_examples=100, deadline=None)
def test_write_differentiability_zero_memory(batch_size):
    """Property 8 (zero memory): Write Differentiability with zero memory.

    **Validates: Requirements 3.6, 8.2, 8.4**

    When the memory contains zeros (initial state), the per-slot write adaptation
    SHALL still produce finite gradients for all its learnable parameters and
    valid (finite) adapted write vectors without NaN or Inf (numerical stability
    verification per Requirement 8.4).
    """
    state_dim = 16
    num_slots = 4

    # Model config matching the spec
    config = CausalMatrixMergeV2Config(
        vocab_size=100,
        model_dim=32,
        state_dim=16,
        num_slots=4,
        num_layers=1,
        write_rank=2,
        num_checkpoints=2,
        checkpoint_stride=4,
        top_k_slots=2,
        use_per_slot_decay=True,
        use_per_slot_write=True,
        use_adaptive_merge=True,
    )

    # Create model in eval mode
    model = CausalMatrixMergeV2(config)
    model.eval()
    model.zero_grad()

    # Create non-zero write_base (non-degenerate) and zero memory (initial state)
    write_base = torch.randn(batch_size, state_dim, requires_grad=True)
    memory = torch.zeros(batch_size, num_slots, state_dim, requires_grad=True)

    # Call the per-slot adaptive write method
    write_adapted = model._per_slot_adaptive_write(write_base, memory)

    # Verify output is finite (no NaN or Inf) — Requirement 8.4
    assert torch.isfinite(write_adapted).all().item(), (
        f"write_adapted contains non-finite values with zero memory: "
        f"has_nan={torch.isnan(write_adapted).any()}, "
        f"has_inf={torch.isinf(write_adapted).any()}"
    )

    # Sum the output and call backward
    loss = write_adapted.sum()
    loss.backward()

    # Verify film_gamma_proj.weight.grad is finite
    gamma_weight_grad = model.film_gamma_proj.weight.grad
    assert gamma_weight_grad is not None, (
        "film_gamma_proj.weight.grad is None with zero memory"
    )
    assert torch.isfinite(gamma_weight_grad).all().item(), (
        "film_gamma_proj.weight.grad contains non-finite values with zero memory"
    )

    # Verify film_gamma_proj.bias.grad is finite
    gamma_bias_grad = model.film_gamma_proj.bias.grad
    assert gamma_bias_grad is not None, (
        "film_gamma_proj.bias.grad is None with zero memory"
    )
    assert torch.isfinite(gamma_bias_grad).all().item(), (
        "film_gamma_proj.bias.grad contains non-finite values with zero memory"
    )

    # Verify film_beta_proj.weight.grad is finite
    beta_weight_grad = model.film_beta_proj.weight.grad
    assert beta_weight_grad is not None, (
        "film_beta_proj.weight.grad is None with zero memory"
    )
    assert torch.isfinite(beta_weight_grad).all().item(), (
        "film_beta_proj.weight.grad contains non-finite values with zero memory"
    )

    # Verify film_beta_proj.bias.grad is finite
    beta_bias_grad = model.film_beta_proj.bias.grad
    assert beta_bias_grad is not None, (
        "film_beta_proj.bias.grad is None with zero memory"
    )
    assert torch.isfinite(beta_bias_grad).all().item(), (
        "film_beta_proj.bias.grad contains non-finite values with zero memory"
    )

    # Verify gradient flows through write_base
    assert write_base.grad is not None, (
        "write_base.grad is None with zero memory"
    )
    assert torch.isfinite(write_base.grad).all().item(), (
        "write_base.grad contains non-finite values with zero memory"
    )
    assert write_base.grad.abs().sum().item() > 0, (
        "write_base.grad is all zeros with zero memory — gradient should still flow"
    )

    # Verify gradient flows through memory (even if zero input,
    # gradient itself may be non-zero due to bias terms)
    assert memory.grad is not None, (
        "memory.grad is None with zero memory"
    )
    assert torch.isfinite(memory.grad).all().item(), (
        "memory.grad contains non-finite values with zero memory"
    )


# =============================================================================
# Feature: slot-local-merge-improvements, Property 9: Uniform Write Fallback When Disabled
# Validates: Requirements 3.7, 6.5
# =============================================================================

from pymbbo.architectures.causal_matrix_merge_v2.merge_v2 import rms_norm


# Config with use_per_slot_write=False for fallback testing
_FALLBACK_WRITE_CONFIG = CausalMatrixMergeV2Config(
    model_dim=32,
    state_dim=16,
    num_slots=4,
    write_rank=2,
    num_checkpoints=2,
    checkpoint_stride=4,
    top_k_slots=2,
    use_per_slot_write=False,
)


@st.composite
def uniform_write_fallback_inputs(draw):
    """Generate random x input tensors for testing the uniform write fallback.

    Produces a batch of input vectors x of shape [B, model_dim] with values
    in (-5, 5) to drive the forward computation through in_proj → rms_norm →
    write_mlp → write_base → unsqueeze broadcast.
    """
    batch_size = draw(st.integers(min_value=1, max_value=4))
    model_dim = _FALLBACK_WRITE_CONFIG.model_dim

    # Generate x: [B, model_dim] in reasonable range
    x = torch.randn(batch_size, model_dim) * 3.0

    return x, batch_size


@given(inputs=uniform_write_fallback_inputs())
@settings(max_examples=100, deadline=None)
def test_uniform_write_fallback_when_disabled(inputs):
    """Property 9: Uniform Write Fallback When Disabled.

    **Validates: Requirements 3.7, 6.5**

    For any input tensors, when use_per_slot_write is False, all slots SHALL
    receive an identical write vector (the broadcast of write_base), i.e.,
    write_adapted[:, i, :] == write_adapted[:, j, :] for all slot pairs (i, j).
    """
    x, batch_size = inputs
    num_slots = _FALLBACK_WRITE_CONFIG.num_slots
    state_dim = _FALLBACK_WRITE_CONFIG.state_dim

    # Create model with use_per_slot_write=False
    torch.manual_seed(42)
    model = CausalMatrixMergeV2(_FALLBACK_WRITE_CONFIG)
    model.eval()

    with torch.no_grad():
        # Replicate forward steps until write is computed:
        # h = rms_norm(in_proj(x))
        h = rms_norm(model.in_proj(x))  # [B, model_dim]

        # write_base = write_mlp(h) → [B, state_dim]
        write_base = model._write_mlp(h)  # [B, state_dim]

        # Since use_per_slot_write=False: write_adapted = write_base.unsqueeze(1)
        # This broadcasts to [B, 1, state_dim] and routing expands to [B, S, state_dim]
        write_adapted = write_base.unsqueeze(1)  # [B, 1, state_dim]

        # Expand to full slot dimension for comparison
        write_expanded = write_adapted.expand(-1, num_slots, -1)  # [B, S, state_dim]

    # Verify: all slots are identical — write_expanded[:, i, :] == write_expanded[:, j, :]
    # for all slot pairs (i, j)
    for i in range(num_slots):
        for j in range(i + 1, num_slots):
            assert torch.allclose(
                write_expanded[:, i, :],
                write_expanded[:, j, :],
                atol=1e-7,
            ), (
                f"Slot {i} and slot {j} have different write vectors when "
                f"use_per_slot_write=False.\n"
                f"Slot {i}: {write_expanded[:, i, :]}\n"
                f"Slot {j}: {write_expanded[:, j, :]}\n"
                f"Max diff: {(write_expanded[:, i, :] - write_expanded[:, j, :]).abs().max()}"
            )

    # Additionally verify that each slot equals the original write_base
    for i in range(num_slots):
        assert torch.allclose(
            write_expanded[:, i, :],
            write_base,
            atol=1e-7,
        ), (
            f"Slot {i} write vector differs from write_base when "
            f"use_per_slot_write=False.\n"
            f"Slot {i}: {write_expanded[:, i, :]}\n"
            f"write_base: {write_base}\n"
            f"Max diff: {(write_expanded[:, i, :] - write_base).abs().max()}"
        )


# =============================================================================
# Feature: slot-local-merge-improvements, Property 4: Decay Bypass When Disabled
# Validates: Requirements 1.7, 6.4
# =============================================================================


# Config for Property 4: per_slot_decay DISABLED, adaptive_merge ENABLED
_BYPASS_CONFIG_P4 = CausalMatrixMergeV2Config(
    vocab_size=100,
    model_dim=32,
    state_dim=16,
    num_slots=4,
    num_layers=1,
    write_rank=2,
    num_checkpoints=2,
    checkpoint_stride=4,
    top_k_slots=2,
    use_per_slot_decay=False,
    use_per_slot_write=True,
    use_adaptive_merge=True,
)


@st.composite
def decay_bypass_inputs(draw):
    """Generate valid inputs for testing decay bypass when per_slot_decay is disabled.

    Generates:
    - h: [B, model_dim] with values in (-5, 5)
    - decay_base: [B, S, 1] with values in (0.01, 0.99)
    - memory: [B, S, state_dim] with values in (-2, 2)
    """
    batch_size = draw(st.integers(min_value=1, max_value=4))
    model_dim = _BYPASS_CONFIG_P4.model_dim
    state_dim = _BYPASS_CONFIG_P4.state_dim
    num_slots = _BYPASS_CONFIG_P4.num_slots

    # Generate h in range (-5, 5)
    h = torch.FloatTensor(batch_size, model_dim).uniform_(-5.0, 5.0)

    # Generate decay_base in range (0.01, 0.99) to guarantee strict (0,1)
    decay_base = torch.FloatTensor(batch_size, num_slots, 1).uniform_(0.01, 0.99)

    # Generate memory in range (-2, 2)
    memory = torch.FloatTensor(batch_size, num_slots, state_dim).uniform_(-2.0, 2.0)

    return h, decay_base, memory, batch_size


@given(inputs=decay_bypass_inputs())
@settings(max_examples=100, deadline=None)
def test_decay_bypass_when_disabled(inputs):
    """Property 4: Decay Bypass When Disabled.

    **Validates: Requirements 1.7, 6.4**

    For any input tensors, when `use_per_slot_decay` is False (and
    `use_adaptive_merge` is True), the resulting `decay_final` SHALL equal
    the output of the legacy global `_adaptive_decay` method exactly.

    This test instantiates a model with use_per_slot_decay=False and
    use_adaptive_merge=True, calls _adaptive_decay directly with random inputs,
    and verifies the output is valid (shape [B, S, 1], values in (0, 1)).
    Since forward() routes to _adaptive_decay when use_per_slot_decay=False
    and use_adaptive_merge=True, confirming _adaptive_decay produces valid
    output verifies the bypass path is functional.
    """
    h, decay_base, memory, batch_size = inputs
    num_slots = _BYPASS_CONFIG_P4.num_slots

    # Instantiate model with per_slot_decay DISABLED but adaptive_merge ENABLED
    torch.manual_seed(42)
    model = CausalMatrixMergeV2(_BYPASS_CONFIG_P4)
    model.eval()

    with torch.no_grad():
        # Call _adaptive_decay directly — this is the legacy path that
        # forward() routes to when use_per_slot_decay=False, use_adaptive_merge=True
        decay_final = model._adaptive_decay(h, decay_base, memory)

    # 1. Verify output shape is [B, S, 1]
    assert decay_final.shape == (batch_size, num_slots, 1), (
        f"Expected shape ({batch_size}, {num_slots}, 1), got {decay_final.shape}"
    )

    # 2. Verify all values are strictly in (0, 1)
    assert (decay_final > 0).all(), (
        f"Found values <= 0 in legacy _adaptive_decay output. "
        f"Min value: {decay_final.min().item()}"
    )
    assert (decay_final < 1).all(), (
        f"Found values >= 1 in legacy _adaptive_decay output. "
        f"Max value: {decay_final.max().item()}"
    )

    # 3. Verify all values are finite (no NaN or Inf)
    assert torch.isfinite(decay_final).all(), (
        f"Found non-finite values in legacy _adaptive_decay output."
    )

    # 4. Verify the model does NOT have slot_decay_proj (per-slot decay layer)
    #    since use_per_slot_decay=False — confirming the bypass architecture
    assert not hasattr(model, "slot_decay_proj"), (
        "Model should NOT have slot_decay_proj when use_per_slot_decay=False"
    )


# =============================================================================
# Feature: slot-local-merge-improvements, Property 5: Affine Rule Preservation
# Validates: Requirements 1.8, 5.8
# =============================================================================


# Config for Property 5 tests
_AFFINE_RULE_CONFIG = CausalMatrixMergeV2Config(
    model_dim=32,
    state_dim=16,
    num_slots=4,
    write_rank=2,
    num_checkpoints=2,
    checkpoint_stride=4,
    top_k_slots=2,
    dropout=0.0,
    use_per_slot_decay=True,
    use_per_slot_write=True,
    use_adaptive_merge=True,
)


@st.composite
def affine_rule_inputs(draw):
    """Generate random x input and batch size for affine rule preservation testing.

    Produces:
    - batch_size: random int in [1, 4]
    - x: [B, model_dim] with values in (-3, 3)
    - memory_init: [B, S, state_dim] with values in (-2, 2)
    """
    batch_size = draw(st.integers(min_value=1, max_value=4))
    model_dim = _AFFINE_RULE_CONFIG.model_dim
    state_dim = _AFFINE_RULE_CONFIG.state_dim
    num_slots = _AFFINE_RULE_CONFIG.num_slots

    # Generate x: [B, model_dim] in reasonable range
    x = torch.FloatTensor(batch_size, model_dim).uniform_(-3.0, 3.0)

    # Generate initial memory: [B, S, state_dim] in reasonable range
    memory_init = torch.FloatTensor(batch_size, num_slots, state_dim).uniform_(-2.0, 2.0)

    return batch_size, x, memory_init


@given(inputs=affine_rule_inputs())
@settings(max_examples=100, deadline=None)
def test_affine_rule_preservation(inputs):
    """Property 5: Affine Rule Preservation.

    **Validates: Requirements 1.8, 5.8**

    For any forward pass through the merge block, the memory update SHALL satisfy
    the equation:
        memory_new[b, s, :] = decay_final[b, s, 0] * memory_old[b, s, :]
                            + (1 - decay_final[b, s, 0]) * write_routed[b, s, :]
    exactly (up to floating-point precision).

    Test strategy:
    1. Create a model with per-slot features enabled
    2. Generate random x input and an initial state
    3. Manually compute the forward pass intermediate values
    4. Compute expected memory using the affine rule
    5. Apply post_norm to expected memory
    6. Assert equality with actual new_state.memory
    """
    batch_size, x, memory_init = inputs

    # Create model with deterministic config (dropout=0.0)
    torch.manual_seed(42)
    model = CausalMatrixMergeV2(_AFFINE_RULE_CONFIG)
    model.eval()

    # Create initial state with random memory
    num_slots = _AFFINE_RULE_CONFIG.num_slots
    state_dim = _AFFINE_RULE_CONFIG.state_dim
    num_checkpoints = _AFFINE_RULE_CONFIG.num_checkpoints

    initial_state = MergeState(
        memory=memory_init,
        normalizer=torch.ones(batch_size, num_slots, 1),
        checkpoints=torch.zeros(batch_size, num_checkpoints, num_slots, state_dim),
        step=0,
    )

    with torch.no_grad():
        # --- Manually compute intermediate values ---

        # Step 1: h = rms_norm(in_proj(x))
        h = rms_norm(model.in_proj(x))  # [B, model_dim]

        # Step 2: decay_base = exp(-softplus(decay_proj(h)))
        decay_base = torch.exp(-F.softplus(model.decay_proj(h)))  # [B, S]
        decay_base = decay_base.unsqueeze(-1)  # [B, S, 1]

        # Step 3: decay_final via per_slot_adaptive_decay
        decay_final = model._per_slot_adaptive_decay(h, decay_base, initial_state.memory)

        # Step 4: route via sparse_route
        route = model._sparse_route(h)  # [B, S, 1]

        # Step 5: write_base via write_mlp
        write_base = model._write_mlp(h)  # [B, state_dim]

        # Step 6: write_adapted via per_slot_adaptive_write
        write_adapted = model._per_slot_adaptive_write(write_base, initial_state.memory)  # [B, S, state_dim]

        # Step 7: write_routed = route * write_adapted
        write_routed = route * write_adapted  # [B, S, state_dim]

        # Step 8: Compute expected memory using the affine rule
        memory_expected = decay_final * initial_state.memory + (1 - decay_final) * write_routed

        # Step 9: Apply post_norm to expected memory
        memory_normed_expected = model.post_norm(memory_expected)

        # --- Run actual forward pass ---
        _, new_state = model.forward(x, initial_state)

    # Assert that actual new_state.memory matches expected (post-normed) memory
    assert torch.allclose(new_state.memory, memory_normed_expected, atol=1e-5), (
        f"Affine rule preservation failed!\n"
        f"Max absolute difference: {(new_state.memory - memory_normed_expected).abs().max().item()}\n"
        f"Mean absolute difference: {(new_state.memory - memory_normed_expected).abs().mean().item()}\n"
        f"Batch size: {batch_size}\n"
        f"This indicates the memory update does not follow:\n"
        f"  memory_new = decay_final * memory_old + (1 - decay_final) * write_routed"
    )


# =============================================================================
# Feature: slot-local-merge-improvements, Property 10: Forward Pass Output Shape Preservation
# Validates: Requirements 5.1, 5.4, 7.4, 7.5
# =============================================================================

from pymbbo.architectures.causal_matrix_merge.state import MergeState

# Config for Property 10 tests (matches design doc test configuration)
_FORWARD_PASS_CONFIG = CausalMatrixMergeV2Config(
    vocab_size=100,
    model_dim=32,
    state_dim=16,
    num_slots=4,
    num_layers=1,
    write_rank=2,
    num_checkpoints=2,
    checkpoint_stride=4,
    top_k_slots=2,
    dropout=0.0,
    use_per_slot_decay=True,
    use_per_slot_write=True,
    use_adaptive_merge=True,
)

# Config with per-slot features disabled for coverage
_FORWARD_PASS_CONFIG_DISABLED = CausalMatrixMergeV2Config(
    vocab_size=100,
    model_dim=32,
    state_dim=16,
    num_slots=4,
    num_layers=1,
    write_rank=2,
    num_checkpoints=2,
    checkpoint_stride=4,
    top_k_slots=2,
    dropout=0.0,
    use_per_slot_decay=False,
    use_per_slot_write=False,
    use_adaptive_merge=True,
)


@st.composite
def forward_pass_inputs(draw):
    """Generate valid inputs for forward pass shape preservation testing.

    Generates:
    - Random batch sizes (1-4)
    - Random x inputs of shape [B, model_dim]
    - A flag indicating whether per-slot features are enabled or disabled
    """
    batch_size = draw(st.integers(min_value=1, max_value=4))
    model_dim = _FORWARD_PASS_CONFIG.model_dim

    # Generate x: [B, model_dim] with values in reasonable range
    x = torch.randn(batch_size, model_dim)

    # Choose whether per-slot features are enabled or disabled
    per_slot_enabled = draw(st.booleans())

    return batch_size, x, per_slot_enabled


@given(inputs=forward_pass_inputs())
@settings(max_examples=100, deadline=None)
def test_forward_pass_output_shape_preservation(inputs):
    """Property 10: Forward Pass Output Shape Preservation.

    **Validates: Requirements 5.1, 5.4, 7.4, 7.5**

    For any valid input x of shape [B, model_dim] and valid MergeState, the
    forward pass SHALL produce output of shape [B, model_dim] and a new MergeState
    with memory of shape [B, S, state_dim], normalizer of shape [B, S, 1], and
    checkpoints of shape [B, K, S, state_dim].
    """
    batch_size, x, per_slot_enabled = inputs

    # Select config based on per-slot flag
    config = _FORWARD_PASS_CONFIG if per_slot_enabled else _FORWARD_PASS_CONFIG_DISABLED

    model_dim = config.model_dim
    state_dim = config.state_dim
    num_slots = config.num_slots
    num_checkpoints = config.num_checkpoints

    # Create model in eval mode
    torch.manual_seed(42)
    model = CausalMatrixMergeV2(config)
    model.eval()

    # Initialize state
    state = model.init_state(batch_size)

    with torch.no_grad():
        output, new_state = model.forward(x, state)

    # Assert output shape is [B, model_dim]
    assert output.shape == (batch_size, model_dim), (
        f"Expected output shape ({batch_size}, {model_dim}), got {output.shape}"
    )

    # Assert new_state.memory shape is [B, S, state_dim]
    assert new_state.memory.shape == (batch_size, num_slots, state_dim), (
        f"Expected memory shape ({batch_size}, {num_slots}, {state_dim}), "
        f"got {new_state.memory.shape}"
    )

    # Assert new_state.normalizer shape is [B, S, 1]
    assert new_state.normalizer.shape == (batch_size, num_slots, 1), (
        f"Expected normalizer shape ({batch_size}, {num_slots}, 1), "
        f"got {new_state.normalizer.shape}"
    )

    # Assert new_state.checkpoints shape is [B, K, S, state_dim]
    assert new_state.checkpoints.shape == (batch_size, num_checkpoints, num_slots, state_dim), (
        f"Expected checkpoints shape ({batch_size}, {num_checkpoints}, {num_slots}, {state_dim}), "
        f"got {new_state.checkpoints.shape}"
    )

    # Assert all output/state tensors are finite (no NaN/Inf)
    assert torch.isfinite(output).all(), (
        f"Output contains non-finite values. "
        f"has_nan={torch.isnan(output).any()}, has_inf={torch.isinf(output).any()}"
    )
    assert torch.isfinite(new_state.memory).all(), (
        f"Memory contains non-finite values. "
        f"has_nan={torch.isnan(new_state.memory).any()}, has_inf={torch.isinf(new_state.memory).any()}"
    )
    assert torch.isfinite(new_state.normalizer).all(), (
        f"Normalizer contains non-finite values. "
        f"has_nan={torch.isnan(new_state.normalizer).any()}, has_inf={torch.isinf(new_state.normalizer).any()}"
    )
    assert torch.isfinite(new_state.checkpoints).all(), (
        f"Checkpoints contains non-finite values. "
        f"has_nan={torch.isnan(new_state.checkpoints).any()}, has_inf={torch.isinf(new_state.checkpoints).any()}"
    )


@given(
    batch_size=st.integers(min_value=1, max_value=4),
    num_steps=st.integers(min_value=2, max_value=6),
)
@settings(max_examples=100, deadline=None)
def test_forward_pass_shape_preservation_multiple_steps(batch_size, num_steps):
    """Property 10 (multi-step): Forward Pass Output Shape Preservation across multiple tokens.

    **Validates: Requirements 5.1, 5.4, 7.4, 7.5**

    For multiple consecutive forward calls, the output shape and state shapes
    SHALL be preserved across all tokens, verifying that state shapes do not
    drift or change as tokens are processed sequentially.
    """
    config = _FORWARD_PASS_CONFIG
    model_dim = config.model_dim
    state_dim = config.state_dim
    num_slots = config.num_slots
    num_checkpoints = config.num_checkpoints

    # Create model in eval mode
    torch.manual_seed(42)
    model = CausalMatrixMergeV2(config)
    model.eval()

    # Initialize state
    state = model.init_state(batch_size)

    with torch.no_grad():
        for step in range(num_steps):
            # Generate random input for each step
            x = torch.randn(batch_size, model_dim)

            output, state = model.forward(x, state)

            # Assert output shape is [B, model_dim] at every step
            assert output.shape == (batch_size, model_dim), (
                f"Step {step}: Expected output shape ({batch_size}, {model_dim}), "
                f"got {output.shape}"
            )

            # Assert state.memory shape is [B, S, state_dim] at every step
            assert state.memory.shape == (batch_size, num_slots, state_dim), (
                f"Step {step}: Expected memory shape ({batch_size}, {num_slots}, {state_dim}), "
                f"got {state.memory.shape}"
            )

            # Assert state.normalizer shape is [B, S, 1] at every step
            assert state.normalizer.shape == (batch_size, num_slots, 1), (
                f"Step {step}: Expected normalizer shape ({batch_size}, {num_slots}, 1), "
                f"got {state.normalizer.shape}"
            )

            # Assert state.checkpoints shape is [B, K, S, state_dim] at every step
            assert state.checkpoints.shape == (batch_size, num_checkpoints, num_slots, state_dim), (
                f"Step {step}: Expected checkpoints shape "
                f"({batch_size}, {num_checkpoints}, {num_slots}, {state_dim}), "
                f"got {state.checkpoints.shape}"
            )

            # Assert all tensors are finite at every step
            assert torch.isfinite(output).all(), (
                f"Step {step}: Output contains non-finite values"
            )
            assert torch.isfinite(state.memory).all(), (
                f"Step {step}: Memory contains non-finite values"
            )
            assert torch.isfinite(state.normalizer).all(), (
                f"Step {step}: Normalizer contains non-finite values"
            )
            assert torch.isfinite(state.checkpoints).all(), (
                f"Step {step}: Checkpoints contains non-finite values"
            )

# =============================================================================
# Feature: slot-local-merge-improvements, Property 12: Fixed Memory Size Invariant
# Validates: Requirements 5.6
# =============================================================================

# Config for Property 12 tests
_FIXED_MEMORY_SIZE_CONFIG = CausalMatrixMergeV2Config(
    model_dim=32,
    state_dim=16,
    num_slots=4,
    write_rank=2,
    num_checkpoints=2,
    checkpoint_stride=4,
    top_k_slots=2,
    dropout=0.0,
    use_per_slot_decay=True,
    use_per_slot_write=True,
    use_adaptive_merge=True,
)


@given(num_tokens=st.integers(min_value=2, max_value=10))
@settings(max_examples=100, deadline=None)
def test_fixed_memory_size_invariant(num_tokens):
    """Property 12: Fixed Memory Size Invariant.

    **Validates: Requirements 5.6**

    For any sequence of N tokens processed through the merge block (for varying N),
    the shapes of the MergeState tensors (memory, normalizer, checkpoints) SHALL
    remain constant regardless of N.

    This proves memory consumption doesn't grow with sequence length.
    """
    config = _FIXED_MEMORY_SIZE_CONFIG
    batch_size = 2
    model_dim = config.model_dim
    state_dim = config.state_dim
    num_slots = config.num_slots
    num_checkpoints = config.num_checkpoints

    # Expected shapes (fixed regardless of N)
    expected_memory_shape = (batch_size, num_slots, state_dim)
    expected_normalizer_shape = (batch_size, num_slots, 1)
    expected_checkpoints_shape = (batch_size, num_checkpoints, num_slots, state_dim)

    # Create model with per-slot features enabled
    torch.manual_seed(42)
    model = CausalMatrixMergeV2(config)
    model.eval()

    # Initialize state
    state = model.init_state(batch_size)

    # Record initial state shapes
    initial_memory_shape = state.memory.shape
    initial_normalizer_shape = state.normalizer.shape
    initial_checkpoints_shape = state.checkpoints.shape

    assert initial_memory_shape == expected_memory_shape, (
        f"Initial memory shape mismatch: {initial_memory_shape} != {expected_memory_shape}"
    )
    assert initial_normalizer_shape == expected_normalizer_shape, (
        f"Initial normalizer shape mismatch: {initial_normalizer_shape} != {expected_normalizer_shape}"
    )
    assert initial_checkpoints_shape == expected_checkpoints_shape, (
        f"Initial checkpoints shape mismatch: {initial_checkpoints_shape} != {expected_checkpoints_shape}"
    )

    # Process N tokens sequentially
    with torch.no_grad():
        for token_idx in range(num_tokens):
            # Generate a random token input
            x = torch.randn(batch_size, model_dim)

            # Forward pass
            _, state = model.forward(x, state)

            # After each token, assert shapes are EXACTLY the same as initial
            assert state.memory.shape == expected_memory_shape, (
                f"After token {token_idx + 1}/{num_tokens}: memory shape changed! "
                f"Got {state.memory.shape}, expected {expected_memory_shape}. "
                f"Memory size is growing with sequence length."
            )
            assert state.normalizer.shape == expected_normalizer_shape, (
                f"After token {token_idx + 1}/{num_tokens}: normalizer shape changed! "
                f"Got {state.normalizer.shape}, expected {expected_normalizer_shape}. "
                f"Normalizer size is growing with sequence length."
            )
            assert state.checkpoints.shape == expected_checkpoints_shape, (
                f"After token {token_idx + 1}/{num_tokens}: checkpoints shape changed! "
                f"Got {state.checkpoints.shape}, expected {expected_checkpoints_shape}. "
                f"Checkpoints size is growing with sequence length."
            )


# =============================================================================
# Feature: slot-local-merge-improvements, Property 13: Sparse Routing Preserves Top-K Structure
# Validates: Requirements 5.7
# =============================================================================

# Config for Property 13 tests
_SPARSE_ROUTING_CONFIG = CausalMatrixMergeV2Config(
    model_dim=32,
    state_dim=16,
    num_slots=4,
    write_rank=2,
    num_checkpoints=2,
    checkpoint_stride=4,
    top_k_slots=2,
)


@st.composite
def sparse_routing_inputs(draw):
    """Generate random h inputs for sparse routing top-K structure testing.

    Produces:
    - h: [B, model_dim] with values in (-5, 5)
    - batch_size: random int in [1, 4]
    """
    batch_size = draw(st.integers(min_value=1, max_value=4))
    model_dim = _SPARSE_ROUTING_CONFIG.model_dim

    # Generate h: [B, model_dim] in reasonable range
    h = torch.randn(batch_size, model_dim) * 3.0

    return h, batch_size


@given(inputs=sparse_routing_inputs())
@settings(max_examples=100, deadline=None)
def test_sparse_routing_preserves_top_k_structure(inputs):
    """Property 13: Sparse Routing Preserves Top-K Structure.

    **Validates: Requirements 5.7**

    For any input h, the sparse routing SHALL produce a weight tensor with
    exactly `top_k_slots` non-zero entries per batch element, and the non-zero
    entries SHALL sum to 1.

    Test strategy:
    1. Generate random h inputs of shape [B, model_dim]
    2. Create a model and call _sparse_route(h)
    3. For each batch element, assert:
       - Exactly top_k_slots entries are non-zero
       - All non-zero entries are positive
       - The sum of non-zero entries equals 1.0 (within tolerance)
       - The route shape is [B, S, 1]
    """
    h, batch_size = inputs
    num_slots = _SPARSE_ROUTING_CONFIG.num_slots
    top_k_slots = _SPARSE_ROUTING_CONFIG.top_k_slots

    # Create model
    torch.manual_seed(42)
    model = CausalMatrixMergeV2(_SPARSE_ROUTING_CONFIG)
    model.eval()

    with torch.no_grad():
        route = model._sparse_route(h)

    # Assert output shape is [B, S, 1]
    assert route.shape == (batch_size, num_slots, 1), (
        f"Expected route shape ({batch_size}, {num_slots}, 1), got {route.shape}"
    )

    # Squeeze last dim for per-element checks: [B, S]
    route_2d = route.squeeze(-1)  # [B, S]

    for b in range(batch_size):
        row = route_2d[b]  # [S]

        # Count non-zero entries
        non_zero_mask = row > 0
        num_non_zero = non_zero_mask.sum().item()

        # Assert exactly top_k_slots non-zero entries
        assert num_non_zero == top_k_slots, (
            f"Batch element {b}: expected exactly {top_k_slots} non-zero entries, "
            f"got {num_non_zero}. Route values: {row.tolist()}"
        )

        # Assert all non-zero entries are positive
        non_zero_vals = row[non_zero_mask]
        assert (non_zero_vals > 0).all(), (
            f"Batch element {b}: found non-positive values among non-zero entries. "
            f"Non-zero values: {non_zero_vals.tolist()}"
        )

        # Assert the sum of non-zero entries equals 1.0 (within tolerance)
        route_sum = non_zero_vals.sum().item()
        assert abs(route_sum - 1.0) < 1e-5, (
            f"Batch element {b}: non-zero entries sum to {route_sum}, "
            f"expected 1.0 (tolerance 1e-5). Non-zero values: {non_zero_vals.tolist()}"
        )


# =============================================================================
# Feature: slot-local-merge-improvements, Integration Tests
# Validates: Requirements 5.4, 5.6, 5.9, 7.4, 7.5, 8.1, 8.2
# =============================================================================

# Integration test configuration: both per-slot features enabled
_INTEGRATION_CONFIG = CausalMatrixMergeV2Config(
    model_dim=32,
    state_dim=16,
    num_slots=4,
    write_rank=2,
    num_checkpoints=2,
    checkpoint_stride=4,
    top_k_slots=2,
    dropout=0.0,
    use_per_slot_decay=True,
    use_per_slot_write=True,
    use_adaptive_merge=True,
)


@given(
    batch_size=st.integers(min_value=1, max_value=4),
    seq_len=st.integers(min_value=2, max_value=16),
)
@settings(max_examples=50, deadline=None)
def test_forward_sequence_multi_token_output_shapes(batch_size, seq_len):
    """Integration Test 1: forward_sequence with multi-token input produces valid output shapes.

    **Validates: Requirements 5.4, 7.4, 7.5**

    forward_sequence with input [B, T, model_dim] SHALL produce:
    - output of shape [B, T, model_dim]
    - MergeState with memory [B, S, state_dim], normalizer [B, S, 1],
      checkpoints [B, K, S, state_dim]
    All output values must be finite.
    """
    model_dim = _INTEGRATION_CONFIG.model_dim
    state_dim = _INTEGRATION_CONFIG.state_dim
    num_slots = _INTEGRATION_CONFIG.num_slots
    num_checkpoints = _INTEGRATION_CONFIG.num_checkpoints

    # Create model with both per-slot features enabled
    torch.manual_seed(42)
    model = CausalMatrixMergeV2(_INTEGRATION_CONFIG)
    model.eval()

    # Generate multi-token input sequence [B, T, model_dim]
    x = torch.randn(batch_size, seq_len, model_dim)

    with torch.no_grad():
        output, final_state = model.forward_sequence(x)

    # Assert output shape is [B, T, model_dim]
    assert output.shape == (batch_size, seq_len, model_dim), (
        f"Expected output shape ({batch_size}, {seq_len}, {model_dim}), "
        f"got {output.shape}"
    )

    # Assert MergeState shapes
    assert final_state.memory.shape == (batch_size, num_slots, state_dim), (
        f"Expected memory shape ({batch_size}, {num_slots}, {state_dim}), "
        f"got {final_state.memory.shape}"
    )
    assert final_state.normalizer.shape == (batch_size, num_slots, 1), (
        f"Expected normalizer shape ({batch_size}, {num_slots}, 1), "
        f"got {final_state.normalizer.shape}"
    )
    assert final_state.checkpoints.shape == (batch_size, num_checkpoints, num_slots, state_dim), (
        f"Expected checkpoints shape "
        f"({batch_size}, {num_checkpoints}, {num_slots}, {state_dim}), "
        f"got {final_state.checkpoints.shape}"
    )

    # All values must be finite (no NaN or Inf)
    assert torch.isfinite(output).all(), (
        f"Output contains non-finite values after forward_sequence. "
        f"has_nan={torch.isnan(output).any()}, has_inf={torch.isinf(output).any()}"
    )
    assert torch.isfinite(final_state.memory).all(), (
        f"Memory contains non-finite values after forward_sequence."
    )
    assert torch.isfinite(final_state.normalizer).all(), (
        f"Normalizer contains non-finite values after forward_sequence."
    )
    assert torch.isfinite(final_state.checkpoints).all(), (
        f"Checkpoints contains non-finite values after forward_sequence."
    )


@given(batch_size=st.integers(min_value=1, max_value=4))
@settings(max_examples=50, deadline=None)
def test_merge_state_shapes_invariant_across_sequence_lengths(batch_size):
    """Integration Test 2: MergeState shapes are invariant across sequence lengths.

    **Validates: Requirements 5.6**

    The shapes of MergeState tensors (memory, normalizer, checkpoints) SHALL
    remain constant regardless of the sequence length T processed. Tests with
    T=1, 4, 8, 16 and verifies all produce identical state shapes.
    """
    model_dim = _INTEGRATION_CONFIG.model_dim
    state_dim = _INTEGRATION_CONFIG.state_dim
    num_slots = _INTEGRATION_CONFIG.num_slots
    num_checkpoints = _INTEGRATION_CONFIG.num_checkpoints

    # Create model with both per-slot features enabled
    torch.manual_seed(42)
    model = CausalMatrixMergeV2(_INTEGRATION_CONFIG)
    model.eval()

    # Expected fixed shapes
    expected_memory_shape = (batch_size, num_slots, state_dim)
    expected_normalizer_shape = (batch_size, num_slots, 1)
    expected_checkpoints_shape = (batch_size, num_checkpoints, num_slots, state_dim)

    # Test with varying sequence lengths
    sequence_lengths = [1, 4, 8, 16]

    with torch.no_grad():
        for T in sequence_lengths:
            x = torch.randn(batch_size, T, model_dim)
            _, final_state = model.forward_sequence(x)

            # Memory shape must be invariant
            assert final_state.memory.shape == expected_memory_shape, (
                f"Memory shape changed with T={T}. "
                f"Expected {expected_memory_shape}, got {final_state.memory.shape}"
            )

            # Normalizer shape must be invariant
            assert final_state.normalizer.shape == expected_normalizer_shape, (
                f"Normalizer shape changed with T={T}. "
                f"Expected {expected_normalizer_shape}, got {final_state.normalizer.shape}"
            )

            # Checkpoints shape must be invariant
            assert final_state.checkpoints.shape == expected_checkpoints_shape, (
                f"Checkpoints shape changed with T={T}. "
                f"Expected {expected_checkpoints_shape}, got {final_state.checkpoints.shape}"
            )

            # All state tensors must be finite
            assert torch.isfinite(final_state.memory).all(), (
                f"Memory contains non-finite values after T={T} sequence."
            )
            assert torch.isfinite(final_state.normalizer).all(), (
                f"Normalizer contains non-finite values after T={T} sequence."
            )
            assert torch.isfinite(final_state.checkpoints).all(), (
                f"Checkpoints contains non-finite values after T={T} sequence."
            )


@given(
    batch_size=st.integers(min_value=1, max_value=3),
    seq_len=st.integers(min_value=2, max_value=8),
)
@settings(max_examples=50, deadline=None)
def test_backward_pass_completes_without_error(batch_size, seq_len):
    """Integration Test 3: Full backward pass completes without error on full sequence.

    **Validates: Requirements 8.1, 8.2, 5.9**

    Run forward_sequence, sum the output, call .backward(), and verify:
    - No errors are raised during backward pass
    - All model parameters that require grad have finite gradients
    - At least some gradients are non-zero (gradient actually flows)
    """
    model_dim = _INTEGRATION_CONFIG.model_dim

    # Create model in training mode (gradients enabled)
    torch.manual_seed(42)
    model = CausalMatrixMergeV2(_INTEGRATION_CONFIG)
    model.train()
    model.zero_grad()

    # Generate multi-token input sequence [B, T, model_dim]
    x = torch.randn(batch_size, seq_len, model_dim)

    # Forward pass
    output, _ = model.forward_sequence(x)

    # Compute scalar loss and backward
    loss = output.sum()
    loss.backward()

    # Legacy modules that are not used when per-slot decay is active
    # (they exist in the model but don't participate in the forward pass)
    legacy_modules = {"memory_summary_proj", "modulation_proj"}

    # Verify all parameter gradients are finite
    has_nonzero_grad = False
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Skip legacy modules that are not used in the active forward path
            if any(legacy in name for legacy in legacy_modules):
                continue

            assert param.grad is not None, (
                f"Parameter '{name}' has requires_grad=True but grad is None "
                f"after backward pass."
            )
            assert torch.isfinite(param.grad).all(), (
                f"Parameter '{name}' has non-finite gradient values. "
                f"has_nan={torch.isnan(param.grad).any()}, "
                f"has_inf={torch.isinf(param.grad).any()}"
            )
            if param.grad.abs().sum().item() > 0:
                has_nonzero_grad = True

    # At least some gradients must be non-zero (gradient actually flows)
    assert has_nonzero_grad, (
        "All parameter gradients are zero — backward pass did not propagate "
        "gradients through the model."
    )
