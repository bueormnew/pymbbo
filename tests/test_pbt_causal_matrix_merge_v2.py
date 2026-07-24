"""Property-based tests for Causal Matrix Merge v2.

Uses hypothesis to verify universal invariants of the CausalMatrixMergeV2Config.
"""

import pytest
from hypothesis import given, settings, assume
import hypothesis.strategies as st

from pymbbo.architectures.causal_matrix_merge_v2.config import CausalMatrixMergeV2Config


# =============================================================================
# Feature: causal-matrix-merge-v2, Property 2: Validación de Configuración Rechaza Valores Inválidos
# Validates: Requirements 8.6
# =============================================================================


@st.composite
def invalid_config_params(draw):
    """Generate a config dictionary with at least one invalid field value.

    Strategy: generate valid base values, then inject one or more invalid values.
    """
    # First, generate valid base values
    num_slots = draw(st.integers(min_value=1, max_value=32))

    valid_params = {
        "vocab_size": draw(st.integers(min_value=1, max_value=10000)),
        "model_dim": draw(st.integers(min_value=1, max_value=512)),
        "state_dim": draw(st.integers(min_value=1, max_value=256)),
        "num_slots": num_slots,
        "num_layers": draw(st.integers(min_value=1, max_value=8)),
        "num_checkpoints": draw(st.integers(min_value=1, max_value=16)),
        "checkpoint_stride": draw(st.integers(min_value=1, max_value=64)),
        "write_rank": draw(st.integers(min_value=1, max_value=16)),
        "dropout": draw(st.floats(min_value=0.0, max_value=1.0)),
        "max_context": draw(st.integers(min_value=1, max_value=4096)),
        "ffn_mult": draw(st.floats(min_value=0.01, max_value=10.0)),
        "top_k_slots": draw(st.integers(min_value=1, max_value=num_slots)),
    }

    # Define strategies for invalid values per field
    invalid_strategies = {
        "vocab_size": st.integers(min_value=-100, max_value=0),
        "model_dim": st.integers(min_value=-100, max_value=0),
        "state_dim": st.integers(min_value=-100, max_value=0),
        "num_slots": st.integers(min_value=-100, max_value=0),
        "num_layers": st.integers(min_value=-100, max_value=0),
        "num_checkpoints": st.integers(min_value=-100, max_value=0),
        "checkpoint_stride": st.integers(min_value=-100, max_value=0),
        "write_rank": st.integers(min_value=-100, max_value=0),
        "max_context": st.integers(min_value=-100, max_value=0),
        "dropout": st.one_of(
            st.floats(min_value=-10.0, max_value=-0.001),
            st.floats(min_value=1.001, max_value=10.0),
        ),
        "ffn_mult": st.floats(min_value=-10.0, max_value=0.0),
        "top_k_slots": st.one_of(
            st.integers(min_value=-100, max_value=0),
            st.integers(min_value=num_slots + 1, max_value=num_slots + 100),
        ),
    }

    # Choose at least one field to invalidate
    fields_to_invalidate = draw(
        st.lists(
            st.sampled_from(list(invalid_strategies.keys())),
            min_size=1,
            max_size=4,
        )
    )

    # Apply invalid values to the chosen fields
    params = dict(valid_params)
    for field in fields_to_invalidate:
        params[field] = draw(invalid_strategies[field])

    return params


@given(params=invalid_config_params())
@settings(max_examples=200)
def test_config_rejects_invalid_values(params):
    """Property 2: Validación de Configuración Rechaza Valores Inválidos.

    **Validates: Requirements 8.6**

    For any combination of configuration values where at least one field is
    out of range, instantiating CausalMatrixMergeV2Config SHALL raise ValueError.
    """
    with pytest.raises(ValueError):
        CausalMatrixMergeV2Config(**params)


# =============================================================================
# Feature: causal-matrix-merge-v2, Property 8: SwiGLU MLP Computa la Fórmula Correcta
# Validates: Requirements 2.2
# =============================================================================

import torch
import torch.nn.functional as F

from pymbbo.architectures.causal_matrix_merge_v2.mlp import SwiGLUMLP


@given(
    model_dim=st.integers(min_value=4, max_value=64),
    ffn_mult=st.floats(min_value=0.5, max_value=4.0),
    batch_size=st.integers(min_value=1, max_value=8),
)
@settings(max_examples=100)
def test_swiglu_mlp_computes_correct_formula(
    model_dim: int, ffn_mult: float, batch_size: int
):
    """Property 8: SwiGLU MLP Computa la Fórmula Correcta.

    **Validates: Requirements 2.2**

    Para cualquier tensor de entrada x, la salida del SwiGLUMLP (en eval mode,
    sin dropout) es equivalente a (x @ W_gate.T) * silu(x @ W_up.T) @ W_down.T.
    """
    # Crear instancia y poner en eval mode (desactiva dropout)
    mlp = SwiGLUMLP(model_dim=model_dim, ffn_mult=ffn_mult, dropout=0.1)
    mlp.eval()

    # Generar input aleatorio
    x = torch.randn(batch_size, model_dim)

    # Computación del módulo
    with torch.no_grad():
        module_output = mlp(x)

    # Computación manual de la fórmula SwiGLU:
    # SwiGLU(x) = (x @ W_gate) * silu(x @ W_up) @ W_down
    with torch.no_grad():
        gate = x @ mlp.w_gate.weight.T  # [B, hidden_dim]
        up = F.silu(x @ mlp.w_up.weight.T)  # [B, hidden_dim]
        manual_output = (gate * up) @ mlp.w_down.weight.T  # [B, model_dim]

    assert torch.allclose(module_output, manual_output, atol=1e-5, rtol=1e-5), (
        f"SwiGLU output mismatch: max diff = {(module_output - manual_output).abs().max().item()}"
    )

# =============================================================================
# Feature: causal-matrix-merge-v2, Property 17: Dimensiones Internas Determinadas por Config
# Validates: Requirements 2.3, 4.2
# =============================================================================


@given(
    model_dim=st.integers(4, 128),
    ffn_mult=st.floats(0.5, 8.0),
)
@settings(max_examples=100)
def test_swiglu_internal_dimensions_determined_by_config(model_dim: int, ffn_mult: float):
    """Property 17: Dimensiones Internas Determinadas por Config.

    **Validates: Requirements 2.3, 4.2**

    Para cualquier CausalMatrixMergeV2Config válido, las dimensiones internas
    del SwiGLU MLP SHALL ser int(model_dim * ffn_mult).

    NOTE: WriteMLP dimension test (write_rank * state_dim) will be added once
    Task 3.2 implements WriteMLP.
    """
    expected_hidden_dim = int(model_dim * ffn_mult)

    # Skip degenerate cases where hidden_dim would be 0
    if expected_hidden_dim < 1:
        return

    mlp = SwiGLUMLP(model_dim=model_dim, ffn_mult=ffn_mult, dropout=0.0)

    # Verify w_gate dimensions: Linear(model_dim, hidden_dim) -> weight shape is (hidden_dim, model_dim)
    assert mlp.w_gate.weight.shape == (expected_hidden_dim, model_dim), (
        f"w_gate.weight.shape={mlp.w_gate.weight.shape}, "
        f"expected=({expected_hidden_dim}, {model_dim})"
    )

    # Verify w_up dimensions: Linear(model_dim, hidden_dim) -> weight shape is (hidden_dim, model_dim)
    assert mlp.w_up.weight.shape == (expected_hidden_dim, model_dim), (
        f"w_up.weight.shape={mlp.w_up.weight.shape}, "
        f"expected=({expected_hidden_dim}, {model_dim})"
    )

    # Verify w_down dimensions: Linear(hidden_dim, model_dim) -> weight shape is (model_dim, hidden_dim)
    assert mlp.w_down.weight.shape == (model_dim, expected_hidden_dim), (
        f"w_down.weight.shape={mlp.w_down.weight.shape}, "
        f"expected=({model_dim}, {expected_hidden_dim})"
    )


# =============================================================================
# Feature: causal-matrix-merge-v2, Property 3: Sparse Routing Selecciona Exactamente K Slots
# Validates: Requirements 3.1, 3.2
# =============================================================================

from pymbbo.architectures.causal_matrix_merge_v2.merge_v2 import CausalMatrixMergeV2


@given(
    num_slots=st.integers(min_value=4, max_value=16),
    model_dim=st.integers(min_value=16, max_value=64),
    batch_size=st.integers(min_value=1, max_value=4),
)
@settings(max_examples=100)
def test_sparse_routing_selects_exactly_k_slots(
    num_slots: int, model_dim: int, batch_size: int
):
    """Property 3: Sparse Routing Selecciona Exactamente K Slots.

    **Validates: Requirements 3.1, 3.2**

    Para cualquier tensor de entrada h de forma [B, model_dim], el routing
    disperso SHALL producir pesos donde exactamente top_k_slots posiciones
    tienen valor no-cero y las restantes num_slots - top_k_slots posiciones
    son exactamente cero.
    """
    # Generate top_k_slots in valid range (1 to num_slots-1) for sparse case
    top_k_slots = torch.randint(1, num_slots, (1,)).item()

    # Create config with these params
    config = CausalMatrixMergeV2Config(
        vocab_size=100,
        model_dim=model_dim,
        state_dim=32,
        num_slots=num_slots,
        num_layers=1,
        num_checkpoints=2,
        checkpoint_stride=8,
        write_rank=2,
        dropout=0.0,
        use_residual_gate=True,
        max_context=128,
        ffn_mult=2.0,
        top_k_slots=top_k_slots,
        use_adaptive_merge=False,
        use_learned_checkpoints=False,
    )

    # Instantiate the merge module
    merge = CausalMatrixMergeV2(config)
    merge.eval()

    # Create random h tensor [B, model_dim]
    h = torch.randn(batch_size, model_dim)

    # Call _sparse_route → result [B, S, 1]
    with torch.no_grad():
        sparse_route = merge._sparse_route(h)

    # Squeeze last dim: [B, S]
    route_weights = sparse_route.squeeze(-1)

    # For each batch element, verify:
    # 1. Exactly top_k_slots positions are non-zero
    # 2. Remaining positions are exactly zero
    for b in range(batch_size):
        weights_b = route_weights[b]  # [S]

        # Count non-zero entries
        nonzero_count = (weights_b != 0.0).sum().item()

        assert nonzero_count == top_k_slots, (
            f"Batch {b}: expected exactly {top_k_slots} non-zero positions, "
            f"got {nonzero_count}. Weights: {weights_b.tolist()}"
        )

        # Assert zero positions are exactly zero (not just close to zero)
        zero_mask = weights_b == 0.0
        zero_count = zero_mask.sum().item()
        expected_zero_count = num_slots - top_k_slots

        assert zero_count == expected_zero_count, (
            f"Batch {b}: expected {expected_zero_count} zero positions, "
            f"got {zero_count}. Weights: {weights_b.tolist()}"
        )

        # Double check: values at zero positions are exactly 0.0
        assert (weights_b[zero_mask] == 0.0).all(), (
            f"Batch {b}: some 'zero' positions are not exactly zero. "
            f"Zero-position values: {weights_b[zero_mask].tolist()}"
        )


# =============================================================================
# Feature: causal-matrix-merge-v2, Property 7: WriteMLP Produce Forma Correcta
# Validates: Requirements 4.3
# =============================================================================

from pymbbo.architectures.causal_matrix_merge_v2.merge_v2 import WriteMLP


@given(
    model_dim=st.integers(min_value=8, max_value=128),
    write_rank=st.integers(min_value=1, max_value=8),
    state_dim=st.integers(min_value=4, max_value=64),
    batch_size=st.integers(min_value=1, max_value=8),
)
@settings(max_examples=100)
def test_write_mlp_produces_correct_shape(
    model_dim: int, write_rank: int, state_dim: int, batch_size: int
):
    """Property 7: WriteMLP Produce Forma Correcta.

    **Validates: Requirements 4.3**

    Para cualquier tensor de entrada de forma [B, model_dim] con batch size
    y model_dim arbitrarios (consistentes con config), WriteMLP SHALL producir
    un tensor de forma [B, state_dim].
    """
    # Instanciar WriteMLP con dimensiones generadas
    write_mlp = WriteMLP(model_dim, write_rank, state_dim)
    write_mlp.eval()

    # Crear tensor de entrada aleatorio [B, model_dim]
    h = torch.randn(batch_size, model_dim)

    # Forward pass
    with torch.no_grad():
        output = write_mlp(h)

    # Verificar forma de salida
    assert output.shape == (batch_size, state_dim), (
        f"WriteMLP output shape mismatch: got {output.shape}, "
        f"expected ({batch_size}, {state_dim})"
    )


# =============================================================================
# Feature: causal-matrix-merge-v2, Property 4: Sparse Routing con K=num_slots Equivale a Softmax Denso
# Validates: Requirements 3.4
# =============================================================================

from pymbbo.architectures.causal_matrix_merge_v2.merge_v2 import CausalMatrixMergeV2


@given(
    num_slots=st.integers(min_value=2, max_value=16),
    model_dim=st.integers(min_value=16, max_value=64),
    batch_size=st.integers(min_value=1, max_value=4),
)
@settings(max_examples=100)
def test_sparse_routing_k_equals_s_equivalent_to_dense_softmax(
    num_slots: int, model_dim: int, batch_size: int
):
    """Property 4: Sparse Routing con K=num_slots Equivale a Softmax Denso.

    **Validates: Requirements 3.4**

    Para cualquier tensor de entrada, cuando top_k_slots == num_slots, los pesos
    del routing disperso SHALL ser idénticos (dentro de tolerancia numérica) a los
    pesos producidos por un softmax denso sobre los mismos scores.
    """
    # Create config with top_k_slots = num_slots (K=S)
    config = CausalMatrixMergeV2Config(
        model_dim=model_dim,
        state_dim=32,
        num_slots=num_slots,
        top_k_slots=num_slots,  # K=S: all slots selected
        write_rank=2,
        num_layers=1,
        num_checkpoints=2,
        checkpoint_stride=8,
        use_adaptive_merge=False,
        use_learned_checkpoints=False,
    )

    # Instantiate model
    merge = CausalMatrixMergeV2(config)
    merge.eval()

    # Create random h [B, model_dim]
    h = torch.randn(batch_size, model_dim)

    with torch.no_grad():
        # Get sparse route output: [B, num_slots, 1] → squeeze to [B, S]
        actual = merge._sparse_route(h).squeeze(-1)

        # Compute dense softmax directly from the same scores
        scores = merge.route_proj(h)  # [B, S]
        expected = F.softmax(scores, dim=-1)  # [B, S]

    assert torch.allclose(actual, expected, atol=1e-5), (
        f"Sparse routing with K=S should equal dense softmax. "
        f"Max diff = {(actual - expected).abs().max().item()}"
    )


# =============================================================================
# Feature: causal-matrix-merge-v2, Property 6: Composición Afín (compose_affine)
# Validates: Requirements 7.9
# =============================================================================


@given(
    batch_size=st.integers(min_value=1, max_value=4),
    num_slots=st.integers(min_value=2, max_value=8),
    state_dim=st.integers(min_value=4, max_value=32),
)
@settings(max_examples=100)
def test_compose_affine_equivalence(
    batch_size: int, num_slots: int, state_dim: int
):
    """Property 6: Composición Afín (compose_affine).

    **Validates: Requirements 7.9**

    Para cualquier par de transformaciones afines (decay₁, write₁) y (decay₂, write₂),
    aplicar compose_affine y luego aplicar el resultado a un estado inicial SHALL
    producir el mismo estado que aplicar T₁ seguido de T₂ secuencialmente.
    """
    # Generate random tensors for decay1, write1, decay2, write2 (same shape [B, S, D])
    # Use sigmoid to constrain decays to (0, 1)
    decay1 = torch.sigmoid(torch.randn(batch_size, num_slots, state_dim))
    write1 = torch.randn(batch_size, num_slots, state_dim)
    decay2 = torch.sigmoid(torch.randn(batch_size, num_slots, state_dim))
    write2 = torch.randn(batch_size, num_slots, state_dim)

    # Initial state S0 (random tensor)
    S0 = torch.randn(batch_size, num_slots, state_dim)

    # Sequential application:
    # S1 = decay1 * S0 + write1
    S1 = decay1 * S0 + write1
    # S2 = decay2 * S1 + write2
    S2 = decay2 * S1 + write2

    # Composed application:
    # decay_c, write_c = compose_affine(decay1, write1, decay2, write2)
    decay_c, write_c = CausalMatrixMergeV2.compose_affine(decay1, write1, decay2, write2)
    # S2_composed = decay_c * S0 + write_c
    S2_composed = decay_c * S0 + write_c

    # Assert equivalence within numerical tolerance
    assert torch.allclose(S2, S2_composed, atol=1e-5), (
        f"compose_affine result differs from sequential application. "
        f"Max diff = {(S2 - S2_composed).abs().max().item()}"
    )


# =============================================================================
# Feature: causal-matrix-merge-v2, Property 5: Invariante de Regla Afín
# Validates: Requirements 3.6, 6.3, 7.2
# =============================================================================

from pymbbo.architectures.causal_matrix_merge_v2.merge_v2 import CausalMatrixMergeV2, rms_norm
from pymbbo.architectures.causal_matrix_merge_v2.config import CausalMatrixMergeV2Config
from pymbbo.architectures.causal_matrix_merge.state import MergeState


@given(
    model_dim=st.integers(min_value=16, max_value=64),
    state_dim=st.integers(min_value=8, max_value=32),
    num_slots=st.integers(min_value=4, max_value=12),
    write_rank=st.integers(min_value=1, max_value=4),
    batch_size=st.integers(min_value=1, max_value=4),
)
@settings(max_examples=100)
def test_affine_rule_invariant_without_adaptive(
    model_dim: int, state_dim: int, num_slots: int, write_rank: int, batch_size: int
):
    """Property 5: Invariante de Regla Afín (sin adaptive merge).

    **Validates: Requirements 3.6, 6.3, 7.2**

    Para cualquier estado de memoria previo y cualquier token de entrada, la
    actualización de memoria SHALL seguir la forma:
        M_t = decay_final * M_{t-1} + (1 - decay_final) * write_routed

    Con use_adaptive_merge=False, decay_final = decay_base.
    Se verifica que la memoria post-norm del forward coincide con la
    computación manual de la regla afín seguida de post_norm.
    """
    top_k_slots = min(num_slots, max(1, num_slots // 2))

    config = CausalMatrixMergeV2Config(
        vocab_size=100,
        model_dim=model_dim,
        state_dim=state_dim,
        num_slots=num_slots,
        num_layers=1,
        num_checkpoints=2,
        checkpoint_stride=1000,  # Large stride to avoid checkpoint interference
        write_rank=write_rank,
        dropout=0.0,
        use_residual_gate=False,
        max_context=128,
        ffn_mult=2.0,
        top_k_slots=top_k_slots,
        use_adaptive_merge=False,
        use_learned_checkpoints=False,
    )

    merge = CausalMatrixMergeV2(config)
    merge.eval()

    # Create random initial state with non-zero memory
    old_memory = torch.randn(batch_size, num_slots, state_dim)
    old_normalizer = torch.ones(batch_size, num_slots, 1)
    old_checkpoints = torch.zeros(batch_size, 2, num_slots, state_dim)
    old_state = MergeState(
        memory=old_memory,
        normalizer=old_normalizer,
        checkpoints=old_checkpoints,
        step=0,
    )

    # Random input token
    x = torch.randn(batch_size, model_dim)

    with torch.no_grad():
        # Manual computation of the affine rule
        h = rms_norm(merge.in_proj(x))  # [B, model_dim]

        # decay_base: exp(-softplus(·)) → (0, 1), unsqueeze to [B, S, 1]
        decay_base = torch.exp(-F.softplus(merge.decay_proj(h))).unsqueeze(-1)
        # Without adaptive merge: decay_final = decay_base
        decay_final = decay_base

        # Sparse routing
        route = merge._sparse_route(h)  # [B, S, 1]

        # Write MLP
        write = merge._write_mlp(h)  # [B, state_dim]

        # Write routed: route * write broadcast
        write_routed = route * write.unsqueeze(1)  # [B, S, state_dim]

        # Affine rule: M_t = decay_final * M_{t-1} + (1 - decay_final) * write_routed
        expected_raw_memory = decay_final * old_memory + (1 - decay_final) * write_routed

        # Apply post_norm (LayerNorm)
        expected_normed = merge.post_norm(expected_raw_memory)

        # Run actual forward pass
        _, new_state = merge(x, old_state)

    # Compare: the memory in new_state should match our manual computation
    assert torch.allclose(new_state.memory, expected_normed, atol=1e-5), (
        f"Affine rule invariant violated (no adaptive merge). "
        f"Max diff = {(new_state.memory - expected_normed).abs().max().item()}"
    )


@given(
    model_dim=st.integers(min_value=16, max_value=64),
    state_dim=st.integers(min_value=8, max_value=32),
    num_slots=st.integers(min_value=4, max_value=12),
    write_rank=st.integers(min_value=1, max_value=4),
    batch_size=st.integers(min_value=1, max_value=4),
)
@settings(max_examples=100)
def test_affine_rule_invariant_with_adaptive_merge(
    model_dim: int, state_dim: int, num_slots: int, write_rank: int, batch_size: int
):
    """Property 5: Invariante de Regla Afín (con adaptive merge).

    **Validates: Requirements 3.6, 6.3, 7.2**

    Para cualquier estado de memoria previo y cualquier token de entrada, la
    actualización de memoria SHALL seguir la forma:
        M_t = decay_final * M_{t-1} + (1 - decay_final) * write_routed

    Con use_adaptive_merge=True, decay_final es el decay modulado por contenido
    de memoria. Se verifica que la regla afín se preserva con modulación activa.
    """
    top_k_slots = min(num_slots, max(1, num_slots // 2))

    config = CausalMatrixMergeV2Config(
        vocab_size=100,
        model_dim=model_dim,
        state_dim=state_dim,
        num_slots=num_slots,
        num_layers=1,
        num_checkpoints=2,
        checkpoint_stride=1000,  # Large stride to avoid checkpoint interference
        write_rank=write_rank,
        dropout=0.0,
        use_residual_gate=False,
        max_context=128,
        ffn_mult=2.0,
        top_k_slots=top_k_slots,
        use_adaptive_merge=True,
        use_learned_checkpoints=False,
    )

    merge = CausalMatrixMergeV2(config)
    merge.eval()

    # Create random initial state with non-zero memory
    old_memory = torch.randn(batch_size, num_slots, state_dim)
    old_normalizer = torch.ones(batch_size, num_slots, 1)
    old_checkpoints = torch.zeros(batch_size, 2, num_slots, state_dim)
    old_state = MergeState(
        memory=old_memory,
        normalizer=old_normalizer,
        checkpoints=old_checkpoints,
        step=0,
    )

    # Random input token
    x = torch.randn(batch_size, model_dim)

    with torch.no_grad():
        # Manual computation of the affine rule with adaptive merge
        h = rms_norm(merge.in_proj(x))  # [B, model_dim]

        # decay_base: exp(-softplus(·)) → (0, 1), unsqueeze to [B, S, 1]
        decay_base = torch.exp(-F.softplus(merge.decay_proj(h))).unsqueeze(-1)

        # Adaptive decay: modulate decay_base using memory content
        decay_final = merge._adaptive_decay(h, decay_base, old_memory)

        # Sparse routing
        route = merge._sparse_route(h)  # [B, S, 1]

        # Write MLP
        write = merge._write_mlp(h)  # [B, state_dim]

        # Write routed: route * write broadcast
        write_routed = route * write.unsqueeze(1)  # [B, S, state_dim]

        # Affine rule: M_t = decay_final * M_{t-1} + (1 - decay_final) * write_routed
        expected_raw_memory = decay_final * old_memory + (1 - decay_final) * write_routed

        # Apply post_norm (LayerNorm)
        expected_normed = merge.post_norm(expected_raw_memory)

        # Run actual forward pass
        _, new_state = merge(x, old_state)

    # Compare: the memory in new_state should match our manual computation
    assert torch.allclose(new_state.memory, expected_normed, atol=1e-5), (
        f"Affine rule invariant violated (with adaptive merge). "
        f"Max diff = {(new_state.memory - expected_normed).abs().max().item()}"
    )


# =============================================================================
# Feature: causal-matrix-merge-v2, Property 13: Merge Adaptativo Depende del Contenido de Memoria
# Validates: Requirements 6.1, 6.4
# =============================================================================


@given(
    model_dim=st.integers(min_value=16, max_value=64),
    state_dim=st.integers(min_value=8, max_value=32),
    num_slots=st.integers(min_value=4, max_value=12),
    write_rank=st.integers(min_value=1, max_value=4),
    batch_size=st.integers(min_value=1, max_value=4),
)
@settings(max_examples=100)
def test_adaptive_merge_depends_on_memory_content(
    model_dim: int, state_dim: int, num_slots: int, write_rank: int, batch_size: int
):
    """Property 13: Merge Adaptativo Depende del Contenido de Memoria.

    **Validates: Requirements 6.1, 6.4**

    Para cualquier token de entrada fijo h, cuando se aplica adaptive merge con
    dos estados de memoria diferentes (memory₁ ≠ memory₂), los valores de decay
    modulado SHALL diferir.
    """
    top_k_slots = min(num_slots, max(1, num_slots // 2))

    config = CausalMatrixMergeV2Config(
        vocab_size=100,
        model_dim=model_dim,
        state_dim=state_dim,
        num_slots=num_slots,
        num_layers=1,
        num_checkpoints=2,
        checkpoint_stride=16,
        write_rank=write_rank,
        dropout=0.0,
        use_residual_gate=False,
        max_context=128,
        ffn_mult=2.0,
        top_k_slots=top_k_slots,
        use_adaptive_merge=True,
        use_learned_checkpoints=False,
    )

    merge = CausalMatrixMergeV2(config)
    merge.eval()

    # Fixed input h [B, model_dim]
    h = torch.randn(batch_size, model_dim)

    # Compute decay_base once (shared between both calls)
    with torch.no_grad():
        h_proj = rms_norm(merge.in_proj(h))  # [B, model_dim]
        decay_base = torch.exp(-F.softplus(merge.decay_proj(h_proj))).unsqueeze(-1)  # [B, S, 1]

    # Two different memories — ensure they are sufficiently different
    memory1 = torch.randn(batch_size, num_slots, state_dim)
    # Add a meaningful offset to ensure memories are different
    memory2 = memory1 + torch.randn_like(memory1) * 2.0 + 1.0

    # Verify memories are indeed different
    assert not torch.allclose(memory1, memory2), "Memories should differ"

    # Call _adaptive_decay with the same h and decay_base but different memories
    with torch.no_grad():
        decay1 = merge._adaptive_decay(h_proj, decay_base, memory1)
        decay2 = merge._adaptive_decay(h_proj, decay_base, memory2)

    # Assert that decays differ — different memory content should produce different modulation
    assert not torch.allclose(decay1, decay2, atol=1e-6), (
        f"Adaptive merge decay should depend on memory content. "
        f"decay1 and decay2 should differ for different memories, "
        f"but max diff = {(decay1 - decay2).abs().max().item()}"
    )



# =============================================================================
# Feature: causal-matrix-merge-v2, Property 14: Checkpoint Selector es Query-Dependent
# Validates: Requirements 5.1
# =============================================================================

from pymbbo.architectures.causal_matrix_merge_v2.checkpoint_reader import CheckpointSelector
import torch.nn as nn


@given(
    state_dim=st.integers(min_value=8, max_value=32),
    num_checkpoints=st.integers(min_value=2, max_value=8),
    num_slots=st.integers(min_value=2, max_value=8),
    model_dim=st.integers(min_value=8, max_value=32),
    batch_size=st.integers(min_value=1, max_value=2),
)
@settings(max_examples=100)
def test_checkpoint_selector_is_query_dependent(
    state_dim: int,
    num_checkpoints: int,
    num_slots: int,
    model_dim: int,
    batch_size: int,
):
    """Property 14: Checkpoint Selector es Query-Dependent.

    **Validates: Requirements 5.1**

    Para cualquier par de queries distintos (q₁ ≠ q₂) y un mismo banco de
    checkpoints no-cero, el CheckpointSelector SHALL producir distribuciones
    de atención diferentes.
    """
    # Create CheckpointSelector
    checkpoint_selector = CheckpointSelector(state_dim, num_checkpoints)
    checkpoint_selector.eval()

    # Shared projections (like in the merge module)
    key_proj = nn.Linear(state_dim, state_dim, bias=False)
    value_proj = nn.Linear(state_dim, model_dim, bias=False)

    # Create non-zero checkpoints [B, K, S, state_dim]
    checkpoints = torch.randn(batch_size, num_checkpoints, num_slots, state_dim)
    # Ensure they are non-zero (add small offset to avoid degenerate case)
    checkpoints = checkpoints + 0.1

    # Create two distinct queries q1, q2 [B, state_dim]
    q1 = torch.randn(batch_size, state_dim)
    # Ensure q2 is different from q1 by adding a meaningful offset
    q2 = q1 + torch.randn(batch_size, state_dim) * 0.5 + 0.5

    # Ensure q1 != q2 (they should always differ due to offset, but double-check)
    assume(not torch.allclose(q1, q2, atol=1e-6))

    with torch.no_grad():
        out1 = checkpoint_selector(q1, checkpoints, key_proj, value_proj)
        out2 = checkpoint_selector(q2, checkpoints, key_proj, value_proj)

    # Outputs should differ because queries differ
    assert not torch.allclose(out1, out2, atol=1e-6), (
        f"CheckpointSelector produced identical outputs for different queries. "
        f"Max diff = {(out1 - out2).abs().max().item()}"
    )


# =============================================================================
# Feature: causal-matrix-merge-v2, Property 15: Promoción de Checkpoints Ocurre en Stride Correcto
# Validates: Requirements 7.6
# =============================================================================


@given(
    seq_len=st.integers(min_value=1, max_value=50),
    checkpoint_stride=st.integers(min_value=2, max_value=16),
    model_dim=st.integers(min_value=16, max_value=32),
    state_dim=st.integers(min_value=8, max_value=16),
    num_slots=st.integers(min_value=4, max_value=8),
    num_checkpoints=st.integers(min_value=2, max_value=6),
)
@settings(max_examples=100)
def test_checkpoint_promotion_occurs_at_correct_stride(
    seq_len: int,
    checkpoint_stride: int,
    model_dim: int,
    state_dim: int,
    num_slots: int,
    num_checkpoints: int,
):
    """Property 15: Promoción de Checkpoints Ocurre en Stride Correcto.

    **Validates: Requirements 7.6**

    Para cualquier secuencia de N tokens procesados, los checkpoints SHALL
    actualizarse exactamente floor(N / checkpoint_stride) veces, y cada
    actualización SHALL contener el snapshot de memoria en el paso correspondiente.

    Strategy:
    - Process seq_len tokens one by one through the merge block
    - Track checkpoint promotions by detecting when checkpoints change from
      their previous value
    - Expected promotions = floor(seq_len / checkpoint_stride), since step goes
      1, 2, ..., seq_len and promotion happens when step % stride == 0
    - Verify the count of non-zero checkpoint slots matches expectations
    """
    top_k_slots = min(num_slots, max(1, num_slots // 2))

    config = CausalMatrixMergeV2Config(
        vocab_size=100,
        model_dim=model_dim,
        state_dim=state_dim,
        num_slots=num_slots,
        num_layers=1,
        num_checkpoints=num_checkpoints,
        checkpoint_stride=checkpoint_stride,
        write_rank=2,
        dropout=0.0,
        use_residual_gate=False,
        max_context=128,
        ffn_mult=2.0,
        top_k_slots=top_k_slots,
        use_adaptive_merge=False,
        use_learned_checkpoints=False,
    )

    merge = CausalMatrixMergeV2(config)
    merge.eval()

    batch_size = 1

    # Initialize state
    state = merge.init_state(batch_size, dtype=torch.float32)

    # Verify initial checkpoints are all zero
    assert (state.checkpoints == 0).all(), "Initial checkpoints should be all zero"

    # Track promotions: count how many times checkpoints change
    promotion_count = 0
    prev_checkpoints = state.checkpoints.clone()

    with torch.no_grad():
        for t in range(seq_len):
            x = torch.randn(batch_size, model_dim)
            _, state = merge.forward(x, state)

            # Detect if a promotion occurred (checkpoints changed)
            if not torch.equal(state.checkpoints, prev_checkpoints):
                promotion_count += 1
                prev_checkpoints = state.checkpoints.clone()

    # Expected: promotion happens when step % stride == 0
    # Steps go 1, 2, ..., seq_len
    # Number of multiples of stride in [1, seq_len] = floor(seq_len / stride)
    expected_promotions = seq_len // checkpoint_stride

    assert promotion_count == expected_promotions, (
        f"Checkpoint promotions mismatch: got {promotion_count}, "
        f"expected {expected_promotions} (seq_len={seq_len}, "
        f"checkpoint_stride={checkpoint_stride})"
    )

    # Additional verification: after processing, the number of non-zero
    # checkpoints should be min(expected_promotions, num_checkpoints)
    # (because FIFO: once we exceed num_checkpoints, oldest get pushed out)
    expected_nonzero_slots = min(expected_promotions, num_checkpoints)

    # Count non-zero checkpoint positions (each checkpoint is [B, S, state_dim])
    nonzero_count = 0
    for k in range(num_checkpoints):
        ckpt_k = state.checkpoints[0, k]  # [S, state_dim]
        if not torch.equal(ckpt_k, torch.zeros_like(ckpt_k)):
            nonzero_count += 1

    assert nonzero_count == expected_nonzero_slots, (
        f"Non-zero checkpoints mismatch: got {nonzero_count}, "
        f"expected {expected_nonzero_slots} (promotions={expected_promotions}, "
        f"num_checkpoints={num_checkpoints})"
    )


# =============================================================================
# Feature: causal-matrix-merge-v2, Property 10: Forward Shape Invariant
# Validates: Requirements 9.1
# =============================================================================

from pymbbo.architectures.causal_matrix_merge_v2.model import CausalMatrixMergeModelV2


@given(
    batch_size=st.integers(min_value=1, max_value=4),
    seq_len=st.integers(min_value=1, max_value=20),
    vocab_size=st.integers(min_value=50, max_value=200),
)
@settings(max_examples=100)
def test_forward_shape_invariant(
    batch_size: int, seq_len: int, vocab_size: int
):
    """Property 10: Forward Shape Invariant.

    **Validates: Requirements 9.1**

    Para cualquier tensor de entrada de forma [B, T] con token IDs válidos,
    forward(x) SHALL retornar un tensor de forma [B, T, vocab_size].
    """
    # Create model with small dims for speed
    model = CausalMatrixMergeModelV2(
        vocab_size=vocab_size,
        model_dim=32,
        state_dim=16,
        num_slots=4,
        num_layers=1,
        num_checkpoints=2,
        checkpoint_stride=8,
        write_rank=2,
        dropout=0.0,
        use_residual_gate=True,
        max_context=64,
        ffn_mult=2.0,
        top_k_slots=2,
        use_adaptive_merge=True,
        use_learned_checkpoints=True,
    )
    model.eval()

    # Generate random token IDs in [0, vocab_size)
    x = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Call model.forward(x)
    with torch.no_grad():
        output = model.forward(x)

    # Assert output shape == (batch_size, seq_len, vocab_size)
    assert output.shape == (batch_size, seq_len, vocab_size), (
        f"Forward shape invariant violated: got {output.shape}, "
        f"expected ({batch_size}, {seq_len}, {vocab_size})"
    )


# =============================================================================
# Feature: causal-matrix-merge-v2, Property 11: Forward con Estados Reseteados es Determinista
# Validates: Requirements 9.2
# =============================================================================

from pymbbo.architectures.causal_matrix_merge_v2.model import CausalMatrixMergeModelV2


@given(
    batch_size=st.integers(min_value=1, max_value=3),
    seq_len=st.integers(min_value=1, max_value=15),
)
@settings(max_examples=100)
def test_forward_with_reset_states_is_deterministic(
    batch_size: int, seq_len: int
):
    """Property 11: Forward con Estados Reseteados es Determinista.

    **Validates: Requirements 9.2**

    Para cualquier input fijo, dos llamadas consecutivas a forward(x)
    (que resetean estado internamente) SHALL producir outputs idénticos.
    """
    # Create model with small dims for efficiency
    model = CausalMatrixMergeModelV2(
        vocab_size=50,
        model_dim=32,
        state_dim=16,
        num_slots=4,
        num_layers=2,
        num_checkpoints=2,
        checkpoint_stride=8,
        write_rank=2,
        dropout=0.0,
        use_residual_gate=True,
        max_context=64,
        ffn_mult=2.0,
        top_k_slots=2,
        use_adaptive_merge=True,
        use_learned_checkpoints=True,
    )
    model.eval()

    # Generate random token IDs [batch_size, seq_len]
    x = torch.randint(0, 50, (batch_size, seq_len))

    # Call forward twice with the same input
    with torch.no_grad():
        out1 = model.forward(x)
        out2 = model.forward(x)

    # Both outputs must be identical since forward() resets state internally
    assert torch.allclose(out1, out2, atol=1e-6), (
        f"Forward is not deterministic with reset states. "
        f"Max diff = {(out1 - out2).abs().max().item()}"
    )


# =============================================================================
# Feature: causal-matrix-merge-v2, Property 12: Diferenciabilidad End-to-End
# Validates: Requirements 3.3, 4.4, 5.6, 6.7
# =============================================================================

from pymbbo.architectures.causal_matrix_merge_v2.model import CausalMatrixMergeModelV2


@given(
    batch_size=st.integers(min_value=1, max_value=3),
    seq_len=st.integers(min_value=2, max_value=10),
)
@settings(max_examples=100)
def test_differentiability_end_to_end(batch_size: int, seq_len: int):
    """Property 12: Diferenciabilidad End-to-End.

    **Validates: Requirements 3.3, 4.4, 5.6, 6.7**

    Para cualquier tensor de entrada válido con batch_size >= 1, una pasada
    forward seguida de loss.backward() SHALL producir gradientes no-None para
    todos los parámetros entrenables del modelo.
    """
    vocab_size = 50

    # Create model with small dims for speed, train mode
    model = CausalMatrixMergeModelV2(
        vocab_size=vocab_size,
        model_dim=32,
        state_dim=16,
        num_slots=4,
        num_layers=2,
        num_checkpoints=2,
        checkpoint_stride=8,
        write_rank=2,
        dropout=0.0,
        use_residual_gate=True,
        max_context=64,
        ffn_mult=2.0,
        top_k_slots=2,
        use_adaptive_merge=True,
        use_learned_checkpoints=True,
    )
    model.train()

    # Generate random token IDs [batch_size, seq_len]
    x = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Forward pass → logits [B, T, vocab_size]
    logits = model.forward(x)

    # Compute cross-entropy loss with random targets
    # targets shifted: predict next token (standard autoregressive)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, vocab_size), targets.view(-1)
    )

    # Backward pass
    loss.backward()

    # Check that ALL trainable parameters have non-None gradients
    params_without_grad = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, (
                f"Parameter '{name}' has requires_grad=True but grad is None "
                f"after loss.backward(). This breaks end-to-end differentiability."
            )
            params_without_grad.append(name) if p.grad is None else None

    # Also verify we actually have trainable parameters
    trainable_count = sum(1 for p in model.parameters() if p.requires_grad)
    assert trainable_count > 0, "Model has no trainable parameters"

    # Zero gradients for clean state (avoid hypothesis state leakage)
    model.zero_grad()


# =============================================================================
# Feature: causal-matrix-merge-v2, Property 9: Tamaño de Estado Fijo Independiente de Longitud de Secuencia
# Validates: Requirements 7.1
# =============================================================================

from pymbbo.architectures.causal_matrix_merge_v2.model import CausalMatrixMergeModelV2


@given(
    seq_len1=st.integers(min_value=1, max_value=30),
    seq_len2=st.integers(min_value=1, max_value=30),
    batch_size=st.integers(min_value=1, max_value=4),
    num_slots=st.integers(min_value=4, max_value=8),
    state_dim=st.integers(min_value=8, max_value=32),
    num_checkpoints=st.integers(min_value=2, max_value=6),
)
@settings(max_examples=100, deadline=None)
def test_fixed_state_size_independent_of_sequence_length(
    seq_len1: int,
    seq_len2: int,
    batch_size: int,
    num_slots: int,
    state_dim: int,
    num_checkpoints: int,
):
    """Property 9: Tamaño de Estado Fijo Independiente de Longitud de Secuencia.

    **Validates: Requirements 7.1**

    Para cualquier par de secuencias de longitudes diferentes (T₁ ≠ T₂)
    procesadas por el modelo, los tensores del MergeState resultante SHALL
    tener formas idénticas: memory [B, S, Ds], normalizer [B, S, 1],
    checkpoints [B, K, S, Ds].
    """
    # Ensure T1 != T2
    assume(seq_len1 != seq_len2)

    model_dim = 32
    top_k_slots = min(num_slots, max(1, num_slots // 2))
    vocab_size = 100

    # Create model with given hyperparameters
    model = CausalMatrixMergeModelV2(
        vocab_size=vocab_size,
        model_dim=model_dim,
        state_dim=state_dim,
        num_slots=num_slots,
        num_layers=2,
        num_checkpoints=num_checkpoints,
        checkpoint_stride=8,
        write_rank=2,
        dropout=0.0,
        use_residual_gate=True,
        max_context=128,
        ffn_mult=2.0,
        top_k_slots=top_k_slots,
        use_adaptive_merge=True,
        use_learned_checkpoints=True,
    )
    model.eval()

    with torch.no_grad():
        # Process sequence of length T1
        x1 = torch.randint(0, vocab_size, (batch_size, seq_len1))
        _ = model(x1)
        state1 = model.get_state()

        # Process sequence of length T2
        x2 = torch.randint(0, vocab_size, (batch_size, seq_len2))
        _ = model(x2)
        state2 = model.get_state()

    # For each layer, verify state tensor shapes are identical
    assert len(state1) == len(state2), (
        f"Number of layer states differs: {len(state1)} vs {len(state2)}"
    )

    for layer_idx in range(len(state1)):
        s1 = state1[layer_idx]
        s2 = state2[layer_idx]

        # Memory shape: [B, S, Ds]
        assert s1["memory"].shape == s2["memory"].shape, (
            f"Layer {layer_idx}: memory shape differs for T1={seq_len1} vs T2={seq_len2}. "
            f"Got {s1['memory'].shape} vs {s2['memory'].shape}"
        )
        assert s1["memory"].shape == (batch_size, num_slots, state_dim), (
            f"Layer {layer_idx}: memory shape should be [B={batch_size}, S={num_slots}, Ds={state_dim}], "
            f"got {s1['memory'].shape}"
        )

        # Normalizer shape: [B, S, 1]
        assert s1["normalizer"].shape == s2["normalizer"].shape, (
            f"Layer {layer_idx}: normalizer shape differs for T1={seq_len1} vs T2={seq_len2}. "
            f"Got {s1['normalizer'].shape} vs {s2['normalizer'].shape}"
        )
        assert s1["normalizer"].shape == (batch_size, num_slots, 1), (
            f"Layer {layer_idx}: normalizer shape should be [B={batch_size}, S={num_slots}, 1], "
            f"got {s1['normalizer'].shape}"
        )

        # Checkpoints shape: [B, K, S, Ds]
        assert s1["checkpoints"].shape == s2["checkpoints"].shape, (
            f"Layer {layer_idx}: checkpoints shape differs for T1={seq_len1} vs T2={seq_len2}. "
            f"Got {s1['checkpoints'].shape} vs {s2['checkpoints'].shape}"
        )
        assert s1["checkpoints"].shape == (batch_size, num_checkpoints, num_slots, state_dim), (
            f"Layer {layer_idx}: checkpoints shape should be "
            f"[B={batch_size}, K={num_checkpoints}, S={num_slots}, Ds={state_dim}], "
            f"got {s1['checkpoints'].shape}"
        )


# =============================================================================
# Feature: causal-matrix-merge-v2, Property 16: Persistencia Round-Trip del Modelo
# Validates: Requirements 1.3, 9.6
# =============================================================================

import tempfile
import os

from pymbbo.models.factory import build_model
from pymbbo.models.persistence import save_model, load_model


@given(
    model_dim=st.sampled_from([32, 64]),
    state_dim=st.sampled_from([16, 32]),
    num_slots=st.integers(2, 6),
    num_layers=st.integers(1, 2),
    batch_size=st.integers(1, 2),
    seq_len=st.integers(2, 8),
)
@settings(max_examples=20, deadline=None)
def test_persistence_round_trip(
    model_dim: int, state_dim: int, num_slots: int, num_layers: int,
    batch_size: int, seq_len: int
):
    """Property 16: Persistencia Round-Trip del Modelo.

    **Validates: Requirements 1.3, 9.6**

    Para cualquier modelo CMM v2 instanciado con configuración aleatoria válida,
    save_model seguido de load_model SHALL producir un modelo que genera outputs
    idénticos (dado el mismo input y semilla) al modelo original.
    """
    vocab_size = 50
    top_k_slots = min(num_slots, max(1, num_slots // 2))

    # Build model via factory (returns BaseModel wrapping CausalMatrixMergeModelV2)
    model = build_model(
        "causal_matrix_merge_v2",
        vocab_size=vocab_size,
        model_dim=model_dim,
        state_dim=state_dim,
        num_slots=num_slots,
        num_layers=num_layers,
        num_checkpoints=2,
        checkpoint_stride=8,
        write_rank=2,
        dropout=0.0,
        use_residual_gate=True,
        max_context=64,
        ffn_mult=2.0,
        top_k_slots=top_k_slots,
        use_adaptive_merge=True,
        use_learned_checkpoints=True,
    )
    model.eval()

    # Create random input x [B, T]
    x = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Get output before save
    with torch.no_grad():
        out_before = model(x)

    # Save model to temp file and load it back
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        path = f.name

    try:
        save_model(model, path)
        loaded = load_model(path)
        loaded.eval()

        # Get output after load
        with torch.no_grad():
            out_after = loaded(x)

        # Assert outputs are identical (within floating point tolerance)
        assert torch.allclose(out_before, out_after, atol=1e-6), (
            f"Persistence round-trip failed: outputs differ after save/load. "
            f"Max diff = {(out_before - out_after).abs().max().item()}"
        )
    finally:
        if os.path.exists(path):
            os.unlink(path)
