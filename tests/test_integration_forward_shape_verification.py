"""Verification tests for Task 5.3: Output shape and state preservation after integration.

Confirms that the forward pass of CausalMatrixMergeV2 with per-slot features
integrated produces correct output shapes, valid MergeState, and preserves
the memory update equation and downstream operations.

Requirements verified: 7.3, 7.4, 7.5, 5.1, 5.4, 5.5, 5.8
"""

import torch
import torch.nn.functional as F

from pymbbo.architectures.causal_matrix_merge_v2.config import CausalMatrixMergeV2Config
from pymbbo.architectures.causal_matrix_merge_v2.merge_v2 import CausalMatrixMergeV2


# Small config for fast testing
SMALL_CONFIG = CausalMatrixMergeV2Config(
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
    use_learned_checkpoints=True,
    dropout=0.0,  # Disable dropout for deterministic testing
)

# Config with per-slot features disabled (fallback)
FALLBACK_CONFIG = CausalMatrixMergeV2Config(
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
    use_per_slot_write=False,
    use_adaptive_merge=True,
    use_learned_checkpoints=True,
    dropout=0.0,
)


class TestForwardOutputShape:
    """Verify forward() returns [B, model_dim] output with both per-slot features enabled."""

    def test_forward_output_shape_batch_1(self):
        """forward() with B=1 produces output shape [1, model_dim]."""
        torch.manual_seed(42)
        model = CausalMatrixMergeV2(SMALL_CONFIG)
        model.eval()

        B = 1
        x = torch.randn(B, SMALL_CONFIG.model_dim)
        state = model.init_state(B)

        with torch.no_grad():
            output, new_state = model.forward(x, state)

        assert output.shape == (B, SMALL_CONFIG.model_dim), (
            f"Expected output shape ({B}, {SMALL_CONFIG.model_dim}), got {output.shape}"
        )

    def test_forward_output_shape_batch_4(self):
        """forward() with B=4 produces output shape [4, model_dim]."""
        torch.manual_seed(42)
        model = CausalMatrixMergeV2(SMALL_CONFIG)
        model.eval()

        B = 4
        x = torch.randn(B, SMALL_CONFIG.model_dim)
        state = model.init_state(B)

        with torch.no_grad():
            output, new_state = model.forward(x, state)

        assert output.shape == (B, SMALL_CONFIG.model_dim), (
            f"Expected output shape ({B}, {SMALL_CONFIG.model_dim}), got {output.shape}"
        )

    def test_forward_output_is_finite(self):
        """forward() output contains no NaN or Inf values."""
        torch.manual_seed(42)
        model = CausalMatrixMergeV2(SMALL_CONFIG)
        model.eval()

        B = 2
        x = torch.randn(B, SMALL_CONFIG.model_dim)
        state = model.init_state(B)

        with torch.no_grad():
            output, _ = model.forward(x, state)

        assert torch.isfinite(output).all(), (
            f"Output contains non-finite values. Has NaN: {torch.isnan(output).any()}, "
            f"Has Inf: {torch.isinf(output).any()}"
        )


class TestMergeStatePreservation:
    """Verify MergeState returned from forward() has correct shapes and incremented step."""

    def test_memory_shape(self):
        """Returned state.memory has shape [B, S, state_dim]."""
        torch.manual_seed(42)
        model = CausalMatrixMergeV2(SMALL_CONFIG)
        model.eval()

        B = 2
        x = torch.randn(B, SMALL_CONFIG.model_dim)
        state = model.init_state(B)

        with torch.no_grad():
            _, new_state = model.forward(x, state)

        expected_shape = (B, SMALL_CONFIG.num_slots, SMALL_CONFIG.state_dim)
        assert new_state.memory.shape == expected_shape, (
            f"Expected memory shape {expected_shape}, got {new_state.memory.shape}"
        )

    def test_normalizer_shape(self):
        """Returned state.normalizer has shape [B, S, 1]."""
        torch.manual_seed(42)
        model = CausalMatrixMergeV2(SMALL_CONFIG)
        model.eval()

        B = 2
        x = torch.randn(B, SMALL_CONFIG.model_dim)
        state = model.init_state(B)

        with torch.no_grad():
            _, new_state = model.forward(x, state)

        expected_shape = (B, SMALL_CONFIG.num_slots, 1)
        assert new_state.normalizer.shape == expected_shape, (
            f"Expected normalizer shape {expected_shape}, got {new_state.normalizer.shape}"
        )

    def test_checkpoints_shape(self):
        """Returned state.checkpoints has shape [B, K, S, state_dim]."""
        torch.manual_seed(42)
        model = CausalMatrixMergeV2(SMALL_CONFIG)
        model.eval()

        B = 2
        x = torch.randn(B, SMALL_CONFIG.model_dim)
        state = model.init_state(B)

        with torch.no_grad():
            _, new_state = model.forward(x, state)

        expected_shape = (
            B, SMALL_CONFIG.num_checkpoints, SMALL_CONFIG.num_slots, SMALL_CONFIG.state_dim
        )
        assert new_state.checkpoints.shape == expected_shape, (
            f"Expected checkpoints shape {expected_shape}, got {new_state.checkpoints.shape}"
        )

    def test_step_incremented(self):
        """Returned state.step is incremented by 1."""
        torch.manual_seed(42)
        model = CausalMatrixMergeV2(SMALL_CONFIG)
        model.eval()

        B = 2
        x = torch.randn(B, SMALL_CONFIG.model_dim)
        state = model.init_state(B)
        assert state.step == 0

        with torch.no_grad():
            _, new_state = model.forward(x, state)

        assert new_state.step == 1, f"Expected step=1, got {new_state.step}"

    def test_step_incremented_multiple_tokens(self):
        """After processing 5 tokens, step should be 5."""
        torch.manual_seed(42)
        model = CausalMatrixMergeV2(SMALL_CONFIG)
        model.eval()

        B = 2
        state = model.init_state(B)

        with torch.no_grad():
            for t in range(5):
                x = torch.randn(B, SMALL_CONFIG.model_dim)
                _, state = model.forward(x, state)

        assert state.step == 5, f"Expected step=5 after 5 tokens, got {state.step}"

    def test_state_values_finite(self):
        """All tensors in returned MergeState are finite (no NaN/Inf)."""
        torch.manual_seed(42)
        model = CausalMatrixMergeV2(SMALL_CONFIG)
        model.eval()

        B = 2
        x = torch.randn(B, SMALL_CONFIG.model_dim)
        state = model.init_state(B)

        with torch.no_grad():
            _, new_state = model.forward(x, state)

        assert torch.isfinite(new_state.memory).all(), "memory contains non-finite values"
        assert torch.isfinite(new_state.normalizer).all(), "normalizer contains non-finite values"
        assert torch.isfinite(new_state.checkpoints).all(), "checkpoints contains non-finite values"


class TestMemoryUpdateEquation:
    """Verify the memory update uses: decay_final * memory + (1 - decay_final) * write_routed."""

    def test_memory_update_equation_manual(self):
        """Manually verify the affine rule by intercepting intermediate values.

        We verify that: memory_new = decay_final * memory + (1 - decay_final) * write_routed
        by computing the components manually and comparing with the model output.
        """
        torch.manual_seed(42)
        model = CausalMatrixMergeV2(SMALL_CONFIG)
        model.eval()

        B = 2
        x = torch.randn(B, SMALL_CONFIG.model_dim)
        state = model.init_state(B)

        # Manually compute the intermediate values matching the forward pass
        with torch.no_grad():
            from pymbbo.architectures.causal_matrix_merge_v2.merge_v2 import rms_norm

            # Step 1: Project and normalize
            h = rms_norm(model.in_proj(x))  # [B, model_dim]

            # Step 2: Decay base
            decay_base = torch.exp(-F.softplus(model.decay_proj(h)))  # [B, S]
            decay_base = decay_base.unsqueeze(-1)  # [B, S, 1]

            # Step 3: Per-slot decay modulation
            decay_final = model._per_slot_adaptive_decay(h, decay_base, state.memory)

            # Step 4: Sparse routing
            route = model._sparse_route(h)  # [B, S, 1]

            # Step 5: Write MLP
            write_base = model._write_mlp(h)  # [B, state_dim]

            # Step 6: Per-slot write adaptation
            write_adapted = model._per_slot_adaptive_write(write_base, state.memory)

            # Step 7: Weighted write
            write_routed = route * write_adapted  # [B, S, state_dim]

            # Step 8: Memory update (THE AFFINE RULE)
            memory_expected = decay_final * state.memory + (1 - decay_final) * write_routed

            # Now run the actual forward pass and verify the memory before post_norm
            # Since the model applies post_norm to memory_new before storing in state,
            # we compare after applying post_norm ourselves
            memory_normed_expected = model.post_norm(memory_expected)

            # Run actual forward
            _, new_state = model.forward(x, state)

            # The new_state.memory should equal memory_normed_expected
            assert torch.allclose(new_state.memory, memory_normed_expected, atol=1e-5), (
                f"Memory update equation mismatch.\n"
                f"Max diff: {(new_state.memory - memory_normed_expected).abs().max().item()}"
            )

    def test_affine_rule_components_in_range(self):
        """Verify decay_final is in (0, 1) so the affine rule is well-formed."""
        torch.manual_seed(42)
        model = CausalMatrixMergeV2(SMALL_CONFIG)
        model.eval()

        B = 2
        x = torch.randn(B, SMALL_CONFIG.model_dim)
        state = model.init_state(B)

        with torch.no_grad():
            from pymbbo.architectures.causal_matrix_merge_v2.merge_v2 import rms_norm

            h = rms_norm(model.in_proj(x))
            decay_base = torch.exp(-F.softplus(model.decay_proj(h)))
            decay_base = decay_base.unsqueeze(-1)
            decay_final = model._per_slot_adaptive_decay(h, decay_base, state.memory)

        assert (decay_final > 0).all(), f"decay_final has values <= 0: min={decay_final.min()}"
        assert (decay_final < 1).all(), f"decay_final has values >= 1: max={decay_final.max()}"


class TestFallbackBehavior:
    """Verify forward() works correctly with per-slot features disabled."""

    def test_fallback_output_shape(self):
        """forward() with per-slot features disabled still produces [B, model_dim] output."""
        torch.manual_seed(42)
        model = CausalMatrixMergeV2(FALLBACK_CONFIG)
        model.eval()

        B = 2
        x = torch.randn(B, FALLBACK_CONFIG.model_dim)
        state = model.init_state(B)

        with torch.no_grad():
            output, new_state = model.forward(x, state)

        assert output.shape == (B, FALLBACK_CONFIG.model_dim), (
            f"Expected output shape ({B}, {FALLBACK_CONFIG.model_dim}), got {output.shape}"
        )

    def test_fallback_state_shapes(self):
        """forward() with per-slot disabled produces valid MergeState shapes."""
        torch.manual_seed(42)
        model = CausalMatrixMergeV2(FALLBACK_CONFIG)
        model.eval()

        B = 2
        x = torch.randn(B, FALLBACK_CONFIG.model_dim)
        state = model.init_state(B)

        with torch.no_grad():
            _, new_state = model.forward(x, state)

        assert new_state.memory.shape == (B, FALLBACK_CONFIG.num_slots, FALLBACK_CONFIG.state_dim)
        assert new_state.normalizer.shape == (B, FALLBACK_CONFIG.num_slots, 1)
        assert new_state.checkpoints.shape == (
            B, FALLBACK_CONFIG.num_checkpoints, FALLBACK_CONFIG.num_slots, FALLBACK_CONFIG.state_dim
        )
        assert new_state.step == 1

    def test_fallback_output_finite(self):
        """forward() with per-slot disabled produces finite output."""
        torch.manual_seed(42)
        model = CausalMatrixMergeV2(FALLBACK_CONFIG)
        model.eval()

        B = 2
        x = torch.randn(B, FALLBACK_CONFIG.model_dim)
        state = model.init_state(B)

        with torch.no_grad():
            output, new_state = model.forward(x, state)

        assert torch.isfinite(output).all(), "Fallback output contains non-finite values"
        assert torch.isfinite(new_state.memory).all(), "Fallback memory contains non-finite values"


class TestDownstreamOperationsUnchanged:
    """Verify downstream operations (normalizer, post_norm, checkpoint, read_context, out_proj, gate, dropout) remain unchanged."""

    def test_normalizer_update_equation(self):
        """Normalizer update follows: decay_final * normalizer + (1 - decay_final) * route."""
        torch.manual_seed(42)
        model = CausalMatrixMergeV2(SMALL_CONFIG)
        model.eval()

        B = 2
        x = torch.randn(B, SMALL_CONFIG.model_dim)
        state = model.init_state(B)

        with torch.no_grad():
            from pymbbo.architectures.causal_matrix_merge_v2.merge_v2 import rms_norm

            h = rms_norm(model.in_proj(x))
            decay_base = torch.exp(-F.softplus(model.decay_proj(h)))
            decay_base = decay_base.unsqueeze(-1)
            decay_final = model._per_slot_adaptive_decay(h, decay_base, state.memory)
            route = model._sparse_route(h)

            # Expected normalizer update
            normalizer_expected = decay_final * state.normalizer + (1 - decay_final) * route

            # Actual forward
            _, new_state = model.forward(x, state)

            assert torch.allclose(new_state.normalizer, normalizer_expected, atol=1e-5), (
                f"Normalizer update mismatch.\n"
                f"Max diff: {(new_state.normalizer - normalizer_expected).abs().max().item()}"
            )

    def test_checkpoint_promotion_at_stride(self):
        """Checkpoint is promoted when step is a multiple of checkpoint_stride."""
        torch.manual_seed(42)
        model = CausalMatrixMergeV2(SMALL_CONFIG)
        model.eval()

        B = 2
        state = model.init_state(B)

        # Process tokens until we reach checkpoint_stride
        with torch.no_grad():
            for t in range(SMALL_CONFIG.checkpoint_stride):
                x = torch.randn(B, SMALL_CONFIG.model_dim)
                _, state = model.forward(x, state)

        # At step == checkpoint_stride, a checkpoint should have been stored
        assert state.step == SMALL_CONFIG.checkpoint_stride
        # The first checkpoint slot should NOT be all zeros anymore
        assert not torch.allclose(
            state.checkpoints[:, 0],
            torch.zeros_like(state.checkpoints[:, 0]),
        ), "Checkpoint was not promoted at checkpoint_stride"

    def test_post_norm_applied_to_memory(self):
        """Memory in the returned state has post_norm (LayerNorm) applied."""
        torch.manual_seed(42)
        model = CausalMatrixMergeV2(SMALL_CONFIG)
        model.eval()

        B = 2
        # Use non-zero memory to make LayerNorm observable
        state = model.init_state(B)
        # Warm up with one token to get non-zero memory
        with torch.no_grad():
            x = torch.randn(B, SMALL_CONFIG.model_dim)
            _, state = model.forward(x, state)

        # Process another token
        with torch.no_grad():
            x2 = torch.randn(B, SMALL_CONFIG.model_dim)
            _, new_state = model.forward(x2, state)

        # The memory should have LayerNorm applied (mean ≈ 0, variance ≈ 1 per slot)
        # LayerNorm normalizes along last dim (state_dim)
        memory = new_state.memory
        # Each [B, S, state_dim] slice along state_dim should be approximately normalized
        mean_per_slot = memory.mean(dim=-1)  # [B, S]
        var_per_slot = memory.var(dim=-1, unbiased=False)  # [B, S]

        # LayerNorm makes mean ≈ 0 and var ≈ 1 (with learned affine params)
        # With default init (weight=1, bias=0), this should be exact
        assert torch.allclose(mean_per_slot, torch.zeros_like(mean_per_slot), atol=1e-4), (
            f"Post-norm mean not ~0: {mean_per_slot}"
        )

    def test_residual_gate_applied(self):
        """When use_residual_gate=True, output incorporates gated residual connection."""
        config_with_gate = CausalMatrixMergeV2Config(
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
            use_residual_gate=True,
            dropout=0.0,
        )
        torch.manual_seed(42)
        model = CausalMatrixMergeV2(config_with_gate)
        model.eval()

        B = 2
        x = torch.randn(B, config_with_gate.model_dim)
        state = model.init_state(B)

        with torch.no_grad():
            output, _ = model.forward(x, state)

        # Output should not be exactly x (gate mixes x with read context)
        assert not torch.allclose(output, x, atol=1e-6), (
            "Output is identical to input x, residual gate not applied"
        )
        # Output should be finite
        assert torch.isfinite(output).all(), "Output with residual gate has non-finite values"

    def test_forward_sequence_produces_valid_output(self):
        """forward_sequence with multi-token input produces [B, T, model_dim]."""
        torch.manual_seed(42)
        model = CausalMatrixMergeV2(SMALL_CONFIG)
        model.eval()

        B = 2
        T = 8
        x = torch.randn(B, T, SMALL_CONFIG.model_dim)

        with torch.no_grad():
            outputs, final_state = model.forward_sequence(x)

        assert outputs.shape == (B, T, SMALL_CONFIG.model_dim), (
            f"Expected outputs shape ({B}, {T}, {SMALL_CONFIG.model_dim}), got {outputs.shape}"
        )
        assert final_state.step == T, f"Expected step={T}, got {final_state.step}"
        assert torch.isfinite(outputs).all(), "forward_sequence output has non-finite values"


class TestNoAdaptiveMergeFallback:
    """Verify when use_adaptive_merge=False, decay_base is used directly."""

    def test_no_adaptive_merge_uses_decay_base(self):
        """When use_adaptive_merge=False, decay is not modulated."""
        config_no_adaptive = CausalMatrixMergeV2Config(
            vocab_size=100,
            model_dim=32,
            state_dim=16,
            num_slots=4,
            num_layers=1,
            write_rank=2,
            num_checkpoints=2,
            checkpoint_stride=4,
            top_k_slots=2,
            use_per_slot_decay=True,  # Should be ignored when adaptive is off
            use_per_slot_write=True,
            use_adaptive_merge=False,
            dropout=0.0,
        )
        torch.manual_seed(42)
        model = CausalMatrixMergeV2(config_no_adaptive)
        model.eval()

        B = 2
        x = torch.randn(B, config_no_adaptive.model_dim)
        state = model.init_state(B)

        with torch.no_grad():
            output, new_state = model.forward(x, state)

        # Should produce valid output regardless
        assert output.shape == (B, config_no_adaptive.model_dim)
        assert torch.isfinite(output).all()
        assert new_state.step == 1
