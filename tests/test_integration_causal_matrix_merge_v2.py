"""
Integration tests for Causal Matrix Merge v2 architecture.

Validates requirements 1.1, 1.2, 1.4, 9.1, 9.3:
  - v1 and v2 can be instantiated simultaneously without conflict
  - build_model("causal_matrix_merge_v2", ...) returns a functional model
  - Full training step: forward + backward + optimizer.step
  - Generation of 50+ tokens without errors
  - Edge cases: empty sequence, empty prompt, max_new_tokens=0
"""

import os
import sys

import pytest
import torch
import torch.nn.functional as F

# Ensure project root is on sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from pymbbo.models.factory import build_model
from pymbbo.architectures.causal_matrix_merge_v2.model import CausalMatrixMergeModelV2


class TestV2Integration:
    """Integration tests for Causal Matrix Merge v2."""

    def test_v1_v2_coexistence(self):
        """v1 and v2 can be instantiated simultaneously without conflict."""
        from pymbbo.architectures.causal_matrix_merge.model import CausalMatrixMergeModel

        v1 = build_model("causal_matrix_merge", vocab_size=100, model_dim=64,
                         state_dim=32, num_slots=4, num_layers=2, write_rank=2,
                         num_checkpoints=2, checkpoint_stride=8)
        v2 = build_model("causal_matrix_merge_v2", vocab_size=100, model_dim=64,
                         state_dim=32, num_slots=4, num_layers=2, write_rank=2,
                         num_checkpoints=2, checkpoint_stride=8)

        # Both should be functional BaseModel instances with correct architectures
        assert isinstance(v1.architecture, CausalMatrixMergeModel)
        assert isinstance(v2.architecture, CausalMatrixMergeModelV2)

        # Both should work independently on the same input
        x = torch.randint(0, 100, (2, 10))
        out_v1 = v1(x)
        out_v2 = v2(x)
        assert out_v1.shape == out_v2.shape == (2, 10, 100)

    def test_build_model_works(self):
        """build_model("causal_matrix_merge_v2", ...) returns functional model."""
        model = build_model("causal_matrix_merge_v2", vocab_size=200, model_dim=64,
                            state_dim=32, num_slots=4, num_layers=2, write_rank=2,
                            num_checkpoints=2, checkpoint_stride=8)

        assert isinstance(model.architecture, CausalMatrixMergeModelV2)

        x = torch.randint(0, 200, (1, 5))
        out = model(x)
        assert out.shape == (1, 5, 200)

    def test_training_minibatch(self):
        """Full training step: forward + backward + optimizer.step."""
        model = CausalMatrixMergeModelV2(
            vocab_size=100, model_dim=64, state_dim=32,
            num_slots=4, num_layers=2, write_rank=2,
            num_checkpoints=2, checkpoint_stride=4,
        )
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        x = torch.randint(0, 100, (4, 20))
        targets = torch.randint(0, 100, (4, 20))

        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, 100), targets.view(-1))
        loss.backward()
        optimizer.step()

        assert loss.item() > 0  # Loss should be positive
        assert not torch.isnan(loss)

    def test_generation_50_tokens(self):
        """Generate 50+ tokens without errors or NaN."""
        model = CausalMatrixMergeModelV2(
            vocab_size=100, model_dim=64, state_dim=32,
            num_slots=4, num_layers=2, write_rank=2,
            num_checkpoints=2, checkpoint_stride=8,
        )
        prompt = torch.randint(0, 100, (1, 5))
        output = model.generate(prompt, max_new_tokens=55, temperature=1.0)

        assert output.shape == (1, 60)  # 5 prompt + 55 generated
        assert (output >= 0).all() and (output < 100).all()

    def test_empty_sequence(self):
        """Forward with T=0 returns empty logits."""
        model = CausalMatrixMergeModelV2(
            vocab_size=100, model_dim=64, state_dim=32,
            num_slots=4, num_layers=2, write_rank=2,
        )
        model.eval()
        x = torch.randint(0, 100, (2, 0))
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 0, 100)

    def test_empty_prompt_generate(self):
        """Generate from empty prompt uses dummy token."""
        model = CausalMatrixMergeModelV2(
            vocab_size=100, model_dim=64, state_dim=32,
            num_slots=4, num_layers=2, write_rank=2,
        )
        prompt = torch.zeros(1, 0, dtype=torch.long)
        output = model.generate(prompt, max_new_tokens=10)
        assert output.shape == (1, 10)

    def test_max_new_tokens_zero(self):
        """max_new_tokens=0 returns prompt unchanged."""
        model = CausalMatrixMergeModelV2(
            vocab_size=100, model_dim=64, state_dim=32,
            num_slots=4, num_layers=2, write_rank=2,
        )
        prompt = torch.randint(0, 100, (1, 5))
        output = model.generate(prompt, max_new_tokens=0)
        assert torch.equal(output, prompt)
