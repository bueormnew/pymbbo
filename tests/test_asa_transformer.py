import unittest
import torch
import numpy as np

from pymbbo import build_model, Dataset, token_scaling_benchmark, compare_models
from pymbbo.architectures.asa_transformer import (
    ASATransformerArchitecture,
    AdaptiveSelectiveAttention,
    ASALayerGroup
)


class TestASATransformer(unittest.TestCase):
    
    def setUp(self):
        self.vocab_size = 200
        self.d_model = 64
        self.nhead = 2
        self.num_layers = 4
        self.max_seq_len = 256
        self.group_size = 2
        self.max_a = 16

    def test_01_instantiation_and_registry(self):
        """Test model instantiation through pymbbo build_model factory using registered names."""
        model_asa = build_model(
            "asa_transformer",
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            max_seq_len=self.max_seq_len,
            group_size=self.group_size,
            max_a=self.max_a
        )
        self.assertIsInstance(model_asa.architecture, ASATransformerArchitecture)
        
        # Test alias asa_gpt
        model_gpt = build_model(
            "asa_gpt",
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            max_seq_len=self.max_seq_len
        )
        self.assertIsNotNone(model_gpt)

    def test_02_forward_pass_and_dynamic_max_a(self):
        """Test forward pass shapes and dynamic selection budget 'max_a' parameter at runtime."""
        model = build_model(
            "asa_transformer",
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            max_seq_len=self.max_seq_len,
            group_size=2
        )
        
        batch_size = 4
        seq_len = 32
        x = torch.randint(0, self.vocab_size, (batch_size, seq_len))
        
        # Default forward
        logits = model(x)
        self.assertEqual(logits.shape, (batch_size, seq_len, self.vocab_size))

        # Dynamic max_a = 8
        logits_8 = model(x, max_a=8)
        self.assertEqual(logits_8.shape, (batch_size, seq_len, self.vocab_size))

        # Dynamic max_a = 32
        logits_32 = model(x, max_a=32)
        self.assertEqual(logits_32.shape, (batch_size, seq_len, self.vocab_size))

    def test_03_selection_sharing_and_router_follower(self):
        """Test selection sharing across layer groups (1 router layer + g-1 follower layers)."""
        group = ASALayerGroup(
            group_size=3,
            d_model=self.d_model,
            nhead=self.nhead,
            max_a=8
        )
        
        B, N = 2, 16
        x = torch.randn(B, N, self.d_model)
        
        out, aux_loss, kv_caches = group(x, max_a=8, return_aux_loss=True)
        self.assertEqual(out.shape, (B, N, self.d_model))
        self.assertIsNotNone(aux_loss)
        
        # Check router is layer 0 and followers are layers 1 and 2
        self.assertTrue(group.layers[0].attn.is_router)
        self.assertFalse(group.layers[1].attn.is_router)
        self.assertFalse(group.layers[2].attn.is_router)

    def test_04_auxiliary_margin_loss(self):
        """Test auxiliary margin loss calculation and backpropagation."""
        model_wrap = build_model(
            "asa_transformer",
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=2,
            margin_loss_weight=0.05
        )
        model = model_wrap.architecture
        
        x = torch.randint(0, self.vocab_size, (2, 20))
        logits, aux_loss = model(x, max_a=8, return_aux_loss=True)
        
        self.assertIsNotNone(aux_loss)
        self.assertTrue(aux_loss.requires_grad)
        
        # Backward check
        loss = logits.sum() + 0.05 * aux_loss
        loss.backward()
        
        # Ensure gradients exist
        for param in model.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)
                break

    def test_05_kv_cache_autoregressive_generation(self):
        """Test fast autoregressive generate() method using KV caching."""
        model_wrap = build_model(
            "asa_transformer",
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=2,
            max_seq_len=self.max_seq_len
        )
        
        prompt = torch.tensor([[10, 20, 30, 40]])
        max_new_tokens = 15
        
        # Generate with max_a=8
        generated = model_wrap.architecture.generate(prompt, max_new_tokens=max_new_tokens, max_a=8)
        self.assertEqual(generated.shape, (1, 4 + max_new_tokens))
        self.assertTrue(torch.equal(generated[:, :4], prompt))

    def test_06_model_fit_and_benchmarking(self):
        """Test PYMBBO model compile, fit training, and token scaling benchmark."""
        x = np.random.randint(0, self.vocab_size, (20, 16)).astype(np.int64)
        y = np.random.randint(0, self.vocab_size, (20, 16)).astype(np.int64)
        ds = Dataset(x, y)
        
        model = build_model("asa_transformer", vocab_size=self.vocab_size, d_model=32, nhead=2, num_layers=2)
        model.compile(optimizer="adam", loss_function="cross_entropy")
        
        history = model.fit(ds, epochs=2, batch_size=4)
        self.assertIn("loss", history)
        self.assertEqual(len(history["loss"]), 2)
        
        # Token scaling benchmark
        report = token_scaling_benchmark(
            model=model,
            vocab_size=self.vocab_size,
            min_tokens=5,
            max_tokens=15,
            steps=2
        )
        self.assertEqual(len(report["token_counts"]), 2)


if __name__ == "__main__":
    unittest.main()
