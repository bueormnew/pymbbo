import os
import shutil
import tempfile
import unittest
import torch
import numpy as np

from pymbbo import (
    Config, Hyperparameters, Dataset, load_dataset, Batcher,
    build_model, BaseModel, register_architecture, BaseArchitecture,
    load_model, EarlyStopping, ModelCheckpoint, LRScheduler,
    token_scaling_benchmark, compare_models, discover_architectures
)

class TestPYMBBOFramework(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_01_hyperparameters(self):
        """Test hyperparameter creation, attribute/dict access, and JSON save/load."""
        config = Config(learning_rate=0.005, batch_size=16, epochs=5, custom_param="test")
        self.assertEqual(config.learning_rate, 0.005)
        self.assertEqual(config["batch_size"], 16)
        
        json_path = os.path.join(self.test_dir, "config.json")
        config.save(json_path)
        self.assertTrue(os.path.exists(json_path))

        loaded_config = Config.load(json_path)
        self.assertEqual(loaded_config.learning_rate, 0.005)
        self.assertEqual(loaded_config.custom_param, "test")

    def test_02_data_ingestion_split_transform(self):
        """Test Dataset, load_dataset, split, transform, and Batcher."""
        x = np.random.randn(100, 10).astype(np.float32)
        y = np.random.randint(0, 2, (100, 1)).astype(np.float32)
        
        ds = load_dataset((x, y))
        self.assertEqual(len(ds), 100)

        # Transformation
        ds.transform(lambda a, b: (a * 2.0, b))
        sample_x, _ = ds[0]
        self.assertAlmostEqual(sample_x[0].item(), x[0][0] * 2.0, places=4)

        # Split
        train_ds, val_ds, test_ds = ds.split(train=0.8, val=0.1, test=0.1)
        self.assertEqual(len(train_ds), 80)
        self.assertEqual(len(val_ds), 10)
        self.assertEqual(len(test_ds), 10)

        # Batcher iteration
        batcher = Batcher(train_ds, batch_size=16)
        batches = list(batcher)
        self.assertEqual(len(batches), 5)
        bx, by = batches[0]
        self.assertEqual(bx.shape, (16, 10))

    def test_03_model_sequential_summary_freeze(self):
        """Test sequential model assembly, summary printout, and layer freezing/unfreezing."""
        model = build_model("sequential")
        model.add_layer("dense", units=32, activation="relu", in_features=10)
        model.add_layer("dropout", rate=0.2)
        model.add_layer("dense", units=1, activation="sigmoid")

        model.summary()

        # Freeze all layers
        model.freeze_layers("all")
        trainable_after_freeze = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.assertEqual(trainable_after_freeze, 0)

        # Unfreeze all layers
        model.unfreeze("all")
        trainable_after_unfreeze = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.assertGreater(trainable_after_unfreeze, 0)

    def test_04_architecture_plugin_discovery(self):
        """Test built-in architecture plugins (mlp, cnn, transformer) and dynamic user plugins."""
        mlp_model = build_model("mlp", input_dim=10, hidden_units=[32, 16], output_dim=1)
        self.assertIsNotNone(mlp_model)

        cnn_model = build_model("cnn", in_channels=1, num_classes=10)
        self.assertIsNotNone(cnn_model)

        trans_model = build_model("transformer", vocab_size=100, d_model=32)
        self.assertIsNotNone(trans_model)

        # Test creating a dynamic user architecture folder inside test_dir
        plugin_dir = os.path.join(self.test_dir, "custom_plugin")
        os.makedirs(plugin_dir, exist_ok=True)
        plugin_file = os.path.join(plugin_dir, "model.py")
        with open(plugin_file, "w") as f:
            f.write('''
import torch
import torch.nn as nn
from pymbbo.architectures.base_arch import BaseArchitecture

class DynamicTestNet(BaseArchitecture):
    ARCH_NAME = "dynamic_test_net"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fc = nn.Linear(10, 2)
    def forward(self, x):
        return self.fc(x)
''')
        discover_architectures(self.test_dir)
        dynamic_model = build_model("dynamic_test_net")
        self.assertIsNotNone(dynamic_model)

    def test_05_training_evaluation_callbacks(self):
        """Test complete model compile, fit, callbacks, evaluate, predict, and get_metrics."""
        x = np.random.randn(80, 10).astype(np.float32)
        y = np.random.randint(0, 2, (80,)).astype(np.float32)
        train_ds = Dataset(x, y)

        vx = np.random.randn(20, 10).astype(np.float32)
        vy = np.random.randint(0, 2, (20,)).astype(np.float32)
        val_ds = Dataset(vx, vy)

        model = build_model("mlp", input_dim=10, hidden_units=[16], output_dim=1)
        model.compile(optimizer="adam", loss_function="bce", metrics=["accuracy"])

        cb_early = EarlyStopping(patience=2)
        cb_ckpt = ModelCheckpoint(filepath=os.path.join(self.test_dir, "checkpoint.mbbo"))
        cb_lr = LRScheduler(patience=1)

        history = model.fit(train_ds, validation_data=val_ds, epochs=3, batch_size=16, callbacks=[cb_early, cb_ckpt, cb_lr])
        self.assertIn("loss", history)
        self.assertEqual(len(history["loss"]), 3)

        # Evaluation & Inference
        eval_res = model.evaluate(val_ds)
        self.assertIn("loss", eval_res)

        preds = model.predict(torch.from_numpy(vx))
        self.assertEqual(preds.shape[0], 20)

        metrics_history = model.get_metrics()
        self.assertIn("val_loss", metrics_history)

    def test_06_token_scaling_benchmark_and_model_comparator(self):
        """Test specialized token scaling benchmarking and side-by-side model comparator."""
        model1 = build_model("transformer", vocab_size=100, d_model=32, max_seq_len=200)
        model2 = build_model("transformer", vocab_size=100, d_model=64, max_seq_len=200)

        # Token Scaling Benchmark
        token_report = token_scaling_benchmark(
            model=model1,
            vocab_size=100,
            min_tokens=10,
            max_tokens=50,
            steps=3
        )
        self.assertEqual(len(token_report["token_counts"]), 3)
        self.assertEqual(token_report["token_counts"][0], 10)
        self.assertEqual(token_report["token_counts"][-1], 50)

        # Model Comparison Engine
        x = np.random.randint(0, 100, (20, 10)).astype(np.int64)
        y = np.random.randint(0, 100, (20, 10)).astype(np.int64)
        test_ds = Dataset(x, y)

        model1.compile(optimizer="adam", loss_function="cross_entropy")
        model2.compile(optimizer="adam", loss_function="cross_entropy")

        comp_report = compare_models(
            models={"Transformer_Small": model1, "Transformer_Medium": model2},
            test_data=test_ds
        )
        self.assertIn("Transformer_Small", comp_report)
        self.assertIn("Transformer_Medium", comp_report)
        self.assertGreater(comp_report["Transformer_Medium"]["total_parameters"], comp_report["Transformer_Small"]["total_parameters"])

    def test_07_persistence_save_load_export(self):
        """Test model saving, loading static function, and ONNX/TorchScript export."""
        model = build_model("mlp", input_dim=10, hidden_units=[16], output_dim=1)
        model.compile(optimizer="adam", loss_function="mse")

        save_path = os.path.join(self.test_dir, "saved_model.mbbo")
        model.save(save_path)
        self.assertTrue(os.path.exists(save_path))

        restored_model = load_model(save_path)
        self.assertEqual(restored_model.is_compiled, True)

        export_path = os.path.join(self.test_dir, "exported_model.onnx")
        model.export(export_path, format="onnx", dummy_input=torch.randn(1, 10))
        self.assertTrue(os.path.exists(export_path))

if __name__ == "__main__":
    unittest.main()
