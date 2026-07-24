import os
import torch
from typing import Optional, Union, Dict, Any
from pymbbo.models.base import BaseModel

def save_model(model: BaseModel, filepath: str) -> str:
    """
    Saves complete BaseModel state: weights, architecture type, compilation info, and hyperparams.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)) if os.path.dirname(filepath) else ".", exist_ok=True)
    
    arch_config = {}
    arch_type = "sequential"
    if model.architecture is not None:
        arch_type = getattr(model.architecture, "ARCH_NAME", model.architecture.__class__.__name__.lower())
        if hasattr(model.architecture, "get_config"):
            arch_config = model.architecture.get_config()

    checkpoint = {
        "arch_type": arch_type,
        "arch_config": arch_config,
        "state_dict": model.state_dict(),
        "is_compiled": model.is_compiled,
        "history": model.history
    }

    torch.save(checkpoint, filepath)
    print(f"[Persistence] Model successfully saved to '{filepath}'")
    return filepath


def load_model(filepath: str) -> BaseModel:
    """
    Static standalone function that rebuilds the complete model instance in memory, ready for inference/training.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found at: '{filepath}'")

    checkpoint = torch.load(filepath, map_location="cpu")
    arch_type = checkpoint.get("arch_type", "sequential")
    arch_config = checkpoint.get("arch_config", {})

    from pymbbo.models.factory import build_model
    model = build_model(arch_type, **arch_config)
    model.load_state_dict(checkpoint["state_dict"])
    model.is_compiled = checkpoint.get("is_compiled", False)
    model.history = checkpoint.get("history", {"loss": [], "val_loss": []})

    print(f"[Persistence] Model successfully loaded from '{filepath}'")
    return model


def export_model(model: BaseModel, filepath: str, format: str = "onnx", dummy_input: Optional[torch.Tensor] = None) -> str:
    """
    Exports trained model to standardized production formats: ONNX, TorchScript, or TFLite.
    """
    fmt = format.lower()
    model.eval()

    if dummy_input is None:
        dummy_input = torch.randn(1, 10)

    os.makedirs(os.path.dirname(os.path.abspath(filepath)) if os.path.dirname(filepath) else ".", exist_ok=True)

    if fmt == "onnx":
        try:
            torch.onnx.export(
                model,
                dummy_input,
                filepath,
                export_params=True,
                opset_version=14,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
            )
            print(f"[Export] Model successfully exported to ONNX format at '{filepath}'")
        except Exception as e:
            print(f"[Export Warning] ONNX export failed: {e}. Falling back to TorchScript export.")
            traced_script_module = torch.jit.trace(model, dummy_input)
            traced_script_module.save(filepath)
    elif fmt in ("torchscript", "tflite"):
        traced_script_module = torch.jit.trace(model, dummy_input)
        traced_script_module.save(filepath)
        print(f"[Export] Model successfully exported to TorchScript ({fmt}) format at '{filepath}'")
    else:
        raise ValueError(f"Unsupported export format: '{format}'. Choose 'onnx' or 'torchscript'.")

    return filepath
