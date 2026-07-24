from typing import Optional
from pymbbo.models.base import BaseModel
from pymbbo.models.registry import ARCHITECTURE_REGISTRY, discover_architectures

def build_model(architecture_type: str = "sequential", **kwargs) -> BaseModel:
    """
    Factory function to instantiate neural network models in PYMBBO.
    Usage:
        model = build_model("sequential")
        model = build_model("mlp", input_dim=10, hidden_units=[64, 32], output_dim=1)
        model = build_model("cnn", in_channels=1, num_classes=10)
        model = build_model("transformer", vocab_size=5000, d_model=256)
    """
    arch_type_lower = architecture_type.lower()
    
    if arch_type_lower == "sequential":
        return BaseModel()

    # Discover built-in and custom architectures
    registry = discover_architectures()

    if arch_type_lower in registry:
        arch_cls = registry[arch_type_lower]
        arch_instance = arch_cls(**kwargs)
        return BaseModel(architecture=arch_instance)

    raise ValueError(
        f"Architecture '{architecture_type}' is not registered in PYMBBO.\n"
        f"Available architectures: {list(registry.keys()) + ['sequential']}"
    )
