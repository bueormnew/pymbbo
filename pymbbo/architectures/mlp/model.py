import torch
import torch.nn as nn
from pymbbo.architectures.base_arch import BaseArchitecture
from pymbbo.models.registry import register_architecture

@register_architecture("mlp")
class MLPArchitecture(BaseArchitecture):
    """
    Multi-Layer Perceptron (MLP) Architecture Plugin.
    """
    ARCH_NAME = "mlp"

    def __init__(self, input_dim: int = 10, hidden_units: list = [64, 32], output_dim: int = 1, activation: str = "relu", **kwargs):
        super().__init__(input_dim=input_dim, hidden_units=hidden_units, output_dim=output_dim, activation=activation, **kwargs)
        
        act_fn = nn.ReLU() if activation.lower() == "relu" else nn.Tanh()
        layers = []
        in_dim = input_dim

        for h in hidden_units:
            layers.append(nn.Linear(in_dim, h))
            layers.append(act_fn)
            in_dim = h

        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
