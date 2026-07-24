import torch
import torch.nn as nn
from typing import Optional, Union, Tuple

class Dense(nn.Module):
    """
    Fully Connected Dense Layer wrapper.
    """
    def __init__(self, in_features: Optional[int] = None, out_features: int = 64, activation: Optional[str] = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation_name = activation
        self.linear = nn.Linear(in_features, out_features) if in_features is not None else None
        
        self.act_fn = None
        if activation:
            act_str = activation.lower()
            if act_str == "relu":
                self.act_fn = nn.ReLU()
            elif act_str == "sigmoid":
                self.act_fn = nn.Sigmoid()
            elif act_str == "tanh":
                self.act_fn = nn.Tanh()
            elif act_str in ("softmax", "log_softmax"):
                self.act_fn = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.linear is None:
            self.in_features = x.shape[-1]
            self.linear = nn.Linear(self.in_features, self.out_features).to(x.device)
        out = self.linear(x)
        if self.act_fn is not None:
            out = self.act_fn(out)
        return out


class Conv2D(nn.Module):
    """
    2D Convolutional Layer wrapper.
    """
    def __init__(self, in_channels: Optional[int] = None, out_channels: int = 32, 
                 kernel_size: Union[int, Tuple[int, int]] = 3, stride: int = 1, padding: int = 0,
                 activation: Optional[str] = "relu"):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding) if in_channels is not None else None
        self.act_fn = nn.ReLU() if activation and activation.lower() == "relu" else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.conv is None:
            self.in_channels = x.shape[1]
            self.conv = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding).to(x.device)
        out = self.conv(x)
        if self.act_fn is not None:
            out = self.act_fn(out)
        return out


class Dropout(nn.Module):
    """
    Dropout Regularization Layer.
    """
    def __init__(self, rate: float = 0.5):
        super().__init__()
        self.dropout = nn.Dropout(p=rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x)


class BatchNorm(nn.Module):
    """
    Batch Normalization Layer wrapper.
    """
    def __init__(self, num_features: Optional[int] = None, dim: int = 1):
        super().__init__()
        self.num_features = num_features
        self.dim = dim
        self.bn = nn.BatchNorm1d(num_features) if (num_features and dim == 1) else (nn.BatchNorm2d(num_features) if num_features else None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bn is None:
            self.num_features = x.shape[1]
            self.bn = nn.BatchNorm1d(self.num_features).to(x.device) if x.dim() <= 2 else nn.BatchNorm2d(self.num_features).to(x.device)
        return self.bn(x)


class Flatten(nn.Module):
    """
    Flattens spatial dimensions into vector features.
    """
    def __init__(self, start_dim: int = 1):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=start_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.flatten(x)
