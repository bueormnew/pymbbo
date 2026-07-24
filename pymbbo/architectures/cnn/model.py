import torch
import torch.nn as nn
from pymbbo.architectures.base_arch import BaseArchitecture
from pymbbo.models.registry import register_architecture

@register_architecture("cnn")
class CNNArchitecture(BaseArchitecture):
    """
    Convolutional Neural Network (CNN) Architecture Plugin.
    """
    ARCH_NAME = "cnn"

    def __init__(self, in_channels: int = 1, num_classes: int = 10, **kwargs):
        super().__init__(in_channels=in_channels, num_classes=num_classes, **kwargs)
        
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        return self.classifier(features)
