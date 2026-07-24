import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseArchitecture(nn.Module, ABC):
    """
    Abstract Base Class for all PYMBBO architecture plugins.
    Any custom neural network placed inside pymbbo/architectures/<subfolder>/ 
    should inherit from BaseArchitecture.
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.config_kwargs = kwargs

    @abstractmethod
    def forward(self, x):
        """Forward pass computation."""
        pass

    def get_config(self) -> Dict[str, Any]:
        """Returns architecture initialization parameters."""
        return self.config_kwargs
