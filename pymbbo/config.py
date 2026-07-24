import json
import os
from typing import Any, Dict, Union

class Hyperparameters:
    """
    Central object to manage experiment hyperparameter configurations cleanly.
    Allows attribute access, dictionary access, serialization to JSON, and loading.
    """
    def __init__(self,
                 learning_rate: float = 1e-3,
                 batch_size: int = 32,
                 epochs: int = 10,
                 optimizer: str = "adam",
                 loss_function: str = "cross_entropy",
                 seed: int = 42,
                 **kwargs):
        self._config: Dict[str, Any] = {
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "epochs": epochs,
            "optimizer": optimizer,
            "loss_function": loss_function,
            "seed": seed
        }
        self._config.update(kwargs)

    def __getattr__(self, name: str) -> Any:
        if name in self._config:
            return self._config[name]
        raise AttributeError(f"'Hyperparameters' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_config":
            super().__setattr__(name, value)
        else:
            self._config[name] = value

    def __getitem__(self, key: str) -> Any:
        return self._config[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._config[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._config

    def __repr__(self) -> str:
        formatted = json.dumps(self._config, indent=2)
        return f"Hyperparameters(\n{formatted}\n)"

    def update(self, **kwargs) -> None:
        """Updates hyperparameters with key-value arguments."""
        self._config.update(kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """Returns the dictionary representation of hyperparameters."""
        return dict(self._config)

    def save(self, filepath: str) -> str:
        """
        Saves hyperparameter configuration to a JSON file.
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self._config, f, indent=2)
        return filepath


    @classmethod
    def load(cls, filepath: str) -> "Hyperparameters":
        """
        Loads a hyperparameter configuration from a JSON file.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        instance = cls()
        instance._config = data
        return instance

# Alias for Config
Config = Hyperparameters
