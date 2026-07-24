"""
PYMBBO: A Modular, Scalable AI Framework for PyPI
"""

__version__ = "0.1.1"

from pymbbo.config import Hyperparameters, Config
from pymbbo.data.dataset import Dataset, load_dataset
from pymbbo.data.batcher import DataCollator, Batcher
from pymbbo.models.base import BaseModel
from pymbbo.models.factory import build_model
from pymbbo.models.registry import register_architecture, discover_architectures, ARCHITECTURE_REGISTRY
from pymbbo.models.persistence import save_model, load_model, export_model
from pymbbo.models.layers import Dense, Conv2D, Dropout, BatchNorm, Flatten
from pymbbo.engine.callbacks import Callback, EarlyStopping, ModelCheckpoint, LRScheduler
from pymbbo.metrics.standard import accuracy_metric, mae_metric, mse_metric, f1_score_metric, perplexity_metric
from pymbbo.metrics.benchmark import token_scaling_benchmark, compare_models
from pymbbo.architectures.base_arch import BaseArchitecture

# Automatically discover architectures in pymbbo/architectures/
discover_architectures()

__all__ = [
    "Hyperparameters", "Config",
    "Dataset", "load_dataset", "DataCollator", "Batcher",
    "BaseModel", "build_model", "register_architecture", "discover_architectures", "ARCHITECTURE_REGISTRY",
    "save_model", "load_model", "export_model",
    "Dense", "Conv2D", "Dropout", "BatchNorm", "Flatten",
    "Callback", "EarlyStopping", "ModelCheckpoint", "LRScheduler",
    "accuracy_metric", "mae_metric", "mse_metric", "f1_score_metric", "perplexity_metric",
    "token_scaling_benchmark", "compare_models", "BaseArchitecture"
]
