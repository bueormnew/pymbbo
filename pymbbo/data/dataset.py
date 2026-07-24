import os
import csv
import math
import random
from typing import Any, Callable, List, Optional, Tuple, Union, Dict
import numpy as np
import torch

class Dataset:
    """
    Core PYMBBO Dataset class managing inputs and targets.
    Supports streaming transformations and automatic dataset splitting.
    """
    def __init__(self, x: Union[np.ndarray, torch.Tensor, List[Any]], 
                 y: Optional[Union[np.ndarray, torch.Tensor, List[Any]]] = None,
                 transforms: Optional[List[Callable]] = None):
        self.x = self._to_tensor(x)
        self.y = self._to_tensor(y) if y is not None else None
        self.transforms = transforms or []

    def _to_tensor(self, data: Any) -> torch.Tensor:
        if isinstance(data, torch.Tensor):
            return data
        if isinstance(data, np.ndarray):
            if np.issubdtype(data.dtype, np.integer):
                return torch.from_numpy(data).long()
            return torch.from_numpy(data).float()
        if isinstance(data, (list, tuple)):
            if len(data) > 0 and isinstance(data[0], int):
                return torch.tensor(data, dtype=torch.long)
            return torch.tensor(data, dtype=torch.float32)
        raise TypeError(f"Unsupported data type for Dataset: {type(data)}")


    def transform(self, preprocess_fn: Callable) -> "Dataset":
        """
        Applies a transformation function to dataset inputs/targets in pipeline fashion.
        """
        self.transforms.append(preprocess_fn)
        return self

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x_val = self.x[idx]
        y_val = self.y[idx] if self.y is not None else None

        for fn in self.transforms:
            if y_val is not None:
                res = fn(x_val, y_val)
                if isinstance(res, tuple) and len(res) == 2:
                    x_val, y_val = res
                else:
                    x_val = res
            else:
                x_val = fn(x_val)

        return x_val, y_val

    def split(self, train: float = 0.8, val: float = 0.1, test: float = 0.1, 
              shuffle: bool = True, seed: int = 42) -> Tuple["Dataset", "Dataset", "Dataset"]:
        """
        Automatically splits the dataset into training, validation, and test sets.
        """
        total_ratio = train + val + test
        if not math.isclose(total_ratio, 1.0, rel_tol=1e-3):
            raise ValueError(f"Train ({train}), val ({val}), and test ({test}) ratios must sum to 1.0")

        n = len(self)
        indices = list(range(n))
        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(indices)

        train_end = int(n * train)
        val_end = train_end + int(n * val)

        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]

        train_x = self.x[train_idx]
        train_y = self.y[train_idx] if self.y is not None else None

        val_x = self.x[val_idx]
        val_y = self.y[val_idx] if self.y is not None else None

        test_x = self.x[test_idx]
        test_y = self.y[test_idx] if self.y is not None else None

        return (
            Dataset(train_x, train_y, transforms=list(self.transforms)),
            Dataset(val_x, val_y, transforms=list(self.transforms)),
            Dataset(test_x, test_y, transforms=list(self.transforms))
        )


def load_dataset(source: Union[str, np.ndarray, Dict[str, Any], Tuple[Any, Any]], 
                 target_column: Optional[Union[str, int]] = None,
                 **kwargs) -> Dataset:
    """
    Loads data from various sources (CSV, dictionary, tuple of (x,y), numpy arrays, or file paths).
    """
    # 1. Tuple or list of (x, y)
    if isinstance(source, (tuple, list)) and len(source) == 2 and not isinstance(source[0], str):
        return Dataset(source[0], source[1])

    # 2. NumPy array or torch Tensor
    if isinstance(source, (np.ndarray, torch.Tensor)):
        return Dataset(source)

    # 3. Dictionary source: {"x": ..., "y": ...}
    if isinstance(source, dict):
        x = source.get("x") or source.get("inputs") or source.get("data")
        y = source.get("y") or source.get("targets") or source.get("labels")
        if x is None:
            raise KeyError("Dictionary source must contain key 'x', 'inputs', or 'data'.")
        return Dataset(x, y)

    # 4. File source (CSV)
    if isinstance(source, str) and source.endswith(".csv"):
        if not os.path.exists(source):
            raise FileNotFoundError(f"CSV file not found: {source}")
        
        with open(source, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = list(reader)

        data_np = np.array(rows, dtype=np.float32)
        if target_column is not None:
            if isinstance(target_column, str):
                col_idx = header.index(target_column)
            else:
                col_idx = int(target_column)
            
            x_np = np.delete(data_np, col_idx, axis=1)
            y_np = data_np[:, col_idx]
            return Dataset(x_np, y_np)
        else:
            # Assume last column is target if multi-column
            if data_np.shape[1] > 1:
                return Dataset(data_np[:, :-1], data_np[:, -1])
            return Dataset(data_np)

    raise ValueError(f"Unsupported data source type for load_dataset: {type(source)}")
