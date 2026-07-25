import torch
import torch.nn as nn
from typing import List, Union, Optional, Dict, Any, Callable
from pymbbo.models.layers import Dense, Conv2D, Dropout, BatchNorm, Flatten

class BaseModel(nn.Module):
    """
    Main PYMBBO Model client wrapper managing network layers, parameter freezing,
    architectural summaries, compilation, and high-level training lifecycle.
    """
    def __init__(self, architecture: Optional[nn.Module] = None):
        super().__init__()
        self.architecture = architecture
        self.sequential_layers = nn.ModuleList() if architecture is None else None
        
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.loss_fn: Optional[Callable] = None
        self.metrics_fn: Dict[str, Callable] = {}
        self.history: Dict[str, List[float]] = {"loss": [], "val_loss": []}
        self.is_compiled: bool = False

    def add_layer(self, layer_or_type: Union[nn.Module, str], **kwargs) -> "BaseModel":
        """
        Adds a layer to sequential architectures.
        Usage:
            model.add_layer("dense", units=64, activation="relu")
            model.add_layer(Dense(128, activation="relu"))
        """
        if self.architecture is not None:
            raise RuntimeError("Cannot use add_layer on a pre-built architecture model.")

        if isinstance(layer_or_type, nn.Module):
            self.sequential_layers.append(layer_or_type)
        elif isinstance(layer_or_type, str):
            ltype = layer_or_type.lower()
            if ltype == "dense":
                units = kwargs.get("units", kwargs.get("out_features", 64))
                activation = kwargs.get("activation", None)
                in_dim = kwargs.get("in_features", None)
                self.sequential_layers.append(Dense(in_features=in_dim, out_features=units, activation=activation))
            elif ltype in ("conv2d", "conv"):
                out_ch = kwargs.get("units", kwargs.get("out_channels", 32))
                k_size = kwargs.get("kernel_size", 3)
                act = kwargs.get("activation", "relu")
                self.sequential_layers.append(Conv2D(out_channels=out_ch, kernel_size=k_size, activation=act))
            elif ltype == "dropout":
                rate = kwargs.get("rate", 0.5)
                self.sequential_layers.append(Dropout(rate=rate))
            elif ltype in ("batchnorm", "bn"):
                self.sequential_layers.append(BatchNorm())
            elif ltype == "flatten":
                self.sequential_layers.append(Flatten())
            else:
                raise ValueError(f"Unknown layer type string: '{layer_or_type}'")
        else:
            raise TypeError(f"Invalid layer specification: {type(layer_or_type)}")

        return self

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.architecture is not None:
            return self.architecture(x, **kwargs)
        
        out = x
        for layer in self.sequential_layers:
            out = layer(out)
        return out

    def summary(self, input_shape: Optional[Tuple[int, ...]] = None) -> None:
        """
        Displays a visual architectural breakdown: layer names, parameter counts, and shapes.
        """
        print("=" * 70)
        print(f"{'PYMBBO Model Architectural Summary':^70}")
        print("=" * 70)
        print(f"{'Layer (Type)':<30} | {'Output Shape':<20} | {'Param #':<12}")
        print("-" * 70)

        total_params = 0
        trainable_params = 0
        frozen_params = 0

        modules_to_inspect = []
        if self.architecture is not None:
            modules_to_inspect = [(name, mod) for name, mod in self.architecture.named_children()]
            if not modules_to_inspect:
                modules_to_inspect = [("architecture", self.architecture)]
        else:
            modules_to_inspect = [(f"layer_{i}", layer) for i, layer in enumerate(self.sequential_layers)]

        for name, mod in modules_to_inspect:
            mod_params = sum(p.numel() for p in mod.parameters())
            mod_trainable = sum(p.numel() for p in mod.parameters() if p.requires_grad)
            mod_frozen = mod_params - mod_trainable

            total_params += mod_params
            trainable_params += mod_trainable
            frozen_params += mod_frozen

            layer_class = mod.__class__.__name__
            status_str = "" if mod_frozen == 0 else f" (Frozen: {mod_frozen})"
            shape_str = f"Dynamic{status_str}"

            print(f"{f'{name} ({layer_class})':<30} | {shape_str:<20} | {mod_params:<12,}")

        print("=" * 70)
        print(f"Total Parameters:      {total_params:,}")
        print(f"Trainable Parameters:  {trainable_params:,}")
        print(f"Frozen Parameters:     {frozen_params:,}")
        print("=" * 70)

    def freeze_layers(self, layer_indices_or_names: Optional[Union[List[int], List[str], str]] = "all") -> None:
        """
        Freezes selected layers or all parameters for fine-tuning & transfer learning.
        """
        if layer_indices_or_names == "all":
            for param in self.parameters():
                param.requires_grad = False
            return

        if self.architecture is None and self.sequential_layers is not None:
            if isinstance(layer_indices_or_names, list):
                for idx in layer_indices_or_names:
                    if isinstance(idx, int) and 0 <= idx < len(self.sequential_layers):
                        for p in self.sequential_layers[idx].parameters():
                            p.requires_grad = False
        else:
            for name, param in self.named_parameters():
                if isinstance(layer_indices_or_names, list) and any(str(item) in name for item in layer_indices_or_names):
                    param.requires_grad = False

    def unfreeze(self, layer_indices_or_names: Optional[Union[List[int], List[str], str]] = "all") -> None:
        """
        Unfreezes selected layers or all parameters.
        """
        if layer_indices_or_names == "all":
            for param in self.parameters():
                param.requires_grad = True
            return

        if self.architecture is None and self.sequential_layers is not None:
            if isinstance(layer_indices_or_names, list):
                for idx in layer_indices_or_names:
                    if isinstance(idx, int) and 0 <= idx < len(self.sequential_layers):
                        for p in self.sequential_layers[idx].parameters():
                            p.requires_grad = True
        else:
            for name, param in self.named_parameters():
                if isinstance(layer_indices_or_names, list) and any(str(item) in name for item in layer_indices_or_names):
                    param.requires_grad = True

    def compile(self, optimizer: Union[str, torch.optim.Optimizer] = "adam",
                loss_function: Union[str, Callable] = "cross_entropy",
                metrics: Optional[List[Union[str, Callable]]] = None,
                learning_rate: float = 1e-3) -> "BaseModel":
        """
        Associates model architecture with an optimizer, loss function, and evaluation metrics.
        """
        from pymbbo.metrics.standard import get_loss_function, get_metric_function

        # 1. Setup Optimizer
        if isinstance(optimizer, str):
            opt_lower = optimizer.lower()
            params = [p for p in self.parameters() if p.requires_grad]
            if opt_lower == "adam":
                self.optimizer = torch.optim.Adam(params, lr=learning_rate)
            elif opt_lower == "sgd":
                self.optimizer = torch.optim.SGD(params, lr=learning_rate)
            elif opt_lower == "adamw":
                self.optimizer = torch.optim.AdamW(params, lr=learning_rate)
            else:
                raise ValueError(f"Unsupported optimizer string: '{optimizer}'")
        else:
            self.optimizer = optimizer

        # 2. Setup Loss
        if isinstance(loss_function, str):
            self.loss_fn = get_loss_function(loss_function)
        else:
            self.loss_fn = loss_function

        # 3. Setup Metrics
        self.metrics_fn = {}
        if metrics:
            for m in metrics:
                if isinstance(m, str):
                    self.metrics_fn[m] = get_metric_function(m)
                elif callable(m):
                    self.metrics_fn[m.__name__] = m

        self.is_compiled = True
        return self

    def fit(self, train_data, validation_data=None, epochs=10, batch_size=32, callbacks=None, device="cpu"):
        from pymbbo.engine.trainer import fit
        return fit(self, train_data=train_data, validation_data=validation_data, epochs=epochs, batch_size=batch_size, callbacks=callbacks, device=device)

    def evaluate(self, test_data, batch_size=32, device="cpu"):
        from pymbbo.engine.trainer import evaluate
        return evaluate(self, test_data=test_data, batch_size=batch_size, device=device)

    def predict(self, input_data, batch_size=32, device="cpu"):
        from pymbbo.engine.trainer import predict
        return predict(self, input_data=input_data, batch_size=batch_size, device=device)

    def get_metrics(self):
        from pymbbo.engine.trainer import get_metrics
        return get_metrics(self)

    def save(self, filepath: str) -> str:
        from pymbbo.models.persistence import save_model
        return save_model(self, filepath)

    def export(self, filepath: str, format: str = "onnx", dummy_input=None) -> str:
        from pymbbo.models.persistence import export_model
        return export_model(self, filepath=filepath, format=format, dummy_input=dummy_input)

    def __call__(self, *args, **kwargs):

        if len(args) == 1 and not self.training and isinstance(args[0], (torch.Tensor, list)):
            if not isinstance(args[0], torch.Tensor):
                t_arg = torch.tensor(args[0], dtype=torch.float32)
                return super().__call__(t_arg, **kwargs)
        return super().__call__(*args, **kwargs)

