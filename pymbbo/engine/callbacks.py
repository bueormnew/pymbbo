import os
import torch
from typing import Dict, Any, Optional

class Callback:
    """
    Base Callback class for PYMBBO training lifecycle events.
    """
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        pass

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        pass

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        pass

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> bool:
        """Return True to request early stopping."""
        return False

    def on_batch_begin(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        pass

    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        pass


class EarlyStopping(Callback):
    """
    Stops training if validation loss or specified metric stops improving.
    """
    def __init__(self, patience: int = 3, monitor: str = "val_loss", min_delta: float = 1e-4, mode: str = "min"):
        super().__init__()
        self.patience = patience
        self.monitor = monitor
        self.min_delta = min_delta
        self.mode = mode
        self.best_score: Optional[float] = None
        self.wait: int = 0

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> bool:
        logs = logs or {}
        current_score = logs.get(self.monitor)
        if current_score is None:
            return False

        if self.best_score is None:
            self.best_score = current_score
            return False

        is_better = (current_score < self.best_score - self.min_delta) if self.mode == "min" else (current_score > self.best_score + self.min_delta)

        if is_better:
            self.best_score = current_score
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print(f"\n[EarlyStopping] Triggered at epoch {epoch+1}. Metric '{self.monitor}' did not improve for {self.patience} epochs.")
                return True
        return False


class ModelCheckpoint(Callback):
    """
    Saves automatically the best model version during training.
    """
    def __init__(self, filepath: str = "best_model.mbbo", monitor: str = "val_loss", save_best_only: bool = True, mode: str = "min"):
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.mode = mode
        self.best_score: Optional[float] = None

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> bool:
        logs = logs or {}
        model = logs.get("model")
        current_score = logs.get(self.monitor)

        if model is None:
            return False

        if not self.save_best_only:
            model.save(f"epoch_{epoch+1}_{self.filepath}")
            return False

        if current_score is not None:
            is_better = (self.best_score is None) or ((current_score < self.best_score) if self.mode == "min" else (current_score > self.best_score))
            if is_better:
                self.best_score = current_score
                model.save(self.filepath)
                print(f"\n[ModelCheckpoint] Model saved to '{self.filepath}' ({self.monitor}: {current_score:.4f})")

        return False


class LRScheduler(Callback):
    """
    Adjusts learning rate dynamically during training.
    """
    def __init__(self, factor: float = 0.5, patience: int = 2, monitor: str = "val_loss", min_lr: float = 1e-6):
        super().__init__()
        self.factor = factor
        self.patience = patience
        self.monitor = monitor
        self.min_lr = min_lr
        self.best_score: Optional[float] = None
        self.wait: int = 0

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> bool:
        logs = logs or {}
        model = logs.get("model")
        current_score = logs.get(self.monitor)

        if model is None or model.optimizer is None or current_score is None:
            return False

        if self.best_score is None:
            self.best_score = current_score
            return False

        if current_score < self.best_score:
            self.best_score = current_score
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                for param_group in model.optimizer.param_groups:
                    old_lr = param_group["lr"]
                    new_lr = max(old_lr * self.factor, self.min_lr)
                    param_group["lr"] = new_lr
                    print(f"\n[LRScheduler] Reduced learning rate from {old_lr:.6f} to {new_lr:.6f}")
                self.wait = 0

        return False
