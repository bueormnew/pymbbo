import time
import torch
from typing import Union, Optional, List, Dict, Any, Tuple
from pymbbo.data.dataset import Dataset, load_dataset
from pymbbo.data.batcher import Batcher
from pymbbo.engine.callbacks import Callback

def fit(
    self,
    train_data: Union[Dataset, Tuple[Any, Any]],
    validation_data: Optional[Union[Dataset, Tuple[Any, Any]]] = None,
    epochs: int = 10,
    batch_size: int = 32,
    callbacks: Optional[List[Callback]] = None,
    device: str = "cpu"
) -> Dict[str, List[float]]:
    """
    Executes model training from start to finish.
    """
    if not self.is_compiled:
        raise RuntimeError("Model is not compiled. Call model.compile(optimizer, loss_function, metrics) before training.")

    self.to(device)
    callbacks = callbacks or []

    # Ensure train_data is Dataset instance
    if not isinstance(train_data, Dataset):
        train_ds = load_dataset(train_data)
    else:
        train_ds = train_data

    val_ds = None
    if validation_data is not None:
        if not isinstance(validation_data, Dataset):
            val_ds = load_dataset(validation_data)
        else:
            val_ds = validation_data

    train_batcher = Batcher(train_ds, batch_size=batch_size, shuffle=True)

    print("\n" + "=" * 70)
    print(f"{'PYMBBO Training Engine Initialized':^70}")
    print("=" * 70)

    for cb in callbacks:
        cb.on_train_begin()

    for epoch in range(epochs):
        self.train()
        total_loss = 0.0
        batch_count = 0
        epoch_start = time.perf_counter()

        for cb in callbacks:
            cb.on_epoch_begin(epoch)

        for b_idx, (x_batch, y_batch) in enumerate(train_batcher):
            x_batch = x_batch.to(device)
            if y_batch is not None:
                y_batch = y_batch.to(device)

            for cb in callbacks:
                cb.on_batch_begin(b_idx)

            self.optimizer.zero_grad()
            preds = self(x_batch)
            loss = self.loss_fn(preds, y_batch)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            batch_count += 1

            for cb in callbacks:
                cb.on_batch_end(b_idx)

        avg_train_loss = total_loss / max(batch_count, 1)
        self.history["loss"].append(avg_train_loss)

        # Validation phase
        avg_val_loss = None
        val_metrics_results = {}
        if val_ds is not None:
            val_res = evaluate(self, val_ds, batch_size=batch_size, device=device)
            avg_val_loss = val_res.get("loss")
            self.history["val_loss"].append(avg_val_loss)
            val_metrics_results = {k: v for k, v in val_res.items() if k != "loss"}

        epoch_time = time.perf_counter() - epoch_start
        val_str = f" - val_loss: {avg_val_loss:.4f}" if avg_val_loss is not None else ""
        metrics_str = "".join([f" - {k}: {v:.4f}" for k, v in val_metrics_results.items()])

        print(f"Epoch {epoch+1:02d}/{epochs:02d} [{epoch_time:.2f}s] - loss: {avg_train_loss:.4f}{val_str}{metrics_str}")

        # Callback logs
        epoch_logs = {
            "epoch": epoch,
            "loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "model": self
        }
        epoch_logs.update(val_metrics_results)

        stop_requested = False
        for cb in callbacks:
            if cb.on_epoch_end(epoch, logs=epoch_logs):
                stop_requested = True

        if stop_requested:
            print(f"Training stopped early by callback at epoch {epoch+1}.")
            break

    for cb in callbacks:
        cb.on_train_end()

    print("=" * 70)
    return self.history


def evaluate(
    self,
    test_data: Union[Dataset, Tuple[Any, Any]],
    batch_size: int = 32,
    device: str = "cpu"
) -> Dict[str, float]:
    """
    Evaluates model performance on unseen test data.
    """
    if not isinstance(test_data, Dataset):
        test_ds = load_dataset(test_data)
    else:
        test_ds = test_data

    batcher = Batcher(test_ds, batch_size=batch_size, shuffle=False)
    self.eval()
    self.to(device)

    total_loss = 0.0
    batch_count = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x_batch, y_batch in batcher:
            x_batch = x_batch.to(device)
            if y_batch is not None:
                y_batch = y_batch.to(device)

            preds = self(x_batch)
            if self.loss_fn is not None and y_batch is not None:
                loss = self.loss_fn(preds, y_batch)
                total_loss += loss.item()

            all_preds.append(preds.cpu())
            if y_batch is not None:
                all_targets.append(y_batch.cpu())
            batch_count += 1

    results = {}
    if batch_count > 0 and self.loss_fn is not None:
        results["loss"] = total_loss / batch_count

    if all_preds and all_targets:
        cat_preds = torch.cat(all_preds, dim=0)
        cat_targets = torch.cat(all_targets, dim=0)

        for name, metric_fn in self.metrics_fn.items():
            results[name] = metric_fn(cat_preds, cat_targets)

    return results


def predict(
    self,
    input_data: Union[torch.Tensor, Any],
    batch_size: int = 32,
    device: str = "cpu"
) -> torch.Tensor:
    """
    Generates predictions / forward pass outputs for new inputs.
    """
    self.eval()
    self.to(device)

    if isinstance(input_data, torch.Tensor):
        with torch.no_grad():
            return self(input_data.to(device)).cpu()

    if isinstance(input_data, Dataset):
        ds = input_data
    else:
        ds = load_dataset(input_data)

    batcher = Batcher(ds, batch_size=batch_size, shuffle=False)
    preds_list = []
    with torch.no_grad():
        for x_batch, _ in batcher:
            preds = self(x_batch.to(device))
            preds_list.append(preds.cpu())

    return torch.cat(preds_list, dim=0)


def get_metrics(self) -> Dict[str, List[float]]:
    """
    Returns full history of training and validation loss and metrics.
    """
    return dict(self.history)
