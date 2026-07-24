import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Union

def accuracy_metric(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """Calculates classification accuracy metric."""
    with torch.no_grad():
        if y_pred.dim() > 1 and y_pred.size(-1) > 1:
            preds = torch.argmax(y_pred, dim=-1)
        else:
            preds = (torch.sigmoid(y_pred) > 0.5).long().squeeze()
        
        targets = y_true.long().squeeze()
        correct = (preds == targets).float().sum()
        return (correct / targets.numel()).item()


def mae_metric(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """Calculates Mean Absolute Error metric."""
    with torch.no_grad():
        return F.l1_loss(y_pred.squeeze(), y_true.squeeze()).item()


def mse_metric(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """Calculates Mean Squared Error metric."""
    with torch.no_grad():
        return F.mse_loss(y_pred.squeeze(), y_true.squeeze()).item()


def f1_score_metric(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """Calculates binary F1-score metric."""
    with torch.no_grad():
        preds = (torch.sigmoid(y_pred) > 0.5).long().squeeze() if y_pred.size(-1) == 1 else torch.argmax(y_pred, dim=-1)
        targets = y_true.long().squeeze()
        
        tp = ((preds == 1) & (targets == 1)).float().sum().item()
        fp = ((preds == 1) & (targets == 0)).float().sum().item()
        fn = ((preds == 0) & (targets == 1)).float().sum().item()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        return float(f1)


def perplexity_metric(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """Calculates language model Perplexity metric."""
    with torch.no_grad():
        loss = F.cross_entropy(y_pred.view(-1, y_pred.size(-1)), y_true.view(-1).long())
        return torch.exp(loss).item()


def get_loss_function(loss_name: str) -> Callable:
    lname = loss_name.lower()
    if lname in ("cross_entropy", "crossentropy", "ce"):
        def ce_loss(preds, targets):
            if preds.dim() == 3:
                return F.cross_entropy(preds.view(-1, preds.size(-1)), targets.view(-1).long())
            return F.cross_entropy(preds, targets.long().squeeze())
        return ce_loss
    elif lname in ("mse", "mean_squared_error"):
        return lambda preds, targets: F.mse_loss(preds.squeeze(), targets.float().squeeze())
    elif lname in ("mae", "mean_absolute_error"):
        return lambda preds, targets: F.l1_loss(preds.squeeze(), targets.float().squeeze())
    elif lname in ("bce", "binary_cross_entropy"):
        return lambda preds, targets: F.binary_cross_entropy_with_logits(preds.squeeze(), targets.float().squeeze())
    else:
        raise ValueError(f"Unknown loss function name: '{loss_name}'")



def get_metric_function(metric_name: str) -> Callable:
    mname = metric_name.lower()
    if mname in ("accuracy", "acc"):
        return accuracy_metric
    elif mname == "mae":
        return mae_metric
    elif mname == "mse":
        return mse_metric
    elif mname in ("f1", "f1_score"):
        return f1_score_metric
    elif mname in ("ppl", "perplexity"):
        return perplexity_metric
    else:
        raise ValueError(f"Unknown metric name: '{metric_name}'")
