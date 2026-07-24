from typing import List, Tuple, Optional, Any, Iterator
import torch
from torch.utils.data import DataLoader
from .dataset import Dataset

class DataCollator:
    """
    Packs raw items into homogeneous batched tensors ready for network input.
    """
    def __init__(self, pad_value: float = 0.0):
        self.pad_value = pad_value

    def __call__(self, batch: List[Tuple[torch.Tensor, Optional[torch.Tensor]]]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        xs = [item[0] for item in batch]
        ys = [item[1] for item in batch if item[1] is not None]

        # Stack xs
        stacked_x = torch.stack(xs, dim=0)

        # Stack ys if available
        stacked_y = torch.stack(ys, dim=0) if len(ys) == len(batch) else None

        return stacked_x, stacked_y


class Batcher:
    """
    DataLoader wrapper creating mini-batches for PyTorch/PYMBBO execution.
    """
    def __init__(self, dataset: Dataset, batch_size: int = 32, shuffle: bool = True, 
                 collator: Optional[DataCollator] = None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.collator = collator or DataCollator()

        self._dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.collator
        )

    def __len__(self) -> int:
        return len(self._dataloader)

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        for x_batch, y_batch in self._dataloader:
            yield x_batch, y_batch
