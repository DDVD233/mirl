import random
from typing import Dict, List, Iterator, Optional
from torch.utils.data import BatchSampler

class ModalitySignatureBatchSampler(BatchSampler):
    """
    Yields batches where each batch is homogeneous by 'modality_signature'.
    No weighting logic — just shuffle (train) or sequential (val).
    """
    def __init__(
        self,
        indices_by_sig: Dict[str, List[int]],
        batch_size: int,
        drop_last: bool = True,
        seed: int = 42,
        shuffle: bool = True,
    ):
        self.indices_by_sig = {s: list(v) for s, v in indices_by_sig.items()}
        self.batch_size = int(batch_size)
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.rng = random.Random(seed)
        self.sigs = list(self.indices_by_sig.keys())

    def __iter__(self) -> Iterator[List[int]]:
        # copy fresh pools
        pools = {s: list(v) for s, v in self.indices_by_sig.items()}
        for s in self.sigs:
            if self.shuffle:
                self.rng.shuffle(pools[s])

        for s in self.sigs:
            pool = pools[s]
            n = len(pool)
            for start in range(0, n, self.batch_size):
                end = start + self.batch_size
                batch = pool[start:end]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                if batch:
                    yield batch

    def __len__(self) -> int:
        total = 0
        for s, pool in self.indices_by_sig.items():
            full, rem = divmod(len(pool), self.batch_size)
            total += full + (0 if self.drop_last or rem == 0 else 1)
        return total
