import random
from typing import Dict, List, Iterator, Optional
from collections import defaultdict, deque
from torch.utils.data import BatchSampler

class ModalitySignatureBatchSampler(BatchSampler):
    """
    Round-robin across modality signatures, pruning exhausted signatures.
    - Shuffles within each signature if shuffle=True (train).
    - Each yielded batch is homogeneous by modality_signature.
    - If a signature runs out of batches, it is removed and RR continues.
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

    def _batches_for(self, pool: List[int]) -> List[List[int]]:
        n = len(pool)
        batches = []
        for start in range(0, n, self.batch_size):
            chunk = pool[start:start + self.batch_size]
            if len(chunk) < self.batch_size and self.drop_last:
                continue
            if chunk:
                batches.append(chunk)
        return batches

    def __iter__(self) -> Iterator[List[int]]:
        # Fresh pools + optional shuffle within each signature
        pools = {s: list(v) for s, v in self.indices_by_sig.items()}
        for s in pools:
            if self.shuffle:
                self.rng.shuffle(pools[s])

        # Build per-signature batch queues; essentially a dictionary with batches of each different modality signature
        per_sig_batches = {s: deque(self._batches_for(pools[s])) for s in self.sigs}

        # Establish RR order
        order = list(self.sigs)
        if self.shuffle:
            # rotate start signature per epoch for variety (keeps RR structure)
            k = self.rng.randrange(len(order)) if order else 0
            order = order[k:] + order[:k]
        else:
            order = sorted(order)

        # Active signatures as a deque for easy rotation
        active = deque([s for s in order if len(per_sig_batches[s]) > 0])

        while active:
            s = active.popleft() # take the queue's leftmost element (modality signature)
            q = per_sig_batches[s] # access all of the batched stuff
            if q:
                yield q.popleft() # yield that batch
                # if still has batches, push to the end to continue RR
                if q:
                    active.append(s) # reappend the modality signature to the active queue
                # if q is empty, we simply don't re-append s → pruned automatically
                else:
                    print(f"Ran-Out: Pruning modality signature: {s}")
            
    def __len__(self) -> int:
        # Total number of batches across all signatures (after drop_last handling)
        total = 0
        for pool in self.indices_by_sig.values():
            full, rem = divmod(len(pool), self.batch_size)
            total += full + (0 if self.drop_last or rem == 0 else 1)
        return total