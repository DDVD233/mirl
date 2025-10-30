import os
import random
import json
from typing import Dict, List, Iterator, Any
from collections import deque
from torch.utils.data import BatchSampler


class DebugModalitySignatureBatchSampler(BatchSampler):
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
        dataset: Any = None,  # <-- NEW: optional dataset reference
        log_path: str = "/home/keaneong/human-behavior/test_intentqa_debug_batches.txt",
    ):
        self.indices_by_sig = {s: list(v) for s, v in indices_by_sig.items()}
        self.batch_size = int(batch_size)
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.rng = random.Random(seed)
        self.sigs = list(self.indices_by_sig.keys())
        self.dataset = dataset  # keep reference if you want to log real samples

        # --- Logging setup ---
        self.log_path = log_path
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path, "w") as f:
            f.write(f"=== Debug Log for ModalitySignatureBatchSampler ===\n")
            f.write(f"Batch size: {self.batch_size}\n")
            f.write(f"Modalities: {self.sigs}\n\n")

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
        # 1) Shuffle pools per signature if needed
        pools = {s: list(v) for s, v in self.indices_by_sig.items()}
        for s in pools:
            if self.shuffle:
                self.rng.shuffle(pools[s])

        # 2) Build per-signature batch queues
        per_sig_batches = {s: deque(self._batches_for(pools[s])) for s in self.sigs}

        # 3) Establish round-robin order
        order = list(self.sigs)
        if self.shuffle:
            k = self.rng.randrange(len(order)) if order else 0
            order = order[k:] + order[:k]
        else:
            order = sorted(order)

        # 4) Iterate in round-robin fashion
        active = deque([s for s in order if len(per_sig_batches[s]) > 0])

        batch_counter = 0
        while active:
            s = active.popleft()
            q = per_sig_batches[s]
            if q:
                batch_indices = q.popleft()
                batch_counter += 1

                # --- Extract actual samples if dataset provided ---
                batch_samples = None
                if self.dataset is not None:
                    try:
                        batch_samples = [self.dataset[i] for i in batch_indices]
                    except Exception as e:
                        batch_samples = f"[ERROR accessing dataset: {e}]"

                # --- Log everything to file ---
                with open(self.log_path, "a") as f:
                    f.write(f"--- Batch {batch_counter} ---\n")
                    f.write(f"Modality signature: {s}\n")
                    f.write(f"Indices: {batch_indices}\n")
                    f.write(f"Batch length: {len(batch_indices)}\n")

                    if batch_samples is not None:
                        f.write("Actual batch content:\n")
                        for i, sample in enumerate(batch_samples):
                            try:
                                # Make JSON-safe if it’s a dict or tensor-like
                                sample_str = json.dumps(sample, default=str)
                            except Exception:
                                sample_str = str(sample)
                            f.write(f"  [{i}] {sample_str}\n")
                    f.write("\n")

                yield batch_indices

                if q:
                    active.append(s)
                else:
                    print(f"Ran-Out: Pruning modality signature: {s}")
                    with open(self.log_path, "a") as f:
                        f.write(f"[INFO] Ran-Out: Pruning modality signature: {s}\n\n")

    def __len__(self) -> int:
        total = 0
        for pool in self.indices_by_sig.values():
            full, rem = divmod(len(pool), self.batch_size)
            total += full + (0 if self.drop_last or rem == 0 else 1)
        return total