from collections import deque
from typing import Dict, List, Iterator, Any
import torch
from torch.utils.data import BatchSampler
import os, json

class ResModalitySignatureBatchSampler(BatchSampler):
    """
    Stateless, deterministic modality-based batch sampler.
    Each batch is homogeneous by modality_signature.
    Round-robins across modalities until all are exhausted.

    - No internal state (safe for resume via StatefulDataLoader)
    - Deterministic given seed, indices_by_sig, shuffle/drop_last/batch_size
    """

    def __init__(
        self,
        indices_by_sig: Dict[str, List[int]],
        batch_size: int,
        drop_last: bool = True,
        shuffle: bool = True,
        base_seed: int = 42,
        dataset: Any = None,          # optional, for debug logging only
        log_path: str | None = None,  # optional, for debug logging only
    ):
        self.indices_by_sig = {s: list(v) for s, v in indices_by_sig.items()}
        self.batch_size  = int(batch_size)
        self.drop_last   = bool(drop_last)
        self.shuffle     = bool(shuffle)
        self._base_seed  = int(base_seed)
        self._dataset    = dataset
        self._log_path   = log_path

        if self._log_path:
            os.makedirs(os.path.dirname(self._log_path), exist_ok=True)
            with open(self._log_path, "w") as f:
                f.write("=== ModalitySignatureBatchSampler (stateless) ===\n")
                f.write(f"Modalities: {list(self.indices_by_sig.keys())}\n")

    def __len__(self) -> int:
        total = 0
        B = self.batch_size
        for pool in self.indices_by_sig.values():
            n = len(pool)
            full, rem = divmod(n, B)
            total += full + (0 if self.drop_last or rem == 0 else 1)
        return total

    def __iter__(self) -> Iterator[List[int]]:
        # Local torch RNG; deterministic given base_seed
        g = torch.Generator()
        g.manual_seed(self._base_seed)

        # 1) Per-signature permutations
        per_sig_batches: Dict[str, deque] = {}
        sigs = list(self.indices_by_sig.keys())

        for s in sigs:
            idxs = torch.tensor(self.indices_by_sig[s], dtype=torch.long)
            if self.shuffle and idxs.numel() > 0:
                perm = idxs[torch.randperm(idxs.numel(), generator=g)]
            else:
                perm = idxs

            batches = []
            for start in range(0, perm.numel(), self.batch_size):
                chunk = perm[start:start + self.batch_size]
                if chunk.numel() < self.batch_size and self.drop_last:
                    continue
                if chunk.numel() > 0:
                    batches.append(chunk.tolist())
            per_sig_batches[s] = deque(batches)

        # 2) Round-robin order (deterministic rotation)
        if self.shuffle and len(sigs) > 0:
            k = int(torch.randint(low=0, high=len(sigs), size=(1,), generator=g).item())
            order = sigs[k:] + sigs[:k]
        else:
            order = sorted(sigs)

        active = deque([s for s in order if per_sig_batches[s]])

        # 3) Yield batches
        batch_counter = 0
        while active:
            s = active.popleft()
            q = per_sig_batches[s]
            batch = q.popleft()
            batch_counter += 1

            if self._log_path:
                with open(self._log_path, "a") as f:
                    f.write(f"[batch {batch_counter}] sig={s} size={len(batch)}\n")
                    if self._dataset is not None:
                        try:
                            for i in batch:
                                f.write(f"  - {json.dumps(self._dataset[i], default=str)}\n")
                        except Exception as e:
                            f.write(f"[WARN] dataset access error: {e}\n")

            yield batch
            if q:
                active.append(s)
