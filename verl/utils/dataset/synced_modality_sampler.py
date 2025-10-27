import random
from typing import Dict, List, Iterator
from collections import deque
import torch
import torch.distributed as dist
from torch.utils.data import BatchSampler

def _is_dist():
    return dist.is_available() and dist.is_initialized()

def _get_rank():
    return dist.get_rank() if _is_dist() else 0

def _get_world_size():
    return dist.get_world_size() if _is_dist() else 1

class SyncedModalitySignatureBatchSampler(BatchSampler):
    """
    Globally synchronized (per-step) modality-signature sampler.
    Rank 0 selects the next signature each step and broadcasts to all ranks.
    Each rank pops a batch for that signature from its local queue; if empty, it pads
    by sampling with replacement from the signature's pool to keep shapes identical.
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
        self.rng = random.Random(seed + _get_rank())  # rank-shifted for local shuffles
        self.sigs = list(self.indices_by_sig.keys())

        # Precompute per-signature batch queues (local view)
        self._per_sig_batches = {s: self._batches_for(self.indices_by_sig[s]) for s in self.sigs}

        # Rank-0 constructs the global round-robin order; others will receive it step-by-step.
        if self.shuffle:
            # Build a per-rank local shuffle to reduce correlation, but selection is driven by rank0 later.
            for s in self.sigs:
                self.rng.shuffle(self.indices_by_sig[s])

        # Active signatures locally (we still need to know if we ran out and must pad)
        self._active = {s: deque(self._per_sig_batches[s]) for s in self.sigs}

        # Rank 0 keeps a global RR deque of signatures that still have *any* batches somewhere.
        # Non-zero ranks will just receive the selected signature each step.
        if _get_rank() == 0:
            # NOTE: the "global" notion of availability is approximated by rank0's queues.
            order = list(self.sigs)
            if self.shuffle and len(order) > 0:
                k = self.rng.randrange(len(order))
                order = order[k:] + order[:k]
            else:
                order = sorted(order)
            self._global_rr = deque([s for s in order if len(self._active[s]) > 0])

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

    def _pick_next_signature_rank0(self) -> str | None:
        """Rank 0: choose next signature with available batches; rotate RR."""
        while self._global_rr:
            s = self._global_rr[0]
            # If rank0's local queue is empty for s, drop it from RR and continue.
            if len(self._active[s]) == 0:
                self._global_rr.popleft()
                continue
            # Rotate so round-robin continues next time
            self._global_rr.rotate(-1)
            return s
        return None  # no signatures left

    def _broadcast_signature(self, sig: str | None) -> str | None:
        """Broadcast chosen signature (or None to indicate stop)."""
        obj_list = [sig]
        if _is_dist():
            dist.broadcast_object_list(obj_list, src=0)
        return obj_list[0]

    def _pad_batch_from_pool(self, sig: str) -> List[int]:
        """Synthesize a batch by sampling with replacement from the signature's pool."""
        pool = self.indices_by_sig.get(sig, [])
        if len(pool) == 0:
            # Absolute fallback: return a degenerate batch of zeros (harmless indices)
            return [0] * self.batch_size
        # Sample with replacement to reach batch_size.
        return [self.rng.choice(pool) for _ in range(self.batch_size)]

    def __iter__(self) -> Iterator[List[int]]:
        rank = _get_rank()

        while True:
            # Rank 0 picks signature for this global step.
            if rank == 0:
                next_sig = self._pick_next_signature_rank0()
            else:
                next_sig = None

            # Broadcast decision
            next_sig = self._broadcast_signature(next_sig)

            # Stop condition
            if next_sig is None:
                break

            # Each rank tries to pop a batch from its local queue for this signature.
            q = self._active.get(next_sig, deque())
            if len(q) > 0:
                batch_idx = q.popleft()
            else:
                # Local queue exhausted: fabricate a pad batch so shapes match.
                batch_idx = self._pad_batch_from_pool(next_sig)

            yield batch_idx

    def __len__(self) -> int:
        """
        Upper bound on number of synchronized steps (computed from rank0 viewpoint).
        Safe approximation: total #batches across signatures on *this* rank.
        In practice, training will stop when rank0 announces None.
        """
        total = 0
        for pool in self.indices_by_sig.values():
            full, rem = divmod(len(pool), self.batch_size)
            total += full + (0 if self.drop_last or rem == 0 else 1)
        return total



## TO IMPLEMENT THE ABOVE: REPLACE THE FOLLOWING IN THE CREATE RL SAMPLER;

# def create_rl_sampler(data_config, dataset):
#     indices_by_sig = build_indices_by_signature(dataset)  # your existing helper
#     SamplerCls = SyncedModalitySignatureBatchSampler if _is_dist() else ModalitySignatureBatchSampler
#     sampler = SamplerCls(
#         indices_by_sig=indices_by_sig,
#         batch_size=int(data_config.train_batch_size),
#         drop_last=data_config.train_modality_batching.drop_last,
#         seed=data_config.get("seed", 42),
#         shuffle=True,
#     )
#     return sampler
