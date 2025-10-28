import math, random, torch
from typing import Dict, List, Iterator, Optional
from collections import deque
from torch.utils.data import BatchSampler

class DistributedModalitySignatureBatchSampler(BatchSampler):
    """
    Round-robin across modality signatures with per-rank sharding at the *batch* level.
    - Same global shuffle on every rank for a given (seed, epoch)
    - Each rank sees a disjoint slice of batches: batches[rank::world_size]
    - Optional drop_last; else pad deterministically so all ranks yield equal counts
    """
    def __init__(
        self,
        indices_by_sig: Dict[str, List[int]],
        batch_size: int,
        *,
        world_size: int,
        rank: int,
        drop_last: bool = True,
        shuffle: bool = True,
        seed: int = 42,
        pad_to_equal: bool = True,  # keep steps identical across ranks
    ):
        assert 0 <= rank < world_size
        self.indices_by_sig = {s: list(v) for s, v in indices_by_sig.items()}
        self.batch_size = int(batch_size)
        self.world_size = int(world_size)
        self.rank = int(rank)
        self.drop_last = bool(drop_last)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        self.pad_to_equal = bool(pad_to_equal)

        # stable order of signatures to RR over (shuffle via seed/epoch later)
        self.sigs = sorted(self.indices_by_sig.keys())

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def _batches_for(self, pool: List[int]) -> List[List[int]]:
        batches = []
        for start in range(0, len(pool), self.batch_size):
            chunk = pool[start:start + self.batch_size]
            if len(chunk) < self.batch_size and self.drop_last:
                continue
            if chunk:
                batches.append(chunk)
        return batches

    def __iter__(self) -> Iterator[List[int]]:
        # 1) Global RNG that’s identical on all ranks for (seed + epoch)
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # 2) Build global per-signature pools with identical shuffle
        pools = {}
        for s in self.sigs:
            v = list(self.indices_by_sig[s])
            if self.shuffle:
                idx = torch.randperm(len(v), generator=g).tolist()
                v = [v[i] for i in idx]
            pools[s] = v

        # 3) Turn into global batch lists per signature
        per_sig_batches = {s: self._batches_for(pools[s]) for s in self.sigs}

        # 4) Optional padding to equalize total number of global batches
        if not self.drop_last and self.pad_to_equal:
            # pad each signature's batches to the same modulo world_size
            for s, blist in per_sig_batches.items():
                rem = len(blist) % self.world_size
                if rem != 0 and len(blist) > 0:
                    need = self.world_size - rem
                    # deterministic pad: repeat from start
                    per_sig_batches[s] = blist + blist[:need]

        # 5) Deterministic RR order across signatures
        order = list(self.sigs)
        if self.shuffle and len(order) > 0:
            # rotate start in a deterministic way based on RNG
            k = int(torch.randint(0, len(order), (1,), generator=g).item())
            order = order[k:] + order[:k]

        # 6) Shard at the *batch* level: each rank takes blist[rank::world_size]
        per_sig_shards = {s: deque(per_sig_batches[s][self.rank::self.world_size]) for s in order}

        # 7) RR over active signatures on this rank
        active = deque([s for s in order if len(per_sig_shards[s]) > 0])
        while active:
            s = active.popleft()
            q = per_sig_shards[s]
            if q:
                yield q.popleft()
                if q:
                    active.append(s)  # keep RR
            # else: signature pruned automatically

    def __len__(self) -> int:
        # Per-rank length after sharding (+ padding if enabled)
        total_global_batches = 0
        for pool in self.indices_by_sig.values():
            full, rem = divmod(len(pool), self.batch_size)
            nb = full + (0 if self.drop_last or rem == 0 else 1)
            if not self.drop_last and self.pad_to_equal and nb > 0:
                # pad up to multiple of world_size
                r = nb % self.world_size
                if r != 0:
                    nb += (self.world_size - r)
            total_global_batches += nb
        # Each rank sees ceil(total_global_batches / world_size) only if remainder exists at the global level,
        # but because we pad-to-equal per signature, it becomes exact:
        return total_global_batches // self.world_size



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
