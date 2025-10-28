import math, random, torch
from typing import Dict, List, Iterator, Optional, Tuple
from collections import deque
from torch.utils.data import BatchSampler

class DistributedModalitySignatureBatchSampler(BatchSampler):
    """
    Deterministic, step-aligned sampler:
    1) Build per-signature batches with identical shuffle (seed+epoch) on all ranks.
    2) Interleave them in strict round-robin into ONE global list.
    3) Shard that global list with [rank::world_size].

    => At step t, every rank yields the batch from the SAME modality signature.
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
            pad_to_equal: bool = True,   # pad per-signature to equalize modulo world_size
        ):
            assert 0 <= rank < world_size
            self.indices_by_sig = {s: list(v) for s, v in indices_by_sig.items()}
            self.batch_size = int(batch_size)
            
            # TODO: Please note that world size and rank is always 1 and 0, given that dist cannot be called
            # TODO: the sampler lives on the driver, not within each ray, so we 
            # TODO: just need to make sure that the batches it yields are divisible
            # TODO: by the DP (number of data splits etc.)
            self.world_size = int(world_size)
            self.rank = int(rank)

            self.drop_last = bool(drop_last)
            self.shuffle = bool(shuffle)
            self.seed = int(seed)
            self.epoch = 0
            self.pad_to_equal = bool(pad_to_equal)
            self.sigs = sorted(self.indices_by_sig.keys())

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def _batches_for(self, pool: List[int]) -> List[List[int]]:
        out = []
        for start in range(0, len(pool), self.batch_size):
            chunk = pool[start:start + self.batch_size]
            if len(chunk) < self.batch_size and self.drop_last:
                continue
            if chunk:
                out.append(chunk)
        return out

    def __iter__(self) -> Iterator[List[int]]:
        
        # 1) identical RNG on all ranks
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # 2) build per-sig pools with identical shuffle
        pools = {}
        for s in self.sigs:
            v = list(self.indices_by_sig[s])
            if self.shuffle and len(v) > 1:
                perm = torch.randperm(len(v), generator=g).tolist()
                v = [v[i] for i in perm]
            pools[s] = v

        # 3) per-sig batches
        per_sig_batches = {s: self._batches_for(pools[s]) for s in self.sigs}

        # 4) optionally pad EACH signature to multiple of world_size so RR remains aligned
        if self.pad_to_equal and not self.drop_last:
            for s, blist in per_sig_batches.items():
                if len(blist) == 0:
                    continue
                r = len(blist) % self.world_size
                if r != 0:
                    need = self.world_size - r
                    per_sig_batches[s] = blist + blist[:need]

        # 5) deterministic RR order over signatures
        order = list(self.sigs)
        if self.shuffle and len(order) > 0:
            k = int(torch.randint(0, len(order), (1,), generator=g).item())
            order = order[k:] + order[:k]

        # 6) build ONE global interleaved list by RR popping one batch per signature
        per_sig_queues = {s: deque(per_sig_batches[s]) for s in order}
        active = deque([s for s in order if len(per_sig_queues[s]) > 0])

        global_rr: List[Tuple[str, List[int]]] = []
        while active:
            s = active.popleft()
            q = per_sig_queues[s]
            if q:
                global_rr.append((s, q.popleft()))
                if q:
                    active.append(s)

        # 7) final shard: step-aligned across ranks
        for _, batch in global_rr[self.rank::self.world_size]:
            yield batch

    def __len__(self) -> int:
        # compute total global batches after drop_last/pad, then shard
        total = 0
        for pool in self.indices_by_sig.values():
            full, rem = divmod(len(pool), self.batch_size)
            nb = full + (0 if self.drop_last or rem == 0 else 1)
            if self.pad_to_equal and not self.drop_last and nb > 0:
                r = nb % self.world_size
                if r != 0:
                    nb += (self.world_size - r)
            total += nb
        return total // self.world_size

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
