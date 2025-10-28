# debug_sync_sampler.py
from collections import defaultdict
from synced_modality_sampler import DistributedModalitySignatureBatchSampler

def summarize_run(indices_by_sig, world_size, batch_size, *, seed=123, epoch=0,
                  drop_last=True, shuffle=True, pad_to_equal=False):
    # Reverse map: idx -> signature
    idx2sig = {i: s for s, idxs in indices_by_sig.items() for i in idxs}

    per_rank = []
    for rank in range(world_size):
        s = DistributedModalitySignatureBatchSampler(
            indices_by_sig=indices_by_sig,
            batch_size=batch_size,
            world_size=world_size,
            rank=rank,
            drop_last=drop_last,
            shuffle=shuffle,
            seed=seed,
            pad_to_equal=pad_to_equal,
        )
        s.set_epoch(epoch)
        per_rank.append(list(iter(s)))

    steps = len(per_rank[0])
    print(f"world_size={world_size} batch_size={batch_size} steps={steps}\n")

    for t in range(steps):
        sigs_this_step = []
        print(f"Step {t:02d}:")
        for r in range(world_size):
            batch = per_rank[r][t]
            sig = {idx2sig[i] for i in batch}
            sigs_this_step.append(next(iter(sig)))
            print(f"  rank {r}: sig={next(iter(sig))}  batch={batch}")
        ok = len(set(sigs_this_step)) == 1
        print(f"  aligned={ok}\n")

if __name__ == "__main__":
    # Example pools (edit freely while debugging)
    indices_by_sig = {
        "img": list(range(0, 12)),       # 12 items
        "vid": list(range(100, 112)),    # 12 items
        "aud": list(range(200, 208)),    # 8 items
    }
    summarize_run(
        indices_by_sig=indices_by_sig,
        world_size=4,
        batch_size=2,
        seed=42,
        epoch=0,
        drop_last=True,
        shuffle=True,
        pad_to_equal=False,
    )