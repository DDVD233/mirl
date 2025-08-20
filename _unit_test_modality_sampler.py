# test_stateful_modality_sampler_hardcoded.py

import json
from typing import Dict, Any, List, Iterator
from torch.utils.data import Dataset, BatchSampler
import random

# ==== ADJUST PATHS below to match your repo structure ====
# from verl.utils.dataset.modality_sampler import ModalitySignatureBatchSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from collections import defaultdict, deque
import torch
import numpy as np

# ---------- HARD-CODED PATHS + CONFIG ----------
JSONL_PATH = "/Users/keane/Desktop/research/human-behavior/data/all/sigs_no_lmvd_discretized_v3_template_prompts.jsonl"
TRAIN_BS   = 4
VAL_BS     = 4
SEED       = 42
TRUNCATE_RATIO = 0.001  # for quick testing; set to 1.0 to disable
# ---------------------------------------------

# TODO: Please remove text only; everything should be text_only

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


def rl_collate_fn(data_list: list[dict]) -> dict:
    """
    Collate a batch of sample dicts into batched tensors and arrays.

    Args:
        data_list: List of dicts mapping feature names to torch.Tensor or other values.

    Returns:
        Dict where tensor entries are stacked into a torch.Tensor of shape
        (batch_size, dims) and non-tensor entries are converted to
        np.ndarray of dtype object with shape (batch_size,).
    """
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.fromiter(val, dtype=object, count=len(val))

    return {**tensors, **non_tensors}

def create_rl_sampler(data_config, dataset, split: str = "train"):
    """Create a sampler for the dataset, grouping strictly by existing modality_signature."""
    import torch
    from torch.utils.data import RandomSampler, SequentialSampler

    mb_cfg = data_config.get("modality_batching") if split == "train" \
             else data_config.get("val_modality_batching")

    # (keep curriculum path if you actually use it; omitted here for brevity)

    if mb_cfg and mb_cfg.get("enabled", False):
        by_sig: Dict[str, List[int]] = {}
        for i in range(len(dataset)):
            row = dataset.dataframe[i] if hasattr(dataset, "dataframe") else dataset[i]
            sig = row.get("modality_signature")
            if sig is None:
                print(f"[WARNING] Row {i} missing 'modality_signature'. Skipping.")
                continue
            by_sig.setdefault(sig, []).append(i)

        batch_size = mb_cfg.get("batch_size", data_config.get(
            "train_batch_size" if split=="train" else "val_batch_size"
        ))
        drop_last = mb_cfg.get("drop_last", split=="train")
        shuffle = (split == "train")

        return ModalitySignatureBatchSampler(
            indices_by_sig=by_sig,
            batch_size=int(batch_size),
            drop_last=drop_last,
            seed=data_config.get("seed", 42),
            shuffle=shuffle,
        )

    # Fallbacks
    if data_config.get("shuffle", True) and split == "train":
        g = torch.Generator(); g.manual_seed(data_config.get("seed", 1))
        return RandomSampler(data_source=dataset, generator=g)
    else:
        return SequentialSampler(data_source=dataset)

class JsonlDataset(Dataset):
    def __init__(self, jsonl_path: str, truncate_ratio: float = TRUNCATE_RATIO, seed: int = SEED):
        """
        Loads ONLY entries that already have 'modality_signature'.
        Optionally keeps a proportion per signature for fast debugging.
        """
        all_rows: List[Dict[str, Any]] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                ex = json.loads(ln)
                sig = ex.get("modality_signature")
                if sig is None:
                    print(f"[WARNING] Entry missing 'modality_signature'. Skipping.")
                    continue  # skip missing
                all_rows.append(ex)

        # Group by signature and truncate per signature
        sig_to_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for ex in all_rows:
            sig_to_rows[ex["modality_signature"]].append(ex)

        rng = random.Random(seed)
        truncated_rows: List[Dict[str, Any]] = []
        for sig, rows in sig_to_rows.items():
            if truncate_ratio >= 1.0:
                truncated_rows.extend(rows)
                continue
            keep_n = max(1, int(len(rows) * truncate_ratio))
            rng.shuffle(rows)
            truncated_rows.extend(rows[:keep_n])

        self.rows = truncated_rows
        self.dataframe = self  # preserve your API

        # simple stats
        counts = {sig: sum(1 for r in self.rows if r["modality_signature"] == sig) for sig in sig_to_rows}
        print(f"[DEBUG] After truncation (ratio={truncate_ratio}), total {len(self.rows)}. Per-signature: {counts}")

    def __len__(self): 
        return len(self.rows)

    def __getitem__(self, idx): 
        return self.rows[idx]


def assert_homogeneous(batch_list: List[Dict[str, Any]]):
    sigs = {b.get("modality_signature") for b in batch_list}
    if len(sigs) != 1:
        raise AssertionError(f"Non-homogeneous batch signatures: {sigs}")

def collate_with_guard(batch_list):
    assert_homogeneous(batch_list)
    return rl_collate_fn(batch_list)

def build_cfg(train_bs: int, val_bs: int, seed: int = 42):
    class Dot(dict):
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__
    return Dot({
        "train_batch_size": train_bs,
        "val_batch_size": val_bs,
        "shuffle": True,
        "seed": seed,
        "dataloader_num_workers": 0,
        "validation_shuffle": False,
        "sampler": None,
        "modality_batching":      {"enabled": True, "batch_size": train_bs, "drop_last": True},
        "val_modality_batching":  {"enabled": True, "batch_size": val_bs,   "drop_last": False},
    })

def build_loader(dataset, data_cfg, split: str):
    sampler_or_batch = create_rl_sampler(data_cfg, dataset, split=split)
    if isinstance(sampler_or_batch, BatchSampler):
        return StatefulDataLoader(
            dataset=dataset,
            batch_sampler=sampler_or_batch,
            num_workers=data_cfg["dataloader_num_workers"],
            collate_fn=collate_with_guard,
        )
    else:
        bs = data_cfg.get("train_batch_size" if split == "train" else "val_batch_size")
        return StatefulDataLoader(
            dataset=dataset,
            sampler=sampler_or_batch,
            batch_size=bs,
            num_workers=data_cfg["dataloader_num_workers"],
            drop_last=(split == "train"),
            shuffle=False if split == "val" else False,
            collate_fn=collate_with_guard,
        )

def main():
    ds = JsonlDataset(JSONL_PATH)
    print(f"Dataset size: {len(ds)}; per-signature counts:",
          {sig: sum(1 for r in ds.rows if r['modality_signature']==sig)
           for sig in sorted({r['modality_signature'] for r in ds.rows})})

    cfg = build_cfg(TRAIN_BS, VAL_BS, SEED)

    # TRAIN
    train_loader = build_loader(ds, cfg, split="train")
    print("\n[TRAIN] Iteration 1")
    n_train_batches = sum(1 for _ in train_loader) # iterating as you would with the train loader
    print(f"train steps: {n_train_batches} (drop_last=True)")

    # New epoch
    train_loader2 = build_loader(ds, cfg, split="train")
    n_train_batches2 = sum(1 for _ in train_loader2)
    assert n_train_batches == n_train_batches2
    print("[TRAIN] Iteration 2: step count consistent")

    # VAL
    val_loader = build_loader(ds, cfg, split="val")
    print("\n[VAL] Iteration 1")
    n_val_batches = sum(1 for _ in val_loader)
    print(f"val steps: {n_val_batches} (drop_last=False)")

    # Stateful resume check (if supported)
    if hasattr(train_loader, "state_dict"):
        print("\n[STATEFUL] Testing resume mid-epoch")
        train_loader3 = build_loader(ds, cfg, split="train")
        it = iter(train_loader3)
        next(it); next(it)  # consume 2
        sd = train_loader3.state_dict()
        train_loader4 = build_loader(ds, cfg, split="train")
        train_loader4.load_state_dict(sd)
        resumed = sum(1 for _ in train_loader4)
        print(f"resumed batches after 2 consumed: {resumed}")

    print("\nOK: StatefulDataLoader + sampler test finished.")

if __name__ == "__main__":
    main()