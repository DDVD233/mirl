import numpy as np
import os
import sys
import torch
from torch.utils.data import BatchSampler
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from verl.utils.dataset.rl_dataset import RLHFDataset

import os
import torch

def log_failed_path(
    path: str,
    kind: str,
    logfile: str = "/home/keaneong/human-behavior/verl/multi_task_classification/failed_ext_paths_log/missing_feats.txt"
) -> None:
    """
    Append a single failed path to logfile.
    kind: "video" or "audio" (or any label you like).
    Auto-creates parent directory if missing.
    """
    try:
        os.makedirs(os.path.dirname(logfile), exist_ok=True)
        with open(logfile, "a") as f:
            f.write(f"{kind}\t{path}\n")
    except Exception as e:
        # Best-effort logging; don't break data loading over logging problems
        print(f"[WARN] Failed to write to {logfile}: {e}")

def _resolve_single_path(maybe_path):
    """
    Accepts either a single string path or a list/tuple of paths.
    Returns the first existing path if any, else None.
    """
    if maybe_path is None:
        return None

    if isinstance(maybe_path, (list, tuple)):
        for p in maybe_path:
            if isinstance(p, str) and os.path.exists(p):
                return p
        return None

    if isinstance(maybe_path, str) and os.path.exists(maybe_path):
        return maybe_path

    return None

def load_feat_or_none(path, kind: str, logfile: str = "/home/keaneong/human-behavior/verl/multi_task_classification/failed_ext_paths_log/missing_feats.txt"):
    """
    Try to load a torch .pt feature file. If missing or load fails, log and return None.
    """
    resolved = _resolve_single_path(path)
    if resolved is None:
        # Path missing (or none of the candidates exist)
        log_failed_path(str(path), kind, logfile)
        return None

    try:
        # map_location='cpu' to be robust on non-GPU workers
        return torch.load(resolved, map_location="cpu")
    except Exception as e:
        # Load failed; log and return None
        log_failed_path(f"{resolved} | load_error={type(e).__name__}: {e}", kind, logfile)
        return None
# ----------------------------------------------


class OmniClassifierDataset(RLHFDataset):
    def __init__(self, *args, label_key='answer', label_map=None, dataset_key='dataset', **kwargs):
        super().__init__(*args, **kwargs)
        self.label_key = label_key 
        self.label_map = label_map  # Optional: dict mapping raw label to class index
        self.dataset_key = dataset_key

    def __getitem__(self, item):
        # connects to RLHFDataset __getitem__
        row_dict = super().__getitem__(item)

        # LOADING OF VIDEO/AUDIO FEATURES
        # Accept either a single string path or a list of paths.
        # We look for keys 'ext_video_feats_path' / 'ext_audio_feats_path'.
        # If you sometimes store them under 'ext_video_feats'/'ext_audio_feats' as lists,
        # these helpers still handle list/tuple inputs.
        video_feats_path = row_dict.get('ext_video_feats_path', row_dict.get('ext_video_feats', None))
        audio_feats_path = row_dict.get('ext_audio_feats_path', row_dict.get('ext_audio_feats', None))

        video_feat = load_feat_or_none(video_feats_path, kind="video")
        audio_feat = load_feat_or_none(audio_feats_path, kind="audio")

        row_dict['video_feats'] = video_feat
        row_dict['audio_feats'] = audio_feat

        # --- your existing label extraction ---
        raw_label = row_dict.get(self.label_key, "").lower()
        dataset_name = row_dict.get(self.dataset_key, "").lower()

        if dataset_name:
            full_label_key = f"{dataset_name}_{raw_label}"
        else:
            full_label_key = raw_label

        if self.label_map is not None:
            self.label_map = {k.lower(): v for k, v in self.label_map.items()}
            label = self.label_map.get(full_label_key, 0)

            if label == 0 and full_label_key not in self.label_map:
                print(f"[WARN] Label key '{full_label_key}' not found in label map.")
                print(f"[WARN] Raw label: '{raw_label}', Dataset: '{dataset_name}'")
                print(f"[WARN] Available keys (first 10): {list(self.label_map.keys())[:10]}...")
                raise ValueError(f"Label key '{full_label_key}' not found in label map.")
        else:
            raise ValueError(f"label_map must be provided for mapping raw labels {full_label_key} to class indices")

        row_dict["labels"] = torch.tensor(label, dtype=torch.long)
        return row_dict
    
class SkipBatchSampler(torch.utils.data.Sampler):
    """
    Wrap an existing *batch* sampler and skip the first `skip_batches` batches.
    This only advances the sampler (indices), NOT the dataset (no __getitem__ calls).
    """
    def __init__(self, batch_sampler: BatchSampler, skip_batches: int):
        self.batch_sampler = batch_sampler
        self.skip_batches = int(max(0, skip_batches))
        self.batch_size = getattr(BatchSampler, 'batch_size', None)
        self.drop_last = getattr(BatchSampler, 'drop_last', False)

    def __iter__(self):
        it = iter(self.batch_sampler)
        # consume batch indices without touching dataset
        for _ in range(self.skip_batches):
            try:
                next(it)
            except StopIteration:
                return
        for batch in it:
            yield batch

    def __len__(self):
        try:
            return max(0, len(self.batch_sampler) - self.skip_batches)
        except TypeError:
            # Fallback if underlying batch_sampler has no __len__
            return 0