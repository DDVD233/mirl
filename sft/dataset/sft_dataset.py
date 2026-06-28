import os
import torch
from torch.utils.data import BatchSampler

from dataset.base_dataset import BaseDataset
from models.bam_vl_utils import build_keypoint_feat, build_audio_feat


# Log dir is configurable via env; defaults under the repo
FAILED_PATHS_LOG = os.environ.get(
    "BAM_FAILED_PATHS_LOG",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "failed_ext_paths_log", "missing_feats.txt"),
)

# How to pool each pre-extracted VL stream into a fixed adapter-input vector.
# kind label, loader, default temporal mode. Dims come from the dataset config.
VL_STREAM_SPECS = {
    "facial": ("facial", build_keypoint_feat, "meanstd"),
    "pose":   ("pose",   build_keypoint_feat, "meanstd"),
    "audio":  ("audio",  build_audio_feat,    "none"),
}


def log_failed_path(path: str, kind: str, logfile: str = FAILED_PATHS_LOG) -> None:
    """Append a failed feature path to logfile. Best-effort — won't crash data loading."""
    try:
        os.makedirs(os.path.dirname(logfile), exist_ok=True)
        with open(logfile, "a") as f:
            f.write(f"{kind}\t{path}\n")
    except Exception as e:
        print(f"[WARN] Failed to write to {logfile}: {e}")


def _resolve_single_path(maybe_path):
    """Return the first existing path from a string or list of strings; else None."""
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


def load_feat_or_none(path, kind: str, logfile: str = FAILED_PATHS_LOG):
    """Try to load a torch .pt feature file. Returns None (and logs) on failure."""
    resolved = _resolve_single_path(path)
    if resolved is None:
        log_failed_path(str(path), kind, logfile)
        return None
    try:
        return torch.load(resolved, map_location="cpu")
    except Exception as e:
        log_failed_path(f"{resolved} | load_error={type(e).__name__}: {e}", kind, logfile)
        return None


class OmniClassifierDataset(BaseDataset):
    """
    Extends BaseDataset with classification label mapping and
    external video/audio feature loading (.pt files).

    Optionally supports mixed QA + classification via `qa_datasets`: rows
    whose dataset name appears in that set are treated as language-modeling
    tasks (answer stored as `lm_labels`, `labels` set to 0). All other rows
    go through the normal label-map classification path.
    """

    def __init__(self, *args, label_key='answer', label_map=None, dataset_key='dataset',
                 qa_datasets=None, vl_feat_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_key = label_key
        self.label_map = label_map
        self.dataset_key = dataset_key
        self.qa_datasets = set([d.lower() for d in (qa_datasets or [])])
        # vl_feat_config: {"facial": {"use": bool, "dim": int, "mode": str}, "pose": {...}, "audio": {...}}
        # default collate can stack them and missing files are handled per row.
        self.vl_feat_config = vl_feat_config or {}

    def _load_pooled_stream(self, row_dict, stream):
        """Return (feats[dim] float tensor, mask scalar) for a VL stream; zeros+0 if missing/off."""
        spec = self.vl_feat_config.get(stream, {})
        dim = int(spec["dim"])
        kind, builder, default_mode = VL_STREAM_SPECS[stream]
        mode = spec.get("mode", default_mode)
        path = row_dict.get(f"ext_{stream}_feats_path", None)
        obj = load_feat_or_none(path, kind=kind)
        v = builder(obj, mode, dim) if obj is not None else None
        if v is None:
            return torch.zeros(dim, dtype=torch.float32), torch.tensor(0.0, dtype=torch.float32)
        return v.float(), torch.tensor(1.0, dtype=torch.float32)

    def __getitem__(self, item):
        row_dict = super().__getitem__(item)

        if self.vl_feat_config:
            # Qwen3-VL ChildPlay path: load + pool facial / pose / audio to fixed vectors.
            for stream, spec in self.vl_feat_config.items():
                if not spec.get("use", False):
                    continue
                feats, mask = self._load_pooled_stream(row_dict, stream)
                row_dict[f"{stream}_feats"] = feats
                row_dict[f"{stream}_mask"] = mask
        else:
            # Legacy Omni path: load raw .pt objects; the trainer pools them.
            video_feats_path = row_dict.get('ext_video_feats_path', row_dict.get('ext_video_feats', None))
            audio_feats_path = row_dict.get('ext_audio_feats_path', row_dict.get('ext_audio_feats', None))
            row_dict['video_feats'] = load_feat_or_none(video_feats_path, kind="video")
            row_dict['audio_feats'] = load_feat_or_none(audio_feats_path, kind="audio")

        original_answer = row_dict.get(self.label_key, "").lower()
        dataset_name = row_dict.get(self.dataset_key, "").lower()
        task = str(row_dict.get("task", "")).lower()

        # QA rows: determined by task suffix (_qa) or explicit qa_datasets set
        if task.endswith("_qa") or dataset_name in self.qa_datasets:
            row_dict["lm_labels"] = original_answer
            row_dict["labels"] = torch.tensor(0, dtype=torch.long)
            return row_dict

        # Classification rows: map answer string to class index
        full_label_key = f"{dataset_name}_{original_answer}" if dataset_name else original_answer

        if self.label_map is None:
            raise ValueError(f"label_map must be provided for mapping raw labels '{full_label_key}' to class indices")

        label_map = {k.lower(): v for k, v in self.label_map.items()}
        label = label_map.get(full_label_key, 0)
        if label == 0 and full_label_key not in label_map:
            raise ValueError(f"Label key '{full_label_key}' not found in label map.")

        row_dict["labels"] = torch.tensor(label, dtype=torch.long)
        row_dict.setdefault("lm_labels", "")
        return row_dict


class SkipBatchSampler(torch.utils.data.Sampler):
    """
    Wraps an existing BatchSampler and skips the first `skip_batches` batches.
    Advances sampler indices without calling __getitem__ — safe for checkpoint resumption.
    """

    def __init__(self, batch_sampler: BatchSampler, skip_batches: int):
        self.batch_sampler = batch_sampler
        self.skip_batches = int(max(0, skip_batches))
        self.batch_size = getattr(batch_sampler, 'batch_size', None)
        self.drop_last = getattr(batch_sampler, 'drop_last', False)

    def __iter__(self):
        it = iter(self.batch_sampler)
        for _ in range(self.skip_batches):
            try:
                next(it)
            except StopIteration:
                return
        yield from it

    def __len__(self):
        try:
            return max(0, len(self.batch_sampler) - self.skip_batches)
        except TypeError:
            return 0
