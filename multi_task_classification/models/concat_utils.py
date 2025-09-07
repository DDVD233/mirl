# concat_utils.py
from __future__ import annotations
from typing import Iterable, Optional, Tuple
import torch

# Import ONLY the existing single-sample utilities you already have.
from .adapter_utils import (
    build_audio_feat_single,      # (dict|None) -> 1D tensor or None
    build_video_feat_single,      # (dict|None) -> 1D tensor or None
    _pad_trunc_1d,               # (tensor|None, target_dim) -> 1D tensor (zero-pads/truncs)
)

@torch.no_grad()
def safe_pool_audio_batch(
    opensmile_list: Iterable[dict | None],
    *,
    device: Optional[torch.device],
    temporal_mode: str = "none",
    norm: Optional[str] = None,
    target_dim: int = 0,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Returns [B, target_dim]. Any bad/None sample becomes zeros([target_dim]).
    Uses your existing `build_audio_feat_single` and `_pad_trunc_1d`.
    """
    if target_dim <= 0:
        raise ValueError("safe_pool_audio_batch: target_dim must be > 0")

    vecs = []
    for d in opensmile_list:
        v = build_audio_feat_single(d, temporal_mode=temporal_mode, norm=norm)
        if (v is None) or (v.numel() == 0):
            v = torch.zeros(target_dim)
        else:
            v = _pad_trunc_1d(v, target_dim)
        vecs.append(v)

    out = torch.stack(vecs, dim=0)  # [B, target_dim]
    if dtype is not None:
        out = out.to(dtype=dtype)
    if device is not None:
        out = out.to(device=device, non_blocking=True)
    return out


@torch.no_grad()
def safe_pool_video_batch(
    openpose_list: Iterable[dict | None],
    *,
    device: Optional[torch.device],
    temporal_mode: str = "meanstd",
    use_conf: bool = True,
    norm: Optional[str] = None,
    target_dim: int = 0,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Returns [B, target_dim]. Any bad/None sample becomes zeros([target_dim]).
    Uses your existing `build_video_feat_single` and `_pad_trunc_1d`.
    """
    if target_dim <= 0:
        raise ValueError("safe_pool_video_batch: target_dim must be > 0")

    vecs = []
    for op in openpose_list:
        v = build_video_feat_single(op, temporal_mode=temporal_mode, use_conf=use_conf, norm=norm)
        if (v is None) or (v.numel() == 0):
            v = torch.zeros(target_dim)
        else:
            v = _pad_trunc_1d(v, target_dim)
        vecs.append(v)

    out = torch.stack(vecs, dim=0)  # [B, target_dim]
    if dtype is not None:
        out = out.to(dtype=dtype)
    if device is not None:
        out = out.to(device=device, non_blocking=True)
    return out


def _infer_batch_size(batch: dict) -> int:
    """
    Best-effort batch size inference for producing [B, d] zero tensors when needed.
    Prefers input_ids, then labels, then attention_mask; falls back to 1.
    """
    for k in ("input_ids", "labels", "attention_mask"):
        x = batch.get(k, None)
        if torch.is_tensor(x) and x.ndim >= 1:
            return int(x.size(0))
    # If dataset names exist as list/tuple length B
    ds = batch.get("dataset", None)
    if isinstance(ds, (list, tuple)):
        return len(ds)
    return 1


@torch.no_grad()
def get_concat_features(
    batch: dict,
    *,
    device: torch.device,
    d_audio_feat: int = 0,
    d_video_feat: int = 0,
    audio_temporal: str = "none",
    audio_norm: Optional[str] = None,
    video_temporal: str = "meanstd",
    video_norm: Optional[str] = None,
    use_conf: bool = True,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor | None, torch.Tensor | None]:
    """
    Convenience: build (audio_vec, video_vec) with per-sample zero substitution.
    Returns None for a modality if target_dim == 0.
    If batch["audio_feats"] or batch["video_feats"] is None / missing / not iterable,
    returns zeros([B, d_*]) where B is inferred from batch.
    """
    # Dtype default: keep features in float32 unless caller overrides
    if dtype is None:
        dtype = torch.float32

    B = _infer_batch_size(batch)

    audio_vec = None
    video_vec = None

    # ----- Audio -----
    if d_audio_feat > 0:
        audio_payload = batch.get("audio_feats", None)
        if audio_payload is None:
            # Entire modality absent -> zeros
            audio_vec = torch.zeros(B, d_audio_feat, device=device, dtype=dtype)
        elif isinstance(audio_payload, (list, tuple)):
            # Normal path: pool per-sample with safe substitution
            audio_vec = safe_pool_audio_batch(
                audio_payload,
                device=device,
                temporal_mode=audio_temporal,
                norm=audio_norm,
                target_dim=d_audio_feat,
                dtype=dtype,
            )
            # If list length mismatches inferred B, rectify with zeros of inferred B
            if audio_vec.size(0) != B:
                audio_vec = torch.zeros(B, d_audio_feat, device=device, dtype=dtype)
        else:
            # Unexpected type -> fall back to zeros
            audio_vec = torch.zeros(B, d_audio_feat, device=device, dtype=dtype)

    # ----- Video -----
    if d_video_feat > 0:
        video_payload = batch.get("video_feats", None)
        if video_payload is None:
            video_vec = torch.zeros(B, d_video_feat, device=device, dtype=dtype)
        elif isinstance(video_payload, (list, tuple)):
            video_vec = safe_pool_video_batch(
                video_payload,
                device=device,
                temporal_mode=video_temporal,
                use_conf=use_conf,
                norm=video_norm,
                target_dim=d_video_feat,
                dtype=dtype,
            )
            if video_vec.size(0) != B:
                video_vec = torch.zeros(B, d_video_feat, device=device, dtype=dtype)
        else:
            video_vec = torch.zeros(B, d_video_feat, device=device, dtype=dtype)

    return audio_vec, video_vec