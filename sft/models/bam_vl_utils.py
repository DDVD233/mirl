# models/bam_vl_utils.py
"""
Feature utilities for BAM-on-Qwen3-VL (ChildPlay extracted modalities).

Unlike bam_utils.py (Omni), the side features here are PRE-EXTRACTED, ALREADY
NORMALIZED .pt tensors referenced by path in the JSONL, not raw OpenPose/OpenSmile
dicts. Confirmed shapes (Qwen3-VL ChildPlay splits):

  facial  ext_facial_feats : Tensor [T, 478, 3]  (MediaPipe FaceMesh, x/y/z)
  pose    ext_pose_feats   : Tensor [T,  17, 3]  (COCO-17 keypoints)
  audio   ext_audio_feats  : dict{"features": Tensor [num_windows, 88], ...}  (eGeMAPS)

Per-frame flatten + temporal pooling gives the fixed adapter input dims:
  facial meanstd : 478*3*2 = 2868
  pose   meanstd : 17*3*2  = 102
  audio  none    : 88        (already a single normalized window)

All streams are already normalized, so no extra normalization is applied here.
Variable T means per-sample tensors cannot be collated raw — callers pool to a
fixed vector in the dataset __getitem__ so the default collate can stack [B, D].
"""
from typing import Optional, Literal
import torch

from .bam_adapter import BehavioralAdapterModule

PoolMode = Literal["none", "mean", "meanstd"]

# Locked feature dims for the ChildPlay extracted modalities (meanstd for facial/pose).
D_FACIAL_FEAT_MEANSTD = 478 * 3 * 2   # 2868
D_POSE_FEAT_MEANSTD = 17 * 3 * 2      # 102
D_AUDIO_FEAT = 88                     # eGeMAPS functionals, single window


# ---------------------------------------------------------------------------
# Temporal pooling
# ---------------------------------------------------------------------------

def _pool_temporal(x: torch.Tensor, mode: PoolMode) -> Optional[torch.Tensor]:
    """x: [T, D] -> pooled [D] (mean), [2D] (meanstd), or [D] (none, requires T==1)."""
    if x is None or x.ndim != 2:
        return None
    T, D = x.shape
    if T == 0 or D == 0:
        return None
    x = x.float()
    if mode == "none":
        return x.reshape(-1) if T == 1 else x.mean(dim=0)
    if mode == "mean":
        return x.mean(dim=0)
    if mode == "meanstd":
        return torch.cat([x.mean(dim=0), x.std(dim=0, unbiased=False)], dim=0)
    return None


def _pad_trunc_1d(v: torch.Tensor, target_dim: int) -> torch.Tensor:
    """Defensive: force a 1-D vector to exactly target_dim."""
    D = v.numel()
    if D == target_dim:
        return v
    if D > target_dim:
        return v[:target_dim]
    out = v.new_zeros(target_dim)
    out[:D] = v
    return out


# ---------------------------------------------------------------------------
# Per-stream single-sample builders (consume the loaded .pt object)
# ---------------------------------------------------------------------------

def build_keypoint_feat(obj, temporal_mode: PoolMode = "meanstd",
                        target_dim: Optional[int] = None) -> Optional[torch.Tensor]:
    """
    Facial/pose: tensor [T, K, C] -> flatten keypoints to [T, K*C] -> temporal pool.
    Returns a fixed [target_dim] vector, or None if invalid.
    """
    if obj is None or not torch.is_tensor(obj):
        return None
    t = obj.float()
    if t.ndim == 2:           # already [T, D]
        seq = t
    elif t.ndim == 3:         # [T, K, C] -> [T, K*C]
        seq = t.reshape(t.shape[0], -1)
    else:
        return None
    v = _pool_temporal(seq, temporal_mode)
    if v is None:
        return None
    return _pad_trunc_1d(v, target_dim) if target_dim is not None else v


def build_audio_feat(obj, temporal_mode: PoolMode = "none",
                     target_dim: Optional[int] = None) -> Optional[torch.Tensor]:
    """
    Audio: dict{"features": [num_windows, 88]} (or a bare tensor) -> pooled [target_dim].
    """
    if obj is None:
        return None
    x = obj["features"] if isinstance(obj, dict) and "features" in obj else obj
    if not torch.is_tensor(x):
        try:
            x = torch.as_tensor(x)
        except Exception:
            return None
    x = x.float()
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if x.ndim != 2:
        return None
    v = _pool_temporal(x, temporal_mode)
    if v is None:
        return None
    return _pad_trunc_1d(v, target_dim) if target_dim is not None else v


# ---------------------------------------------------------------------------
# Building the three BAM adapters
# ---------------------------------------------------------------------------

def maybe_build_vl_adapters(
    *,
    out_dim_hidden: int,                 # backbone pooled hidden size H (Qwen3-VL: 4096)
    use_facial: bool = False,
    use_pose: bool = False,
    use_audio: bool = False,
    d_facial_feat: Optional[int] = None,
    d_pose_feat: Optional[int] = None,
    d_audio_feat: Optional[int] = None,
    hidden_facial: int = 128,
    hidden_pose: int = 128,
    hidden_audio: int = 128,
    p_moddrop_facial: float = 0.30,
    p_moddrop_pose: float = 0.30,
    p_moddrop_audio: float = 0.30,
    use_ln_facial: bool = False,
    use_ln_pose: bool = False,
    use_ln_audio: bool = False,
    alpha_init: float = 1.0,
):
    """Build the (facial, pose, audio) BehavioralAdapterModules. Any may be None."""
    def _mk(use, d, hidden, p, use_ln):
        if not use:
            return None
        if d is None:
            raise ValueError("Adapter requested but its feature dim was not provided")
        return BehavioralAdapterModule(
            feat_dim=int(d), hidden=int(hidden), out_dim=int(out_dim_hidden),
            p_moddrop=float(p), use_ln=bool(use_ln), alpha_init=float(alpha_init),
        )

    facial_adapter = _mk(use_facial, d_facial_feat, hidden_facial, p_moddrop_facial, use_ln_facial)
    pose_adapter = _mk(use_pose, d_pose_feat, hidden_pose, p_moddrop_pose, use_ln_pose)
    audio_adapter = _mk(use_audio, d_audio_feat, hidden_audio, p_moddrop_audio, use_ln_audio)
    return facial_adapter, pose_adapter, audio_adapter
