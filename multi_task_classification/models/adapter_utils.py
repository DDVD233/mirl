# models/adapter_utils.py
from typing import Optional, Tuple, Iterable
import torch
import torch.nn.functional as F
# models/feature_builders.py
from typing import Dict, Literal
import torch
from .residual_logit_adapter import ResidualLogitAdapter
import os

ERROR_LOG_FILE = "/home/keaneong/human-behavior/verl/multi_task_classification/failed_ext_paths_log/adapter_utils_errors.txt"

def log_problem(where: str, detail: str, extra: dict | None = None) -> None:
    """
    Append problematic file info into a hardcoded txt file.
    Never throws; creates parent directory if needed.
    """
    if extra is None:
        extra = {}
    try:
        os.makedirs(os.path.dirname(ERROR_LOG_FILE), exist_ok=True)
        with open(ERROR_LOG_FILE, "a") as f:
            f.write(f"[RLA:{where}] {detail} | extra={extra}\n")
    except Exception as e:
        # Last-resort safeguard: print a warning but don't crash training
        print(f"[WARN] Failed to log problem: {e} ({where}: {detail})")

def maybe_build_adapters(
    *,
    domain_id_to_global_indices,
    use_rla_video: bool,
    use_rla_audio: bool,
    rla_hidden_video: int,                     # <<< NEW
    rla_hidden_audio: int,                     # <<< NEW
    p_moddrop_video: float,
    p_moddrop_audio: float,
    d_video_feat: Optional[int] = None,
    d_audio_feat: Optional[int] = None,
     # NEW: per-modality adapter knobs
    video_use_ln: bool = False,
    video_use_conf_gain: bool = False,
    video_conf_init_gain: float = 3.0,
    video_alpha_init: float = 1.0,
    audio_use_ln: bool = False,
    audio_use_conf_gain: bool = False,
    audio_conf_init_gain: float = 3.0,
    audio_alpha_init: float = 1.0,
):
    """
    Builds adapters once given explicit feature dims (no dataset probing).
    Returns: (video_adapter, audio_adapter).
    """
    video_adapter = None
    audio_adapter = None

    if use_rla_video:
        if d_video_feat is None:
            raise ValueError("USE_RLA_VIDEO=True but d_video_feat was not provided")
        
        video_adapter = ResidualLogitAdapter(
            domain_id_to_global_indices,
            feat_key="video_feats",
            feat_dim=d_video_feat,
            hidden=rla_hidden_video,
            p_moddrop=p_moddrop_video,
            use_ln=video_use_ln,
            use_conf_gain=video_use_conf_gain,
            conf_init_gain=video_conf_init_gain,
            alpha_init=video_alpha_init,
        )

    if use_rla_audio:
        if d_audio_feat is None:
            raise ValueError("USE_RLA_AUDIO=True but d_audio_feat was not provided")
        
        audio_adapter = ResidualLogitAdapter(
            domain_id_to_global_indices,
            feat_key="audio_feats",
            feat_dim=d_audio_feat,
            hidden=rla_hidden_audio,
            p_moddrop=p_moddrop_audio,
            use_ln=audio_use_ln,
            use_conf_gain=audio_use_conf_gain,
            conf_init_gain=audio_conf_init_gain,
            alpha_init=audio_alpha_init,
        )
    return video_adapter, audio_adapter

# def apply_adapters(
#     logits: torch.Tensor,
#     domain_ids: torch.Tensor,
#     *,
#     video_adapter: Optional[ResidualLogitAdapter],
#     audio_adapter: Optional[ResidualLogitAdapter],
#     video_feats: Optional[torch.Tensor] = None,   # <<—— direct tensors
#     audio_feats: Optional[torch.Tensor] = None,
#     train_mode: bool,
# ) -> torch.Tensor:
#     """
#     Add residuals in logit space using whatever adapters are present.
#     If feats are None or adapter is None, logits are returned unchanged.
#     """
#     z = logits 
#     if (video_adapter is not None) and (video_feats is not None):
#         z = video_adapter(z, domain_ids, feats=video_feats, train_mode=train_mode)
#     if (audio_adapter is not None) and (audio_feats is not None):
#         z = audio_adapter(z, domain_ids, feats=audio_feats, train_mode=train_mode)
#     return z

def apply_adapters(    logits: torch.Tensor,
    domain_ids: torch.Tensor,
    *,
    video_adapter: Optional[ResidualLogitAdapter],
    audio_adapter: Optional[ResidualLogitAdapter],
    video_feats: Optional[torch.Tensor] = None,   # <<—— direct tensors
    audio_feats: Optional[torch.Tensor] = None,
    train_mode: bool,
) -> torch.Tensor:
    
    z = logits
    B = logits.size(0)

    if (video_adapter is not None 
        and video_feats is not None 
        and video_feats.numel() > 0 
        and video_feats.size(0) == B):
        z = video_adapter(z, domain_ids, feats=video_feats, train_mode=train_mode)

    if (audio_adapter is not None 
        and audio_feats is not None 
        and audio_feats.numel() > 0 
        and audio_feats.size(0) == B):
        z = audio_adapter(z, domain_ids, feats=audio_feats, train_mode=train_mode)

    return z

### -----------------------------------------------------------------
### RETRIEVAL OF OPENPOSE AND OPENSMILE FEATURES / POOLING UTILS
### -----------------------------------------------------------------

# models/adapter_utils.py

from typing import Literal
import torch

PoolMode = Literal["none", "mean", "meanstd", "meanstdp25p75"]

# HELPER FUNCTION FOR TRUNCATION/ PADDING
def _pad_trunc_1d(x: torch.Tensor | None, target_dim: int) -> torch.Tensor:
    if x is None:
        log_problem("_pad_trunc_1d", "received None; zero-filling", extra={"target_dim": target_dim})
        return torch.zeros(target_dim)
    D = x.numel()
    if D == target_dim:
        return x
    if D > target_dim:
        return x[:target_dim]
    out = x.new_zeros(target_dim)
    out[:D] = x
    return out

def _maybe_normalize(v: torch.Tensor, norm: str | None) -> torch.Tensor:
    # NORMALIZE THE AUDIO FEATURES TO PREVENT NUMERICAL ISSUES; i.e. the residuals from blowing up or getting swamped
    if norm is None:  return v
    if norm == "none": return v
    if norm == "l2":  return v / v.norm(p=2).clamp_min(1e-6)
    if norm == "zscore":
        m, s = v.mean(), v.std(unbiased=False).clamp_min(1e-6)
        return (v - m) / s
    raise ValueError(f"Unknown norm: {norm}")

def pool_temporal(x: torch.Tensor, mode: PoolMode = "meanstd") -> torch.Tensor | None:
    """
    x: [T, D]  (for functionals, T=1). Returns pooled tensor or None if invalid/empty.
    """
    if x is None:
        log_problem("pool_temporal", "received None tensor")
        return None

    x = x.float()
    if x.ndim != 2:
        log_problem("pool_temporal", f"expected [T, D], got {tuple(x.shape)}")
        return None

    T, D = x.shape
    if T == 0 or D == 0:
        log_problem("pool_temporal", f"empty input with shape [T={T}, D={D}]")
        return None

    if mode == "none":
        if T != 1:
            log_problem("pool_temporal", f"mode='none' expects T==1, got T={T}")
            return None
        return x.squeeze(0)  # [D]

    if mode == "mean":
        return x.mean(dim=0)  # [D]

    if mode == "meanstd":
        m = x.mean(dim=0)
        s = x.std(dim=0, unbiased=False)
        return torch.cat([m, s], dim=0)  # [2D]

    if mode == "meanstdp25p75":
        m = x.mean(dim=0)
        s = x.std(dim=0, unbiased=False)
        kth25 = max(1, int(0.25 * T))
        kth75 = max(1, int(0.75 * T))
        p25   = x.kthvalue(kth25, dim=0).values
        p75   = x.kthvalue(kth75, dim=0).values
        return torch.cat([m, s, p25, p75], dim=0)  # [4D]

    log_problem("pool_temporal", f"unknown pooling mode: {mode}")
    return None

# ALL OPENPOSE FUNCTIONALITY
def openpose_dict_to_framewise(data: Dict[str, torch.Tensor] | None, use_conf: bool = True) -> torch.Tensor | None:
    if data is None or not isinstance(data, dict):
        log_problem("openpose_to_framewise", "input is None or not dict")
        return None

    chunks = []
    for k in ("pose", "face", "left_hand", "right_hand"):
        if k not in data:
            continue
        t = data[k]
        if t is None:
            log_problem("openpose_to_framewise", f"part '{k}' is None")
            continue
        if not torch.is_tensor(t):
            try:
                t = torch.as_tensor(t)
            except Exception as e:
                log_problem("openpose_to_framewise", f"cannot to_tensor '{k}': {e.__class__.__name__}")
                continue
        t = t.float()
        if t.ndim != 3 or t.shape[-1] not in (2, 3):
            log_problem("openpose_to_framewise", f"bad shape for '{k}': {tuple(t.shape)}")
            continue
        if t.shape[0] == 0:
            log_problem("openpose_to_framewise", f"empty frames for '{k}'")
            continue
        if not use_conf:
            t = t[..., :2]
        chunks.append(t.reshape(t.shape[0], -1))
    if not chunks:
        log_problem("openpose_to_framewise", "no valid parts", extra={"keys": list(data.keys())})
        return None
    try:
        return torch.cat(chunks, dim=-1)
    except Exception as e:
        log_problem("openpose_to_framewise", f"concat failed: {e.__class__.__name__}")
        return None

def build_video_feat_single(openpose: Dict[str, torch.Tensor] | None,
                            temporal_mode: Literal["mean", "meanstd", "meanstdp25p75"] = "meanstd",
                            use_conf: bool = True,
                            norm: Optional[str] = None) -> torch.Tensor | None:
    seq = openpose_dict_to_framewise(openpose, use_conf=use_conf)
    if seq is None:
        return None
    v = pool_temporal(seq, mode=temporal_mode)
    if v is None:
        return None
    return _maybe_normalize(v, norm)

def build_video_feats_batch(openpose_list: Iterable[Dict[str, torch.Tensor] | None],
                            device: Optional[torch.device] = None,
                            temporal_mode: Literal["mean", "meanstd", "meanstdp25p75"] = "meanstd",
                            use_conf: bool = True,
                            norm: Optional[str] = None,
                            target_dim: int = None) -> torch.Tensor | None:
    if target_dim is None:
        log_problem("build_video_feats_batch", "target_dim not set")
        return None

    vecs: list[torch.Tensor] = []
    for idx, op in enumerate(openpose_list):
        v = build_video_feat_single(op, temporal_mode=temporal_mode, use_conf=use_conf, norm=norm)
        if v is None or v.numel() == 0:
            keys = list(op.keys()) if isinstance(op, dict) else None
            tname = type(op).__name__ if op is not None else "None"
            log_problem("build_video_feats_batch", f"invalid sample at idx={idx}",
                        extra={"type": tname, "keys": keys})
            return None
        vecs.append(_pad_trunc_1d(v, target_dim))
    try:
        out = torch.stack(vecs, dim=0)
    except Exception as e:
        log_problem("build_video_feats_batch", f"stack failed: {e.__class__.__name__}", extra={"B": len(vecs)})
        return None
    if device is not None:
        out = out.to(device)
    return out

### ALL OPENSMILE FUNCTIONALITY


def opensmile_to_framewise(d: Dict | None) -> torch.Tensor | None:
    if d is None or not isinstance(d, dict):
        log_problem("opensmile_to_framewise", "input is None or not dict")
        return None
    if "features" not in d:
        log_problem("opensmile_to_framewise", "missing 'features'", extra={"keys": list(d.keys())})
        return None

    x = d["features"]
    if x is None:
        log_problem("opensmile_to_framewise", "'features' is None")
        return None
    try:
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)
        x = x.float()
    except Exception as e:
        log_problem("opensmile_to_framewise", f"tensor convert failed: {e.__class__.__name__}")
        return None
    if x.numel() == 0:
        log_problem("opensmile_to_framewise", "'features' empty")
        return None
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if x.ndim != 2:
        log_problem("opensmile_to_framewise", f"bad shape {tuple(x.shape)} (want [T,D] or [D])")
        return None
    return x

def build_audio_feat_single(opensmile_dict: Dict | None,
                            temporal_mode: PoolMode = "none",
                            norm: str | None = None) -> torch.Tensor | None:
    seq = opensmile_to_framewise(opensmile_dict)
    if seq is None:
        return None
    v = pool_temporal(seq, mode=temporal_mode)
    if v is None:
        return None
    return _maybe_normalize(v, norm)

def build_audio_feats_batch(opensmile_list: Iterable[Dict | None],
                            device: torch.device | None = None,
                            temporal_mode: PoolMode = "none",
                            norm: str | None = None,
                            target_dim: int | None = None) -> torch.Tensor | None:
    if target_dim is None:
        log_problem("build_audio_feats_batch", "target_dim not set")
        return None

    vecs: list[torch.Tensor] = []
    for idx, d in enumerate(opensmile_list):
        v = build_audio_feat_single(d, temporal_mode=temporal_mode, norm=norm)
        if v is None or v.numel() == 0:
            keys = list(d.keys()) if isinstance(d, dict) else None
            tname = type(d).__name__ if d is not None else "None"
            log_problem("build_audio_feats_batch", f"invalid sample at idx={idx}",
                        extra={"type": tname, "keys": keys})
            return None
        vecs.append(_pad_trunc_1d(v, target_dim))
    try:
        out = torch.stack(vecs, dim=0)
    except Exception as e:
        log_problem("build_audio_feats_batch", f"stack failed: {e.__class__.__name__}", extra={"B": len(vecs)})
        return None
    if device is not None:
        out = out.to(device)
    return out