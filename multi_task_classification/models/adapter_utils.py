# models/adapter_utils.py
from typing import Optional, Tuple, Iterable
import torch
import torch.nn.functional as F
# models/feature_builders.py
from typing import Dict, Literal
import torch
from .residual_logit_adapter import ResidualLogitAdapter

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

def apply_adapters(
    logits: torch.Tensor,
    domain_ids: torch.Tensor,
    *,
    video_adapter: Optional[ResidualLogitAdapter],
    audio_adapter: Optional[ResidualLogitAdapter],
    video_feats: Optional[torch.Tensor] = None,   # <<—— direct tensors
    audio_feats: Optional[torch.Tensor] = None,
    train_mode: bool,
) -> torch.Tensor:
    """
    Add residuals in logit space using whatever adapters are present.
    If feats are None or adapter is None, logits are returned unchanged.
    """
    z = logits 
    if (video_adapter is not None) and (video_feats is not None):
        z = video_adapter(z, domain_ids, feats=video_feats, train_mode=train_mode)
    if (audio_adapter is not None) and (audio_feats is not None):
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
def _pad_trunc_1d(x: torch.Tensor, target_dim: int) -> torch.Tensor:
    """x: [D_var] -> [D_target] by right-pad with zeros or truncate."""
    D = x.numel()
    if D == target_dim:
        return x
    if D > target_dim:
        return x[:target_dim]
    out = x.new_zeros(target_dim)
    out[:D] = x
    return out


def pool_temporal(x: torch.Tensor, mode: PoolMode = "meanstd") -> torch.Tensor:
    """
    x: [T, D]  (for functionals, T=1)
    Modes:
      - "none": requires T==1, returns [D] unchanged
      - "mean": returns [D]
      - "meanstd": returns [2D]  (concat mean, std)
      - "meanstdp25p75": returns [4D] (mean, std, p25, p75)
    """
    x = x.float()
    if x.ndim != 2:
        raise ValueError(f"Expected [T, D], got {tuple(x.shape)}")

    T, D = x.shape
    if mode == "none":
        if T != 1:
            raise ValueError(f"pool_temporal(mode='none') expects T==1, got T={T}")
        return x.squeeze(0)  # [D]

    if mode == "mean":
        return x.mean(dim=0)  # [D]

    if mode == "meanstd":
        m = x.mean(dim=0)
        # use unbiased=False to avoid NaNs when T==1
        s = x.std(dim=0, unbiased=False)
        return torch.cat([m, s], dim=0)  # [2D]

    if mode == "meanstdp25p75":
        # stats
        m = x.mean(dim=0)
        s = x.std(dim=0, unbiased=False)
        # quantiles via kthvalue (consistent with your video code)
        kth25 = max(1, int(0.25 * T))
        kth75 = max(1, int(0.75 * T))
        p25   = x.kthvalue(kth25, dim=0).values
        p75   = x.kthvalue(kth75, dim=0).values
        return torch.cat([m, s, p25, p75], dim=0)  # [4D]

    raise ValueError(f"Unknown pooling mode: {mode}")

# ALL OPENPOSE FUNCTIONALITY

def openpose_dict_to_framewise(data: Dict[str, torch.Tensor], use_conf: bool = True) -> torch.Tensor:
    """
    data may include any of: 'pose','face','left_hand','right_hand', each [T, K, 3] (x,y,conf)
    returns [T, D_raw] by concatenating parts per frame.
    """
    chunks = []
    for k in ("pose", "face", "left_hand", "right_hand"):
        if k not in data:
            continue
        t = data[k]
        if not torch.is_tensor(t):
            t = torch.as_tensor(t)
        t = t.float()
        if t.ndim != 3:
            raise ValueError(f"OpenPose part '{k}' must be [T,K,3], got {tuple(t.shape)}")
        if not use_conf:
            t = t[..., :2]  # drop the confidence

        # Flattening the last two dimensions (K, C) into a single dimension (K*C)
        # i.e. we are just putting them side by side
        chunks.append(t.reshape(t.shape[0], -1))  # [T, K*C]
    if not chunks:
        raise KeyError("No valid OpenPose keys found in dict.")
    
    # concatenate them and put them side by side (along the last dimension)
    # e.g., if we have pose, face, left_hand, right_hand, then the final shape is [T, (K1*C1)+(K2*C2)+(K3*C3)+(K4*C4)]
    return torch.cat(chunks, dim=-1)  # [T, D_raw_sum]

def build_video_feat_single(
    openpose: Dict[str, torch.Tensor],
    temporal_mode: Literal["mean", "meanstd", "meanstdp25p75"] = "meanstd",
    use_conf: bool = True,
    norm: Optional[str] = None,  
) -> torch.Tensor:
    """OpenPose dict -> [D_vec]."""
    seq = openpose_dict_to_framewise(openpose, use_conf=use_conf)  # [T, D_raw]
    v = pool_temporal(seq, mode=temporal_mode)  # [D | 2D | 4D]
    return _maybe_normalize(v, norm)                 # [D_vec]

def build_video_feats_batch(
    openpose_list: Iterable[Dict[str, torch.Tensor]],
    device: Optional[torch.device] = None,
    temporal_mode: Literal["mean", "meanstd", "meanstdp25p75"] = "meanstd",
    use_conf: bool = True,
    norm: Optional[str] = None,  # NEW
    target_dim: int = None,   # <-- NEW: pass your D_VIDEO_FEAT (e.g., 3318)
) -> torch.Tensor:
    """
    List of OpenPose dicts -> [B, target_dim].
    We pool each sample to a 1-D vector and pad/truncate to `target_dim`.
    If `target_dim` is None, we use the max length seen in the batch.
    """
    # Build raw per-sample vectors (variable length)
    raw = [build_video_feat_single(op, temporal_mode=temporal_mode, use_conf=use_conf, norm=norm)
           for op in openpose_list]

    if target_dim is None:
        raise ValueError("target_video_feature_dim must be provided to ensure consistent feature size across batches")

    feats = [_pad_trunc_1d(v, target_dim) for v in raw]   # all [target_dim]
    out = torch.stack(feats, dim=0)                       # [B, target_dim]
    if device is not None:
        out = out.to(device)
    return out

### ALL OPENSMILE FUNCTIONALITY

def opensmile_to_framewise(d: Dict) -> torch.Tensor:
    if "features" not in d:
        raise KeyError("OpenSmile dict missing 'features'")
    
    # TODO: depending on this, we can also extract the different types of functionals (i.e. types of features)
    # TODO: from the feature set.
    x = d["features"]
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)
    x = x.float()
    if x.ndim == 1:
        x = x.unsqueeze(0)  # [1, D]
    if x.ndim != 2:
        raise ValueError(f"'features' must be [T, D] or [D], got {tuple(x.shape)}")
    return x  # [T, D]

def _maybe_normalize(v: torch.Tensor, norm: str | None) -> torch.Tensor:
    # NORMALIZE THE AUDIO FEATURES TO PREVENT NUMERICAL ISSUES; i.e. the residuals from blowing up or getting swamped
    if norm is None:  return v
    if norm == "l2":  return v / v.norm(p=2).clamp_min(1e-6)
    if norm == "zscore":
        m, s = v.mean(), v.std(unbiased=False).clamp_min(1e-6)
        return (v - m) / s
    raise ValueError(f"Unknown norm: {norm}")

def build_audio_feat_single(opensmile_dict: Dict,
                            temporal_mode: PoolMode = "none",
                            norm: str | None = None) -> torch.Tensor:
    # we assume that the opensmile_dict contains the "features" key
    # which is basically in the format of [1, 6373]
    seq = opensmile_to_framewise(opensmile_dict)         # [T, D]
    # we take the whole sequence from features
    v   = pool_temporal(seq, mode=temporal_mode)          # [D | 2D | 4D] # NONE FOR NOW
    return _maybe_normalize(v, norm)

def build_audio_feats_batch(opensmile_list: Iterable[Dict],
                            device: torch.device | None = None,
                            temporal_mode: PoolMode = "none",
                            norm: str | None = None,
                            target_dim: int | None = None) -> torch.Tensor:
    # opensmile list is a list of all the samples basically within the batch
    # each item in the list is a dict that contains the opensmile features for that sample
    raw = [build_audio_feat_single(d, temporal_mode=temporal_mode, norm=norm)
           for d in opensmile_list]
    if target_dim is None:
        raise ValueError("Set D_AUDIO_FEAT in config; target_dim is required.")
    feats = [_pad_trunc_1d(v, target_dim) for v in raw]
    out = torch.stack(feats, dim=0)
    if device is not None:
        out = out.to(device)
    return out