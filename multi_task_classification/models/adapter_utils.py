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
    rla_hidden: int,
    p_moddrop_video: float,
    p_moddrop_audio: float,
    d_video_feat: Optional[int] = None,
    d_audio_feat: Optional[int] = None,
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
            hidden=rla_hidden,
            p_moddrop=p_moddrop_video
        )

    if use_rla_audio:
        if d_audio_feat is None:
            raise ValueError("USE_RLA_AUDIO=True but d_audio_feat was not provided")
        audio_adapter = ResidualLogitAdapter(
            domain_id_to_global_indices,
            feat_key="audio_feats",
            feat_dim=d_audio_feat,
            hidden=rla_hidden,
            p_moddrop=p_moddrop_audio
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

def _pool_temporal(x: torch.Tensor,
                   mode: Literal["mean", "meanstd", "meanstdp25p75"] = "meanstd") -> torch.Tensor:
    
    # so essentially x looks like [T, K*C]
    # and then when we take the mean along dim=0, we are taking the average of K*C feature values across T frames
    # so the output shape is [K*C]
    # similarly for std, p25, p75, etc.
    # but then depending on what we want to use i.e. "mean", "meanstd", "meanstdp25p75", the shape will differ
    # i.e. when we concatenate them, the shape will be from non cat [K*C], to cat 2 [2*K*C], or [4*K*C] respectively
    x = x.float()
    if x.ndim != 2:
        raise ValueError(f"Expected [T, D], got {tuple(x.shape)}")
    if mode == "mean":
        return x.mean(dim=0)
    if mode == "meanstd":
        m, s = x.mean(dim=0), x.std(dim=0)
        return torch.cat([m, s], dim=0)
    if mode == "meanstdp25p75":
        T = x.size(0)
        kth25 = max(1, int(0.25 * T))
        kth75 = max(1, int(0.75 * T))
        m, s = x.mean(dim=0), x.std(dim=0)
        p25 = x.kthvalue(kth25, dim=0).values
        p75 = x.kthvalue(kth75, dim=0).values
        return torch.cat([m, s, p25, p75], dim=0)
    raise ValueError(f"Unknown pooling mode: {mode}")

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
) -> torch.Tensor:
    """OpenPose dict -> [D_vec]."""
    seq = openpose_dict_to_framewise(openpose, use_conf=use_conf)  # [T, D_raw]
    return _pool_temporal(seq, mode=temporal_mode)                 # [D_vec]

def build_video_feats_batch(
    openpose_list: Iterable[Dict[str, torch.Tensor]],
    device: Optional[torch.device] = None,
    temporal_mode: Literal["mean", "meanstd", "meanstdp25p75"] = "meanstd",
    use_conf: bool = True,
) -> torch.Tensor:
    """List of OpenPose dicts -> [B, D_vec]."""

    raise Exception(f"DEBUG: Video Feats {openpose_list}")

    # openpose_list is basically the list of dictionaries, each dictionary corresponds to one sample in the batch
    # so openpose_list is of length B (the batch size)
    feats = [build_video_feat_single(op, temporal_mode=temporal_mode, use_conf=use_conf)
             for op in openpose_list]
    out = torch.stack(feats, dim=0)
    if device is not None:
        out = out.to(device)
    return out

# Audio: placeholder for now (return None to signal "not used")
def build_audio_feats_placeholder(*args, **kwargs):
    return None