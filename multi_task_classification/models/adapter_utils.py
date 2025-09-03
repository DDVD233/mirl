# models/adapter_utils.py
from typing import Optional, Tuple
import torch
import torch.nn.functional as F

from .rla_adapters import VideoResidualAdapter, AudioResidualAdapter

@torch.no_grad()
def get_conf_from_local_logits(local_logits: torch.Tensor) -> torch.Tensor:
    p = F.softmax(local_logits, dim=-1)
    p_max, _ = p.max(dim=-1, keepdim=True)
    entropy = -(p * (p.clamp_min(1e-12)).log()).sum(-1, keepdim=True)
    if p.shape[-1] >= 2:
        top2 = torch.topk(p, k=2, dim=-1).values
        margin = top2[:, :1] - top2[:, 1:2]
    else:
        margin = torch.zeros_like(p_max)
    return torch.cat([p_max, entropy, margin], dim=-1)  # [B,3]

def maybe_build_adapters(
    *,
    domain_id_to_global_indices,
    train_dataset,
    use_rla_video: bool,
    use_rla_audio: bool,
    rla_hidden: int,
    p_moddrop_video: float,
    p_moddrop_audio: float,
    rla_stage: str,
    base_model,                 # nn.Module
    d_video_feat: Optional[int] = None,
    d_audio_feat: Optional[int] = None,
) -> Tuple[Optional[VideoResidualAdapter], Optional[AudioResidualAdapter], int, int]:
    """
    Builds adapters once, infers feature dims if not provided, optionally freezes base (residual_only).
    Returns: (video_adapter, audio_adapter, d_video_feat, d_audio_feat)
    """
    video_adapter = None
    audio_adapter = None

    if not (use_rla_video or use_rla_audio):
        return None, None, d_video_feat or 0, d_audio_feat or 0

    # Infer dims from a sample if needed
    sample = train_dataset[0]
    if use_rla_video and d_video_feat is None:
        v = sample.get("video_feats", None)
        if v is None:
            raise KeyError("USE_RLA_VIDEO=True but dataset item lacks 'video_feats'")
        d_video_feat = int(v.numel()) if torch.is_tensor(v) else int(v.size)

    if use_rla_audio and d_audio_feat is None:
        a = sample.get("audio_feats", None)
        if a is None:
            raise KeyError("USE_RLA_AUDIO=True but dataset item lacks 'audio_feats'")
        d_audio_feat = int(a.numel()) if torch.is_tensor(a) else int(a.size)

    if use_rla_video:
        video_adapter = VideoResidualAdapter(
            domain_id_to_global_indices, d_video_feat=d_video_feat,
            hidden=rla_hidden, p_moddrop=p_moddrop_video
        )

    if use_rla_audio:
        audio_adapter = AudioResidualAdapter(
            domain_id_to_global_indices, d_audio_feat=d_audio_feat,
            hidden=rla_hidden, p_moddrop=p_moddrop_audio
        )

    # Stage semantics
    if rla_stage == "residual_only":
        for p in base_model.parameters():
            p.requires_grad = False

    return video_adapter, audio_adapter, d_video_feat, d_audio_feat


def apply_adapters(
    logits: torch.Tensor,
    domain_ids: torch.Tensor,
    batch: dict,
    *,
    video_adapter: Optional[VideoResidualAdapter],
    audio_adapter: Optional[AudioResidualAdapter],
    train_mode: bool
) -> torch.Tensor:
    """
    Adds residuals in logit space using whatever adapters are present.
    Safe if feats are missing: will just return logits unchanged.
    """
    z = logits
    if (video_adapter is not None) and ("video_feats" in batch):
        z = video_adapter(z, domain_ids, train_mode=train_mode, video_feats=batch["video_feats"])
    if (audio_adapter is not None) and ("audio_feats" in batch):
        z = audio_adapter(z, domain_ids, train_mode=train_mode, audio_feats=batch["audio_feats"])
    return z