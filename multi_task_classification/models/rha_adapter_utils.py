from .residual_hidden_adapter import ResidualHiddenAdapter  # NEW
from typing import List, Optional
import torch


def maybe_build_hidden_adapters(
    *,
    domain_id_to_global_indices,
    use_rha_video: bool,
    use_rha_audio: bool,
    rha_hidden_video: int,          # MLP hidden for video
    rha_hidden_audio: int,          # MLP hidden for audio
    p_moddrop_video: float,
    p_moddrop_audio: float,
    out_dim_hidden: int,            # backbone pooled hidden size H
    d_video_feat: Optional[int] = None,
    d_audio_feat: Optional[int] = None,
    # per-modality knobs
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
    Build hidden-space residual adapters per modality.
    Returns: (video_hidden_adapter, audio_hidden_adapter).
    """
    video_hidden_adapter = None
    audio_hidden_adapter = None

    if use_rha_video:
        if d_video_feat is None:
            raise ValueError("USE_RHA_VIDEO=True but d_video_feat was not provided")
        video_hidden_adapter = ResidualHiddenAdapter(
            domain_id_to_global_indices=domain_id_to_global_indices,
            feat_dim=int(d_video_feat),
            hidden=int(rha_hidden_video),
            out_dim=int(out_dim_hidden),
            p_moddrop=float(p_moddrop_video),
            use_ln=bool(video_use_ln),
            use_conf_gain=bool(video_use_conf_gain),
            conf_init_gain=float(video_conf_init_gain),
            alpha_init=float(video_alpha_init),
        )

    if use_rha_audio:
        if d_audio_feat is None:
            raise ValueError("USE_RHA_AUDIO=True but d_audio_feat was not provided")
        audio_hidden_adapter = ResidualHiddenAdapter(
            domain_id_to_global_indices=domain_id_to_global_indices,
            feat_dim=int(d_audio_feat),
            hidden=int(rha_hidden_audio),
            out_dim=int(out_dim_hidden),
            p_moddrop=float(p_moddrop_audio),
            use_ln=bool(audio_use_ln),
            use_conf_gain=bool(audio_use_conf_gain),
            conf_init_gain=float(audio_conf_init_gain),
            alpha_init=float(audio_alpha_init),
        )

    return video_hidden_adapter, audio_hidden_adapter


def apply_hidden_adapters(
    *,
    h_base: torch.Tensor,                         # [B,H] pooled hidden (pre-heads)
    domain_ids: torch.Tensor,                     # [B]
    prelim_global_logits: torch.Tensor,           # [B,C_global] for confidence slicing
    video_hidden_adapter: Optional[ResidualHiddenAdapter],
    audio_hidden_adapter: Optional[ResidualHiddenAdapter],
    video_feats: Optional[torch.Tensor] = None,   # [B, Dv] or None
    audio_feats: Optional[torch.Tensor] = None,   # [B, Da] or None
    train_mode: bool,
) -> torch.Tensor:
    """
    Adds hidden residuals from any present modality adapters.
    No-ops cleanly if an adapter or its feats are missing.
    """
    h = h_base
    B = h.size(0)

    if (video_hidden_adapter is not None
        and video_feats is not None
        and video_feats.numel() > 0
        and video_feats.size(0) == B):
        h = video_hidden_adapter(
            h_base=h,
            domain_ids=domain_ids,
            global_logits=prelim_global_logits,
            feats=video_feats,
            train_mode=train_mode,
        )

    if (audio_hidden_adapter is not None
        and audio_feats is not None
        and audio_feats.numel() > 0
        and audio_feats.size(0) == B):
        h = audio_hidden_adapter(
            h_base=h,
            domain_ids=domain_ids,
            global_logits=prelim_global_logits,
            feats=audio_feats,
            train_mode=train_mode,
        )

    return h
