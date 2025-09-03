# models/rla_adapters.py
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from adapter_utils import get_conf_from_local_logits

class _DomainResidualAdapterBase(nn.Module):
    """
    Shared: domain routing, confidence features, modality dropout,
            and per-domain residual MLPs built inline (no extra head class).
    Subclasses define: feature key + feature dim.
    """
    def __init__(self,
                 domain_id_to_global_indices: List[List[int]],
                 feat_dim: int,
                 hidden: int = 128,
                 p_moddrop: float = 0.3):
        super().__init__()
        self.domain_id_to_global_indices = domain_id_to_global_indices
        self.hidden = hidden
        self.p_moddrop = p_moddrop
        self._feat_key: Optional[str] = None  # set in subclass

        # Per-domain residual MLPs: [feat_dim + 3(conf)] -> K_d
        mlps = []
        for slots in domain_id_to_global_indices:
            k_d = len(slots)
            mlps.append(nn.Sequential(
                nn.Linear(feat_dim + 3, hidden),
                nn.ReLU(),
                nn.Linear(hidden, k_d),
            ))
        self.mlps = nn.ModuleList(mlps)

        # Per-domain learnable alpha scalars (keep residuals small unless useful)
        self.alphas = nn.Parameter(torch.ones(len(domain_id_to_global_indices), dtype=torch.float))

    def _get_feats(self, **kwargs) -> Optional[torch.Tensor]:
        # subclasses set self._feat_key ("video_feats" / "audio_feats")
        return kwargs.get(self._feat_key, None) if self._feat_key is not None else None

    def _maybe_drop(self, feats: Optional[torch.Tensor], train_mode: bool) -> Optional[torch.Tensor]:
        if feats is None or not train_mode or self.p_moddrop <= 0:
            return feats
        keep = (torch.rand(feats.size(0), device=feats.device) >= self.p_moddrop).float().unsqueeze(-1)
        return feats * keep

    def forward(self,
                z_base_global: torch.Tensor,   # [B, C_global]
                domain_ids: torch.Tensor,      # [B]
                train_mode: bool = False,
                **kwargs) -> torch.Tensor:
        feats = self._get_feats(**kwargs)
        if feats is None:
            return z_base_global

        z_out = z_base_global.clone()
        feats = self._maybe_drop(feats, train_mode)

        # Per-domain to handle variable K_d
        for d in domain_ids.unique(sorted=True).tolist():
            rows = (domain_ids == d).nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                continue
            cols = torch.as_tensor(self.domain_id_to_global_indices[d], device=z_out.device, dtype=torch.long)

            local_logits = z_out.index_select(0, rows).index_select(1, cols)    # [B_d, K_d]
            c = get_conf_from_local_logits(local_logits)                                            # [B_d, 3]
            df = feats.index_select(0, rows)                                   # [B_d, D_feat]

            x = torch.cat([df, c], dim=-1)                                     # [B_d, D_feat+3]
            dz_local = self.mlps[d](x) * self.alphas[d]                        # [B_d, K_d]

            z_out[rows.unsqueeze(-1), cols.unsqueeze(0).expand(rows.numel(), cols.numel())] += dz_local

        # obtain the z_out
        return z_out

class VideoResidualAdapter(_DomainResidualAdapterBase):
    def __init__(self, domain_id_to_global_indices: List[List[int]], d_video_feat: int, hidden: int = 128, p_moddrop: float = 0.3):
        super().__init__(domain_id_to_global_indices, feat_dim=d_video_feat, hidden=hidden, p_moddrop=p_moddrop)
        self._feat_key = "video_feats"  # dataloader must pass video_feats=[B, d_video]

class AudioResidualAdapter(_DomainResidualAdapterBase):
    def __init__(self, domain_id_to_global_indices: List[List[int]], d_audio_feat: int, hidden: int = 128, p_moddrop: float = 0.3):
        super().__init__(domain_id_to_global_indices, feat_dim=d_audio_feat, hidden=hidden, p_moddrop=p_moddrop)
        self._feat_key = "audio_feats"  # dataloader must pass audio_feats=[B, d_audio]