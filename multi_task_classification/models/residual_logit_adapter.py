# models/rla_adapters.py
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

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

class ResidualLogitAdapter(nn.Module):
    """
    Single, modality-agnostic residual adapter.
    You specify which feature tensor to read via `feat_key` (only used by caller)
    and the input dimensionality via `feat_dim`.
    """
    def __init__(
        self,
        domain_id_to_global_indices: List[List[int]],
        feat_key: str,           # e.g., "video_feats" or "audio_feats" (for clarity/logging)
        feat_dim: int,
        hidden: int = 128,
        p_moddrop: float = 0.3,
    ):
        super().__init__()
        self.domain_id_to_global_indices = domain_id_to_global_indices
        self.feat_key = feat_key
        self.feat_dim = feat_dim
        self.hidden = hidden
        self.p_moddrop = p_moddrop

        # One residual MLP per domain: [feat_dim + 3(conf)] -> K_d
        mlps = []
        for slots in domain_id_to_global_indices:
            k_d = len(slots)
            mlps.append(
                nn.Sequential(
                    nn.Linear(feat_dim + 3, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, k_d),
                )
            )
        self.mlps = nn.ModuleList(mlps)

        # Per-domain learnable scale on the residual
        self.alphas = nn.Parameter(torch.ones(len(domain_id_to_global_indices), dtype=torch.float))

    def _maybe_drop(self, feats: Optional[torch.Tensor], train_mode: bool) -> Optional[torch.Tensor]:
        if feats is None or not train_mode or self.p_moddrop <= 0:
            return feats
        keep = (torch.rand(feats.size(0), device=feats.device) >= self.p_moddrop).float().unsqueeze(-1)
        return feats * keep

    def forward(
        self,
        z_base_global: torch.Tensor,           # [B, C_global]
        domain_ids: torch.Tensor,              # [B]
        feats: Optional[torch.Tensor] = None,  # [B, D_feat]  <<—— direct tensor that we pass into this
        train_mode: bool = False,
    ) -> torch.Tensor:
        
        raise Exception(f"DEBUG: Video Feats {feats}, z base global Logits, {z_base_global}, Domain IDs {domain_ids}, Domain id to global indices map {self.domain_id_to_global_indices}")
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

            local_logits = z_out.index_select(0, rows).index_select(1, cols)  # [B_d, K_d]
            c = get_conf_from_local_logits(local_logits)                      # [B_d, 3]
            df = feats.index_select(0, rows)                                  # [B_d, D_feat]

            x = torch.cat([df, c], dim=-1)                                    # [B_d, D_feat+3]
            dz_local = self.mlps[d](x) * self.alphas[d]                       # [B_d, K_d]

            z_out[rows.unsqueeze(-1), cols.unsqueeze(0).expand(rows.numel(), cols.numel())] += dz_local

        return z_out