# models/residual_hidden_adapter.py
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from .residual_logit_adapter import get_conf_from_local_logits

class ResidualHiddenAdapter(nn.Module):
    """
    Residual adapter at the pooled hidden (penultimate) state.
    Per-domain routing is used only to compute confidence features from *local* logits.
    The residual is added in H-dim, before the classifier head(s).
    """
    def __init__(
        self,
        *,
        domain_id_to_global_indices: List[List[int]],
        feat_dim: int,           # side feature dim (video/audio)
        hidden: int,             # MLP hidden
        out_dim: int,            # backbone hidden size H
        p_moddrop: float = 0.3,
        use_ln: bool = False,
        use_conf_gain: bool = False,
        conf_init_gain: float = 3.0,
        alpha_init: float = 1.0,
    ):
        super().__init__()
        self.domain_id_to_global_indices = domain_id_to_global_indices
        self.feat_dim = feat_dim
        self.p_moddrop = p_moddrop
        self.use_ln = use_ln
        self.use_conf_gain = use_conf_gain
        self._printed_zero_conf_msg = False   # flag

        if use_ln:
            self.ln_feats = nn.LayerNorm(feat_dim)
            self.ln_conf  = nn.LayerNorm(3)
        if use_conf_gain:
            self.conf_gain = nn.Parameter(torch.full((3,), conf_init_gain))

        self.mlp = nn.Sequential(
            nn.Linear(feat_dim + 3, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )
        self.alpha = nn.Parameter(torch.tensor([float(alpha_init)]))  # shape (1,)


    def _maybe_drop(self, feats: Optional[torch.Tensor], train_mode: bool) -> Optional[torch.Tensor]:
        if feats is None or not train_mode or self.p_moddrop <= 0:
            return feats
        keep = (torch.rand(feats.size(0), device=feats.device) >= self.p_moddrop).float().unsqueeze(-1)
        return feats * keep

    def _conf_feats(self, global_logits: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        # build a [B,3] confidence vector by slicing *local* logits per sample's domain
        # check if none (for QA)

        if global_logits is None:
            B = domain_ids.size(0)
        else:
            B = global_logits.size(0)
        
        device = global_logits.device
        
        out = torch.zeros(B, 3, device=device, dtype=global_logits.dtype)
        # return out # always return zero confidence for QA

        if (global_logits is None) or ((domain_ids == -1).all()):
            if not self._printed_zero_conf_msg:
                print("Returning zero confidence as this is QA (i.e. all domain_ids == -1)")
                self._printed_zero_conf_msg = True
            return out
        
        unique = domain_ids.unique(sorted=True).tolist()
        for d in unique:
            if d == -1: continue  # skip no-domain samples
            rows = (domain_ids == d).nonzero(as_tuple=True)[0]
            if rows.numel() == 0: continue
            cols = torch.as_tensor(self.domain_id_to_global_indices[d], device=device, dtype=torch.long)
            local = global_logits.index_select(0, rows).index_select(1, cols)  # [B_d, K_d]
            c = get_conf_from_local_logits(local.float()).to(global_logits.dtype)
            out[rows] = c
        return out

    def forward(
        self,
        h_base: torch.Tensor,                # [B,H] pooled penultimate
        domain_ids: torch.Tensor,            # [B]
        global_logits: torch.Tensor,         # [B,C_global] (preliminary logits for confidence)
        feats: Optional[torch.Tensor],       # [B,D_feat] (video/audio pooled)
        train_mode: bool = False,
    ) -> torch.Tensor:
        if feats is None:
            return h_base
        feats = self._maybe_drop(feats, train_mode)
        c = self._conf_feats(global_logits, domain_ids)  # [B,3]

        if self.use_ln:
            feats = self.ln_feats(feats)
            c = self.ln_conf(c)
        if self.use_conf_gain:
            c = c * self.conf_gain

        x = torch.cat([feats, c], dim=-1)   # [B, D_feat+3]
        delta = self.mlp(x) * self.alpha.view(1, 1)

        return h_base + delta
