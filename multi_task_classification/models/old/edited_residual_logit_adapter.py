# models/rla_adapters.py (drop-in replacement for class ResidualLogitAdapter)

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
    Memory-light RLA:
      - ONE shared trunk: (feat_dim+3) -> hidden
      - ONE shared head: hidden -> max_k across domains
      - Per-domain alphas (1 scalar per domain)
    We slice head outputs to [ :K_d ] for each domain.
    """
    def __init__(
        self,
        domain_id_to_global_indices: List[List[int]],
        feat_key: str,
        feat_dim: int,
        hidden: int = 128,
        p_moddrop: float = 0.3,
        use_ln: bool = False,
        use_conf_gain: bool = False,
        conf_init_gain: float = 3.0,
        alpha_init: float = 1.0,
    ):
        super().__init__()
        self.domain_id_to_global_indices = domain_id_to_global_indices
        self.feat_key = feat_key
        self.feat_dim = feat_dim
        self.hidden = hidden
        self.p_moddrop = p_moddrop

        # fixed head width (shared across domains)
        self.max_k = max(len(slots) for slots in domain_id_to_global_indices)

        self.use_ln = use_ln
        self.use_conf_gain = use_conf_gain
        if use_ln:
            self.ln_feats = nn.LayerNorm(feat_dim)
            self.ln_conf  = nn.LayerNorm(3)
        if use_conf_gain:
            self.conf_gain = nn.Parameter(torch.full((3,), conf_init_gain))

        # shared trunk + head
        self.trunk = nn.Sequential(
            nn.Linear(feat_dim + 3, hidden),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden, self.max_k)

        # small per-domain scalar gates
        self.alphas = nn.Parameter(
            torch.full((len(domain_id_to_global_indices),), float(alpha_init), dtype=torch.float)
        )

    def _maybe_drop(self, feats: Optional[torch.Tensor], train_mode: bool) -> Optional[torch.Tensor]:
        if feats is None or not train_mode or self.p_moddrop <= 0:
            return feats
        keep = (torch.rand(feats.size(0), device=feats.device) >= self.p_moddrop).float().unsqueeze(-1)
        return feats * keep

    def forward(
        self,
        z_base_global: torch.Tensor,           # [B, C_global]
        domain_ids: torch.Tensor,              # [B]
        feats: Optional[torch.Tensor] = None,  # [B, D_feat]
        train_mode: bool = False,
    ) -> torch.Tensor:
        if feats is None:
            return z_base_global

        z_out = z_base_global.clone()  # keep original for autograd safety
        feats = self._maybe_drop(feats, train_mode)

        # Process each domain present in the batch
        for d in domain_ids.unique(sorted=True).tolist():
            rows = (domain_ids == d).nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                continue

            cols = torch.as_tensor(self.domain_id_to_global_indices[d], device=z_out.device, dtype=torch.long)
            # current local logits
            local_logits = z_out.index_select(0, rows).index_select(1, cols)  # [B_d, K_d]
            Kd = local_logits.size(1)

            # confidences from *current* local logits
            c = get_conf_from_local_logits(local_logits)                      # [B_d, 3]
            df = feats.index_select(0, rows)                                  # [B_d, D_feat]

            if self.use_ln:
                df = self.ln_feats(df)
                c  = self.ln_conf(c)
            if self.use_conf_gain:
                c = c * self.conf_gain

            x = torch.cat([df, c], dim=-1)                                    # [B_d, D_feat+3]
            h = self.trunk(x)                                                 # [B_d, hidden]
            dz_full = self.head(h)                                            # [B_d, max_k]
            dz_local = dz_full[:, :Kd] * self.alphas[d]                       # [B_d, K_d]

            # add residuals back (fused accumulate avoids big temporaries)
            # (rows[:,None], cols[None,:]) broadcast to [B_d, K_d]
            z_out.index_put_(
                (rows.unsqueeze(-1), cols.unsqueeze(0).expand(rows.numel(), Kd)),
                dz_local,
                accumulate=True,  # +=
            )  # doc: index_put_(..., accumulate=True) adds into place
        return z_out