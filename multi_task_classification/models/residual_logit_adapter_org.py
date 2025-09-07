# models/rla_adapters.py
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

@torch.no_grad()
def get_conf_from_local_logits(local_logits: torch.Tensor) -> torch.Tensor:
    p = F.softmax(local_logits, dim=-1) # softmax to obtain the probability
    p_max, _ = p.max(dim=-1, keepdim=True) #now obtain the top probaility for everything
    entropy = -(p * (p.clamp_min(1e-12)).log()).sum(-1, keepdim=True) # compute entropy or uncertainty for the whole amount of logits
    if p.shape[-1] >= 2:
        top2 = torch.topk(p, k=2, dim=-1).values
        margin = top2[:, :1] - top2[:, 1:2] # obtain the difference between the top 2 probabilities
    else:
        margin = torch.zeros_like(p_max)
    return torch.cat([p_max, entropy, margin], dim=-1)  # [B,3] # flatten

# models/rla_adapters.py
class ResidualLogitAdapter(nn.Module):
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
        alpha_init: float = 1.0,             # <<< NEW
    ):
        super().__init__()
        self.domain_id_to_global_indices = domain_id_to_global_indices
        self.feat_key = feat_key
        self.feat_dim = feat_dim
        self.hidden = hidden
        self.p_moddrop = p_moddrop

        self.use_ln = use_ln
        self.use_conf_gain = use_conf_gain
        if use_ln:
            self.ln_feats = nn.LayerNorm(feat_dim)
            self.ln_conf  = nn.LayerNorm(3)
        if use_conf_gain:
            self.conf_gain = nn.Parameter(torch.full((3,), conf_init_gain))

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

        # Per-domain learnable residual scale (configurable init)
        self.alphas = nn.Parameter(torch.full(
            (len(domain_id_to_global_indices),), float(alpha_init), dtype=torch.float
        ))

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
        
        # raise Exception(f"DEBUG: Video Feats {feats}, z base global Logits, {z_base_global}, Domain IDs {domain_ids}, Domain id to global indices map {self.domain_id_to_global_indices}")
        if feats is None:
            return z_base_global

        z_out = z_base_global.clone()

        # for a subset of all samples, drop the features for regulaization
        feats = self._maybe_drop(feats, train_mode)

        # Per-domain to handle variable K_d
        for d in domain_ids.unique(sorted=True).tolist():
            rows = (domain_ids == d).nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                continue

            # domain_id_to_global_indices is a list of lists, 
            # where each sublist contains the global class indices for that domain
            # so essentially we are indexing to obtain the global logits for that domain to begin with.
            cols = torch.as_tensor(self.domain_id_to_global_indices[d], device=z_out.device, dtype=torch.long)

            local_logits = z_out.index_select(0, rows).index_select(1, cols)  # [B_d, K_d]

            # raise Exception(f"DEBUG: Local Logits {local_logits}, shape {local_logits.shape}")

            # obtaining the confidence from the local logits
            c = get_conf_from_local_logits(local_logits)                      # [B_d, 3]

            # gather the rows of the feats that correspond to the current domain
            # (so that we can essentially pass them through the MLP to obtain the residuals)
            df = feats.index_select(0, rows)                                  # [B_d, D_feat]

            # NOTE: Putting the layer norm on the confidence as well as the features
            if self.use_ln:
                df = self.ln_feats(df)
                c = self.ln_conf(c)
            # NOTE: SCALING THE CONFIDENCE BY MULTIPLYING IT
            if self.use_conf_gain:
                c = c * self.conf_gain

            x = torch.cat([df, c], dim=-1)                                    # [B_d, D_feat+3]

            # alpha is a learnable scalar gate that lets training keep residuals small unless
            # beneficial, it will scale the signal of the mlp per the domain
            # dz_local is the actual residual that we will add to the global logits
            # k_d is the number of classes for that domain
            # so essentially its a representation of each amount added to the logits
            dz_local = self.mlps[d](x) * self.alphas[d]                       # [B_d, K_d]

            # raise(Exception(f"DEBUG: DZ Local {dz_local}, shape {dz_local.shape}, Alpha {self.alphas[d]}"))

            # z_out is the global logits, so essentially for we are appending the dz_local to 
            # the correct positions in the global logits corresponding to the same domain
            # rows.unsqueeze(-1) makes it [B_d, 1] (handles the sample indices within the batch)
            # cols.unsqueeze(0).expand(...) makes it [B_d, K_d] (handles the class indices within the global logits)
            # so we append to the batch
            z_out[rows.unsqueeze(-1), cols.unsqueeze(0).expand(rows.numel(), cols.numel())] += dz_local

        return z_out