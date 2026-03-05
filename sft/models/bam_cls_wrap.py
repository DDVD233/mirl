# models/bam_cls_wrap.py
"""
BAMClassifier: wraps MultiHeadOmniClassifier for BAM-augmented classification training.

The trainer computes BAM-fused pooled hidden states externally and passes them in via
the `pooled` keyword argument.  If `pooled` is None the model falls back to computing it
from the backbone as usual.
"""
import torch
from .qwen2_5_omni_classifier_heads_decoder import MultiHeadOmniClassifier


class BAMClassifier(MultiHeadOmniClassifier):
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        domain_ids=None,
        *,
        pooled: torch.Tensor | None = None,  # pre-fused pooled from trainer (post-BAM)
        **kwargs
    ):
        if domain_ids is None:
            raise ValueError("domain_ids must be provided")

        if pooled is None:
            out = self.backbone(
                input_ids=input_ids, attention_mask=attention_mask,
                output_hidden_states=True, **kwargs
            )
            h = out.hidden_states[-2]  # [B,T,H]
            if attention_mask is not None:
                pooled = (h * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1, keepdim=True)
            else:
                pooled = h.mean(dim=1)  # [B,H]

        logits = self._heads_from_pooled(pooled, domain_ids)
        return logits, pooled
