# models/bam_qa_wrap.py
"""
BAMQAClassifier: wraps MultiHeadOmniClassifier for BAM-augmented QA + classification training.

The trainer applies hidden-space BAM adapters externally, producing `video_pooled_bam` and/or
`audio_pooled_bam`.  This forward:
  1. Runs the backbone once to get hidden states.
  2. Computes pooled_base from the penultimate layer.
  3. Computes delta = (pooled_eff - pooled_base) from the BAM-fused pooled tensors.
  4. Injects delta into the last hidden layer and recomputes through lm_head.
  5. Computes teacher-forcing LM loss (if lm_labels provided).
  6. Computes classification logits from pooled_eff via domain heads.
"""
import torch
import torch.nn.functional as F
from .qwen2_5_omni_classifier_heads_decoder import MultiHeadOmniClassifier


class BAMQAClassifier(MultiHeadOmniClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.backbone.config.use_cache = False
        self.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    def forward(
        self,
        input_ids,
        attention_mask=None,
        domain_ids=None,
        lm_labels=None,
        *,
        video_pooled_bam: torch.Tensor | None = None,  # [B,H] or None
        audio_pooled_bam: torch.Tensor | None = None,  # [B,H] or None
        **kwargs
    ):
        """
        Single-pass forward for both LM (QA) and classification.
        Adapters are not called here; the trainer computes them and passes the
        already-fused pooled vectors via `video_pooled_bam` / `audio_pooled_bam`.
        Returns:
            {
                "cls_logits": [B, C_global],
                "lm_loss": scalar or None,
                "lm_output": HF output object with `logits` replaced by BAM-injected logits
            }
        """
        if domain_ids is None:
            raise ValueError("domain_ids must be provided")

        # 1) Backbone once
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            **kwargs,
        )

        hidden_states = out.hidden_states
        h_penult = hidden_states[-2]  # [B,T,H]

        # 2) pooled_base from penultimate layer
        if attention_mask is not None:
            pooled_base = (h_penult * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1, keepdim=True)
        else:
            pooled_base = h_penult.mean(dim=1)  # [B,H]

        B, H = pooled_base.size()
        device, dtype = pooled_base.device, pooled_base.dtype
        domain_ids = domain_ids.to(device)

        # 3) Effective pooled: audio_pooled_bam > video_pooled_bam > pooled_base
        pooled_eff = pooled_base
        if video_pooled_bam is not None:
            pooled_eff = video_pooled_bam.to(dtype)
        if audio_pooled_bam is not None:
            pooled_eff = audio_pooled_bam.to(dtype)

        # 4) Inject delta into the last hidden layer then recompute through lm_head
        delta = (pooled_eff - pooled_base).to(h_penult.dtype)  # [B, H]
        h_last_mod = hidden_states[-1] + delta.unsqueeze(1)    # [B, T, H]

        maybe_model = getattr(self.backbone, "model", None)
        if maybe_model is not None and hasattr(maybe_model, "norm"):
            h_for_lm = maybe_model.norm(h_last_mod)
        else:
            h_for_lm = h_last_mod

        lm_head = getattr(self.backbone, "lm_head", None)
        if lm_head is None:
            raise RuntimeError("Backbone has no lm_head; cannot compute LM logits after BAM injection.")
        lm_logits = lm_head(h_for_lm)  # [B,T,V]

        # 5) Teacher-forcing LM loss
        lm_loss = None
        if lm_labels is not None:
            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_labels = lm_labels[:, 1:].contiguous()
            lm_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        # 6) Classification logits from pooled_eff (QA rows stay neg_inf)
        cls_logits = self._heads_from_pooled(pooled_eff, domain_ids)
        # Zero out QA rows (domain_id == -1) — _heads_from_pooled skips unknown domains,
        # but -1 is not a valid domain index so those rows remain neg_inf already.

        out.logits = lm_logits
        out.loss = lm_loss

        return {"cls_logits": cls_logits, "lm_loss": lm_loss, "lm_output": out}
