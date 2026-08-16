# models/bam_wrapped_qwen3_vl.py
"""
BAM-augmented Qwen3-VL wrappers (parallel to bam_wrapped_qwen.py for Omni).

  BAMVLBase — shared base (3 adapter slots, penultimate pooling, adapter application)
  BAMVLCLS  — classification-only forward
  BAMVLQA   — QA forward with teacher-forcing LM loss; injects the pooled BAM delta
              into the (post-final-norm) last hidden state, then re-runs lm_head.

Differences vs the Omni wrappers:
  * Three side-channel adapters (facial, pose, audio) instead of two (video, audio),
    matching the extracted ChildPlay modalities (MediaPipe FaceMesh / COCO-17 pose /
    eGeMAPS audio).
  * Native video is passed through to the backbone (pixel_values_videos /
    video_grid_thw), so the VL tower and BAM deltas fuse in one forward.
  * The final norm / lm_head are resolved through MultiHeadVLClassifier's resolvers,
    which target Qwen3-VL's backbone.model.language_model.norm (backbone.model.norm
    does not exist on Qwen3-VL).

Adapters are assigned as nn.Module submodules by the trainer BEFORE
accelerator.prepare(), so FSDP/DDP wraps model + adapters as one unit. Missing
modalities (None feats / None adapter) are skipped silently.
"""
import torch
import torch.nn.functional as F

from .qwen3_vl_classifier_heads_decoder import MultiHeadVLClassifier


class BAMVLBase(MultiHeadVLClassifier):
    """Shared base: adapter slots, penultimate pooling, sequential adapter residuals."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Trainer assigns nn.Module instances here before accelerator.prepare(),
        # which auto-registers them as submodules.
        self.facial_adapter = None
        self.pose_adapter = None
        self.audio_adapter = None

    def _compute_dtype(self) -> torch.dtype:
        """Dtype the custom head (adapters / final-norm / lm_head) runs in — the lm_head
        weight dtype (bf16). Used to keep the head consistent when there is no autocast
        (mixed_precision="no"), since Qwen3.5's fp32-preserving RMSNorm / linear-attention
        sublayers can emit fp32 hidden states."""
        return self._resolve_lm_head().weight.dtype

    def _pool_penultimate(self, h, attention_mask) -> torch.Tensor:
        """Attention-masked mean of the penultimate (pre-norm) hidden layer [B,T,H] -> [B, H].
        `h` is the penultimate tensor (already cast to the compute dtype by the caller)."""
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).to(h.dtype)
            return (h * mask).sum(1) / mask.sum(1).clamp_min(1.0)
        return h.mean(dim=1)

    def _apply_adapters(self, pooled_base, facial_feats, pose_feats, audio_feats, train_mode,
                        facial_mask=None, pose_mask=None, audio_mask=None):
        """
        Apply facial, pose, audio adapter deltas as sequential residuals.

        Two levels of "absent":
          * adapter is None or feats is None  -> the whole modality is off for this batch.
          * per-sample mask (0/1, shape [B])  -> the feats tensor is zero-filled for rows
            whose modality file was missing; the mask zeros their delta so the adapter's
            bias does not inject a spurious constant for absent samples.
        """
        h = pooled_base
        for adapter, feats, mask in (
            (self.facial_adapter, facial_feats, facial_mask),
            (self.pose_adapter, pose_feats, pose_mask),
            (self.audio_adapter, audio_feats, audio_mask),
        ):
            if adapter is not None and feats is not None:
                delta = adapter(feats.to(h.dtype), train_mode=train_mode)
                if delta is not None:
                    if mask is not None:
                        delta = delta * mask.to(delta.dtype).view(-1, 1)
                    h = h + delta
        return h

    @staticmethod
    def _collect_mm_kwargs(pixel_values_videos, video_grid_thw, pixel_values, image_grid_thw,
                           mm_token_type_ids=None):
        mm = {}
        if pixel_values_videos is not None:
            mm["pixel_values_videos"] = pixel_values_videos
        if video_grid_thw is not None:
            mm["video_grid_thw"] = video_grid_thw
        if pixel_values is not None:
            mm["pixel_values"] = pixel_values
        if image_grid_thw is not None:
            mm["image_grid_thw"] = image_grid_thw
        # Qwen3.5 M-RoPE requires a per-token type mask (0=text/1=image/2=video) whenever
        # image/video grids are passed; the trainer derives it from the fed input_ids.
        if mm_token_type_ids is not None:
            mm["mm_token_type_ids"] = mm_token_type_ids
        return mm


class BAMVLCLS(BAMVLBase):
    """BAM-augmented multi-head classifier (CLS only). Returns (logits, pooled_eff)."""

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        domain_ids=None,
        *,
        facial_feats=None,
        pose_feats=None,
        audio_feats=None,
        facial_mask=None,
        pose_mask=None,
        audio_mask=None,
        train_mode=False,
        pixel_values_videos=None,
        video_grid_thw=None,
        pixel_values=None,
        image_grid_thw=None,
        mm_token_type_ids=None,
        **kwargs,
    ):
        if domain_ids is None:
            raise ValueError("domain_ids required")

        mm = self._collect_mm_kwargs(pixel_values_videos, video_grid_thw, pixel_values, image_grid_thw,
                                     mm_token_type_ids=mm_token_type_ids)
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            **mm,
            **kwargs,
        )
        # Cast the penultimate state to the head compute dtype (no autocast under
        # mixed_precision="no"; backbone may emit fp32 states while heads/adapters are bf16).
        pooled_base = self._pool_penultimate(out.hidden_states[-2].to(self._compute_dtype()), attention_mask)
        pooled_eff = self._apply_adapters(pooled_base, facial_feats, pose_feats, audio_feats, train_mode,
                                          facial_mask=facial_mask, pose_mask=pose_mask, audio_mask=audio_mask)
        logits = self._heads_from_pooled(pooled_eff, domain_ids)
        return logits, pooled_eff


class BAMVLQA(BAMVLBase):
    """
    BAM-augmented Qwen3-VL with QA (teacher-forcing LM) training.

    Returns {"cls_logits", "lm_loss", "lm_output"}. The pooled BAM delta is injected
    into the (post-final-norm) last hidden state and broadcast across all positions; loss
    is computed only over answer tokens (lm_labels with -100 elsewhere).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Gradient checkpointing requires use_cache off; set on both top-level and text config.
        if hasattr(self.backbone.config, "use_cache"):
            self.backbone.config.use_cache = False
        if hasattr(self.backbone.config, "text_config") and hasattr(self.backbone.config.text_config, "use_cache"):
            self.backbone.config.text_config.use_cache = False
        # Log the outcome on BOTH paths so the training log unambiguously shows whether
        # gradient checkpointing is active — if it silently failed, stage-2 (full backbone
        # in the backward graph) keeps all activations and OOMs.
        try:
            self.backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            print("[BAMVLQA] gradient checkpointing ENABLED (use_reentrant=False)")
        except Exception as e:
            print(f"[BAMVLQA] gradient checkpointing SKIPPED: {type(e).__name__}: {e}")

    def forward(
        self,
        input_ids,
        attention_mask=None,
        domain_ids=None,
        lm_labels=None,
        *,
        facial_feats=None,
        pose_feats=None,
        audio_feats=None,
        facial_mask=None,
        pose_mask=None,
        audio_mask=None,
        train_mode=False,
        pixel_values_videos=None,
        video_grid_thw=None,
        pixel_values=None,
        image_grid_thw=None,
        mm_token_type_ids=None,
        **kwargs,
    ):
        if domain_ids is None:
            raise ValueError("domain_ids required")

        # 1) Single backbone pass (with native video/image if provided).
        #    logits_to_keep=1 makes the backbone compute only a 1-token logit slice instead
        #    of the full [B, T, vocab] tensor (~B*T*150k*2 bytes) — we never use out.logits
        #    because we recompute lm_head from the hidden states below. This avoids
        #    materializing the huge logits twice.
        mm = self._collect_mm_kwargs(pixel_values_videos, video_grid_thw, pixel_values, image_grid_thw,
                                     mm_token_type_ids=mm_token_type_ids)
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            logits_to_keep=1,
            **mm,
            **kwargs,
        )
        # With mixed_precision="no" there is no accelerate autocast, so Qwen3.5's
        # fp32-preserving sublayers (RMSNorm / linear-attention) can emit fp32 hidden states
        # while the BAM adapters / lm_head are bf16 -> "mat1 and mat2 must have the same
        # dtype". Cast the two states the custom head consumes to the compute dtype (what
        # autocast used to do implicitly) so pool -> adapters -> delta -> lm_head is
        # dtype-consistent.
        cdt = self._compute_dtype()
        hs_penult = out.hidden_states[-2].to(cdt)
        # POST-final-norm: transformers ties hidden_states[-1] to last_hidden_state, which the
        # text model returns *after* self.norm. Do NOT norm it again (see
        # _check_head_convention_once for why that silently destroys the logits).
        hs_last = out.hidden_states[-1].to(cdt)
        self._check_head_convention_once(out, hs_last)

        # 2) Pool penultimate + apply BAM adapter deltas
        pooled_base = self._pool_penultimate(hs_penult, attention_mask)
        pooled_eff = self._apply_adapters(pooled_base, facial_feats, pose_feats, audio_feats, train_mode,
                                          facial_mask=facial_mask, pose_mask=pose_mask, audio_mask=audio_mask)

        # 3) Inject delta into the normed last hidden state, so official-equivalent logits are
        #    lm_head(hidden_states[-1] + delta) — matching how the backbone itself builds
        #    out.logits. The lm_head is deferred to step 4 so it runs on ONLY the supervised
        #    positions. Injecting after the norm also keeps the delta meaningful: the norm's
        #    output has O(1) scale, whereas the pre-norm state carries Qwen's massive
        #    activations, against which an adapter delta would be normalized away.
        delta = (pooled_eff - pooled_base).to(hs_last.dtype)           # [B, H]
        h_last_mod = hs_last + delta.unsqueeze(1)                      # [B, T, H]

        # 4) Teacher-forcing LM loss + token accuracy, computed on answer tokens only.
        #    Position p predicts token p+1, so position p is supervised iff lm_labels[:, p+1]
        #    != -100. Gathering those positions BEFORE the lm_head shrinks it from [B, T, V]
        #    to [N, V] (N = #answer tokens, tiny vs. a long video seq), cutting both forward
        #    compute and the backward activation/grad memory that drove the earlier OOM.
        #    Mathematically identical to masking a full-sequence CE with ignore_index=-100:
        #    ignored positions contribute zero gradient either way, and the delta still
        #    receives gradient from exactly the supervised positions it broadcasts to.
        lm_loss = None
        lm_token_correct = lm_token_total = None
        if lm_labels is not None:
            shift_labels = lm_labels[:, 1:]                           # [B, T-1] target tokens
            valid = shift_labels != -100                             # answer-token mask
            lm_token_total = valid.sum()
            lm_token_correct = lm_token_total.new_zeros(())          # 0 unless tokens present
            if lm_token_total > 0:
                h_pred = h_last_mod[:, :-1, :][valid]                # [N, H] supervised states
                target = shift_labels[valid]                        # [N]
                sel_logits = self._resolve_lm_head()(h_pred)       # [N, V]
                lm_loss = F.cross_entropy(sel_logits, target)
                with torch.no_grad():
                    lm_token_correct = (sel_logits.argmax(dim=-1) == target).sum()

        # 5) Classification logits (QA rows have domain_id=-1 -> neg_inf)
        cls_logits = self._heads_from_pooled(pooled_eff, domain_ids)

        # Never carry the full [B, T, vocab] logits out: accelerate's convert_to_fp32 would
        # upcast that huge tensor to fp32 (doubling memory). Training and validation both use
        # lm_loss + the token-accuracy scalars above, so the logits are not needed downstream.
        out.loss = lm_loss
        out.logits = None
        return {"cls_logits": cls_logits, "lm_loss": lm_loss,
                "lm_token_correct": lm_token_correct, "lm_token_total": lm_token_total,
                "lm_output": out}
