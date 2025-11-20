import torch
import time
import numpy as np

from trainer.rha_multi_head_omni_classifier_trainer import RHAMultiHeadOmniClassifierAccelerateTrainer
from models.adapter_utils import build_video_feats_batch, build_audio_feats_batch
from models.rha_adapter_utils import maybe_build_hidden_adapters, apply_hidden_adapters


class RHAMultiHeadOmniClassifierProfiler(RHAMultiHeadOmniClassifierAccelerateTrainer):
    """
    A subclass that ONLY adds profiling utilities.
    No changes to train()/validate().
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Build adapters (same logic as train(), but without touching train())
        H = getattr(self.model, "hidden_size", None)
        if H is None:
            raise RuntimeError("Model must expose .hidden_size")

        self.video_adapter, self.audio_adapter = maybe_build_hidden_adapters(
            domain_id_to_global_indices=self.domain_id_to_global_indices,
            use_rha_video=self.use_rla_video,
            use_rha_audio=self.use_rla_audio,
            rha_hidden_video=self.rla_hidden_video,
            rha_hidden_audio=self.rla_hidden_audio,
            p_moddrop_video=self.rla_pv,
            p_moddrop_audio=self.rla_pa,
            out_dim_hidden=H,
            d_video_feat=self.d_video_feat,
            d_audio_feat=self.d_audio_feat,
            video_use_ln=bool(self.global_config.get("RLA_VIDEO_USE_LN", False)),
            video_use_conf_gain=bool(self.global_config.get("RLA_VIDEO_USE_CONF_GAIN", False)),
            video_conf_init_gain=float(self.global_config.get("RLA_VIDEO_CONF_INIT_GAIN", 3.0)),
            video_alpha_init=float(self.global_config.get("RLA_VIDEO_ALPHA_INIT", 1.0)),
            audio_use_ln=bool(self.global_config.get("RLA_AUDIO_USE_LN", False)),
            audio_use_conf_gain=bool(self.global_config.get("RLA_AUDIO_USE_CONF_GAIN", False)),
            audio_conf_init_gain=float(self.global_config.get("RLA_AUDIO_CONF_INIT_GAIN", 3.0)),
            audio_alpha_init=float(self.global_config.get("RLA_AUDIO_ALPHA_INIT", 1.0)),
        )

    ###########################################################################
    # --- Profiling methods ---------------------------------------------------
    ###########################################################################

    def profile_forward_cost(self, num_batches: int = 10, split: str = "val", description: str = "profile"):
        """
        Measure forward latency + peak VRAM across `num_batches` *timed* batches,
        discarding the first batch as warmup.

        This uses a SINGLE dataloader loop; the first iteration is warmup,
        the next `num_batches` iterations are timed.
        """
        dataloader = (
            self.get_dataloader(self.val_data_files, self.val_batch_size)
            if split == "val"
            else self.get_dataloader(self.test_data_files, self.test_batch_size)
        )

        # Prepare modules w/ Accelerate (same as train/validate)
        dataloader, _, _ = self._accelerate_prepare_modules(
            train_dataloader=dataloader,
            val_dataloader=dataloader,
            prepare_base_model=True,
            prepare_adapters=True,
        )

        device = self.accelerator.device

        # Ensure eval + no grad for inference-style cost
        self.model.eval()
        if self.video_adapter is not None:
            self.video_adapter.eval()
        if self.audio_adapter is not None:
            self.audio_adapter.eval()

        latencies = []

        # Single loop: batch 0 = warmup; batches 1..num_batches = timed
        for step, batch in enumerate(dataloader):
            # Move labels etc handled by dataset/collate; here we just profile
            if step == 0:
                # Warmup (no timing, no memory stats)
                with torch.no_grad():
                    self._profile_single_forward(batch)
                # Reset peak memory *after* warmup
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats(device)
                continue

            if step > num_batches:
                break

            # Timing
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            start = time.perf_counter()

            with torch.no_grad():
                _ = self._profile_single_forward(batch)

            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            end = time.perf_counter()

            latencies.append(end - start)

        if not latencies:
            self.accelerator.print("[PROFILE] No batches were profiled (check num_batches / dataloader length).")
            return {
                "mean_latency_s": float("nan"),
                "std_latency_s": float("nan"),
                "peak_vram_mb": float("nan"),
            }

        mean = float(np.mean(latencies))
        std = float(np.std(latencies))
        peak_mb = (
            torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            if torch.cuda.is_available()
            else float("nan")
        )

        self.accelerator.print(
            f"[PROFILE] {description}: mean {mean:.4f}s, std {std:.4f}s, peak VRAM {peak_mb:.1f} MB "
            f"over {len(latencies)} batches (first batch used for warmup)."
        )

        return {
            "mean_latency_s": mean,
            "std_latency_s": std,
            "peak_vram_mb": peak_mb,
        }

    def _profile_single_forward(self, batch):
        """
        Execute the SAME forward path as in train()/validate, but:
        - no gradients
        - no loss/metrics
        - no optimizer steps

        Important: This still uses the two-step model interface
        (base encoder -> pooled hidden, then head-only forward),
        exactly like the training loop.
        """
        # Required keys
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask", None)

        if "dataset" not in batch:
            raise KeyError("Batch missing 'dataset' needed for domain routing.")

        domain_ids = self._datasets_to_domain_ids(batch["dataset"], input_ids.device)

        # ---- Audio feats (mirrors training conditions) ----
        pooled_audio_feats = None
        if (
            "audio_feats" in batch
            and batch["audio_feats"] is not None
            and (self.rla_stage in {"residual_only", "joint", "residual_and_head"})
            and self.use_rla_audio
        ):
            pooled_audio_feats = build_audio_feats_batch(
                batch["audio_feats"],
                device=input_ids.device,
                temporal_mode=self.audio_temporal,
                norm=self.audio_norm,
                target_dim=self.d_audio_feat,
            )

        # ---- Video feats (mirrors training conditions) ----
        pooled_video_feats = None
        if (
            "video_feats" in batch
            and batch["video_feats"] is not None
            and (self.rla_stage in {"residual_only", "joint", "residual_and_head"})
            and self.use_rla_video
        ):
            pooled_video_feats = build_video_feats_batch(
                batch["video_feats"],
                device=input_ids.device,
                temporal_mode=self.global_config.get("RLA_VIDEO_TEMPORAL", "meanstd"),
                use_conf=self.global_config.get("RLA_VIDEO_USE_CONF", True),
                norm=self.video_norm,
                target_dim=self.d_video_feat,
            )

        # ---- Base model forward (encoder + global head logits) ----
        # This matches the training code:
        #   prelim_logits, pooled = self.model(input_ids=..., attention_mask=..., domain_ids=...)
        prelim_logits, pooled = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            domain_ids=domain_ids,
        )

        # ---- Hidden fusion (RHA) if enabled ----
        if (self.rla_stage in {"residual_only", "joint", "residual_and_head"}) and (
            self.use_rla_video or self.use_rla_audio
        ):
            pooled = apply_hidden_adapters(
                h_base=pooled,
                domain_ids=domain_ids,
                prelim_global_logits=prelim_logits,
                video_hidden_adapter=self.video_adapter,
                audio_hidden_adapter=self.audio_adapter,
                video_feats=pooled_video_feats,
                audio_feats=pooled_audio_feats,
                train_mode=False,  # profiling = inference-style
            )

        # ---- Final logits from pooled state (small head-only forward) ----
        # Again, this directly mirrors the training loop:
        #   logits, _ = self.model(domain_ids=domain_ids, pooled=pooled)
        logits, _ = self.model(
            domain_ids=domain_ids,
            pooled=pooled,
        )

        return logits