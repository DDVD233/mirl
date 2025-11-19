import torch
import time
import numpy as np

from trainer.rha_multi_head_omni_classifier_trainer import RHAMultiHeadOmniClassifierAccelerateTrainer
from models.adapter_utils import build_video_feats_batch, build_audio_feats_batch
from models.rha_adapter_utils import maybe_build_hidden_adapters, apply_hidden_adapters

## TODO NOTES: Select about 100-500 samples only; discard the 1st batch as warmup and run for N batches
# You MUST report:
# 	•	GPU model (A100/H100 etc)
# 	•	Batch size
# 	•	Precision (fp16/bf16)
# 	•	Seq length (if applicable)
# 	•	Input resolution (if applicable)

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

    def profile_forward_cost(self, num_batches=10, split="val", description="profile"):
        """
        Measure forward latency + peak VRAM across num_batches.
        """
        dataloader = (
            self.get_dataloader(self.val_data_files, self.val_batch_size)
            if split == "val"
            else self.get_dataloader(self.test_data_files, self.test_batch_size)
        )

        # Prepare modules w/ Accelerate
        dataloader, _, _ = self._accelerate_prepare_modules(
            train_dataloader=dataloader,
            val_dataloader=dataloader,
            prepare_base_model=True,
            prepare_adapters=True,
        )

        device = self.accelerator.device

        # Warm-up
        for b in dataloader:
            self._profile_single_forward(b)
            break

        # Reset
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)

        latencies = []
        for step, batch in enumerate(dataloader):
            if step >= num_batches:
                break

            torch.cuda.synchronize(device)
            start = time.perf_counter()

            _ = self._profile_single_forward(batch)

            torch.cuda.synchronize(device)
            end = time.perf_counter()

            latencies.append(end - start)

        mean = float(np.mean(latencies))
        std  = float(np.std(latencies))
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2)

        self.accelerator.print(
            f"[PROFILE] {description}: mean {mean:.4f}s, std {std:.4f}s, peak VRAM {peak_mb:.1f} MB"
        )

        return {
            "mean_latency_s": mean,
            "std_latency_s": std,
            "peak_vram_mb": peak_mb,
        }

    def _profile_single_forward(self, batch):
        """
       Executes EXACT same forward path as train/validate, without gradients or loss.
        """
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask", None)

        domain_ids = self._datasets_to_domain_ids(batch["dataset"], input_ids.device)

        # Audio feats
        pooled_audio = None
        if ("audio_feats" in batch) and self.use_rla_audio and (self.rla_stage != "base_only"):
            pooled_audio = build_audio_feats_batch(
                batch["audio_feats"], input_ids.device,
                self.audio_temporal, self.audio_norm, self.d_audio_feat
            )

        # Video feats
        pooled_video = None
        if ("video_feats" in batch) and self.use_rla_video and (self.rla_stage != "base_only"):
            pooled_video = build_video_feats_batch(
                batch["video_feats"], input_ids.device,
                self.global_config.get("RLA_VIDEO_TEMPORAL", "meanstd"),
                self.global_config.get("RLA_VIDEO_USE_CONF", True),
                self.d_video_feat
            )

        # Base model forward
        prelim, pooled = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            domain_ids=domain_ids,
        )

        # Apply adapters
        if self.rla_stage != "base_only" and (self.use_rla_audio or self.use_rla_video):
            pooled = apply_hidden_adapters(
                h_base=pooled,
                domain_ids=domain_ids,
                prelim_global_logits=prelim,
                video_hidden_adapter=self.video_adapter,
                audio_hidden_adapter=self.audio_adapter,
                video_feats=pooled_video,
                audio_feats=pooled_audio,
                train_mode=False,
            )

        logits, _ = self.model(domain_ids=domain_ids, pooled=pooled)
        return logits