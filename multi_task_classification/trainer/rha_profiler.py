import time
from typing import Any, Dict, List, Optional

import torch
import numpy as np
from accelerate import Accelerator
from torch.utils.data import DataLoader

# You already have these in your codebase:
# - OmniClassifierDataset
# - collate_fn
# - build_audio_feats_batch
# - build_video_feats_batch
# - maybe_build_hidden_adapters
# - apply_hidden_adapters
from models.adapter_utils import build_video_feats_batch, build_audio_feats_batch
from models.rha_adapter_utils import maybe_build_hidden_adapters, apply_hidden_adapters
from mt_dataset.omni_classifier_dataset import OmniClassifierDataset, log_failed_path


class RHAMultiHeadOmniClassifierProfiler:
    """
    Standalone computational profiler for the RHA multi-head omni classifier.

    - No inheritance from the training trainer.
    - No optimizer, no scheduler, no loss, no evaluation.
    - Just: dataloader -> forward passes -> latency + peak VRAM.
    """

    def __init__(
        self,
        data_files: List[str],
        tokenizer,
        processor,
        config: Dict[str, Any],
        model,
        global_config: Dict[str, Any],
        batch_size: int = 2,
        num_workers: int = 0,
        mixed_precision: str = "fp16",  # "fp16" or "bf16" or "no"
    ):
        """
        Args:
            data_files: list of JSONL paths (same format as train/val).
            tokenizer, processor, config: same as in RHAMultiHeadOmniClassifierAccelerateTrainer.
            model: the already-constructed omni classifier model.
            global_config: same dict you pass into the trainer (contains RLA knobs + label map).
        """
        self.data_files = data_files
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.model = model
        self.global_config = global_config or {}

        self.batch_size = batch_size
        self.num_workers = num_workers

        # Label / routing info
        self.full_label_scheme = self.global_config.get("FULL_LABEL_SCHEME", None)
        self.label_map = self.global_config.get("LABEL_MAP", {})

        # RLA toggles / knobs
        self.use_rla_video = bool(self.global_config.get("USE_RLA_VIDEO", False))
        self.use_rla_audio = bool(self.global_config.get("USE_RLA_AUDIO", False))
        self.rla_stage = self.global_config.get("RLA_STAGE", "base_only")

        self.rla_hidden = self.global_config.get("RLA_HIDDEN", 128)
        self.rla_hidden_video = int(self.global_config.get("RLA_HIDDEN_VIDEO", self.rla_hidden))
        self.rla_hidden_audio = int(self.global_config.get("RLA_HIDDEN_AUDIO", self.rla_hidden))

        self.rla_pv = self.global_config.get("RLA_P_MODDROP_VIDEO", 0.30)
        self.rla_pa = self.global_config.get("RLA_P_MODDROP_AUDIO", 0.30)

        # Feature pipeline knobs
        self.video_temporal = self.global_config.get("RLA_VIDEO_TEMPORAL", "meanstd")
        self.video_norm = self.global_config.get("RLA_VIDEO_NORM", None)

        self.audio_temporal = self.global_config.get("RLA_AUDIO_TEMPORAL", "none")
        self.audio_norm = self.global_config.get("RLA_AUDIO_NORM", "l2")

        self.d_video_feat = self.global_config.get("D_VIDEO_FEAT", None)
        self.d_audio_feat = self.global_config.get("D_AUDIO_FEAT", None)

        # Domain routing tables (same logic as trainer.build_domain_routing)
        self._build_domain_routing()

        # Adapters (built once we know hidden_size)
        self.video_adapter = None
        self.audio_adapter = None

        # Simple accelerator (no wandb, no schedulers)
        self.accelerator = Accelerator(
            gradient_accumulation_steps=1,
            mixed_precision=mixed_precision,
        )

    # -------------------------------------------------------------------------
    # Domain routing (copied from trainer.build_domain_routing)
    # -------------------------------------------------------------------------
    def _build_domain_routing(self):
        meta = self.full_label_scheme.get("meta", {})
        global_classes = meta.get("global_classes", {})
        domain_names = list(global_classes.keys())  # e.g. ['sentiment_intensity','emotion','mental_health']

        # {'sentiment_intensity': 0, 'emotion': 1, 'mental_health': 2}
        self.domain_name_to_id = {d: i for i, d in enumerate(domain_names)}

        # [[global_idx...], [global_idx...], ...] per domain
        self.domain_id_to_global_indices = [
            [x["index"] for x in global_classes[d]] for d in domain_names
        ]

        dataset_to_domain = meta.get("dataset_domain", {})  # e.g. 'mosei_senti': 'sentiment_intensity'
        self.dataset_to_domain_id = {
            ds: self.domain_name_to_id[dn] for ds, dn in dataset_to_domain.items()
        }

    def _datasets_to_domain_ids(self, dataset_names, device):
        ids = []
        for ds in dataset_names:
            if isinstance(ds, bytes):
                ds = ds.decode("utf-8")
            if ds not in self.dataset_to_domain_id:
                raise KeyError(f"Dataset '{ds}' not in label_map.meta.dataset_domain")
            ids.append(self.dataset_to_domain_id[ds])
        return torch.tensor(ids, dtype=torch.long, device=device)

    # -------------------------------------------------------------------------
    # Dataloader (mirrors trainer.get_dataloader)
    # -------------------------------------------------------------------------
    def _get_dataloader(self, shuffle: bool = True) -> DataLoader:
        dataset = OmniClassifierDataset(
            data_files=self.data_files,
            tokenizer=self.tokenizer,
            config=self.config,
            processor=self.processor,
            label_key=self.config.get("label_key", "answer"),
            label_map=self.label_map,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    # -------------------------------------------------------------------------
    # Build RHA adapters (mirrors test() / __init__ usage)
    # -------------------------------------------------------------------------
    def _build_adapters_if_needed(self):
        H = getattr(self.model, "hidden_size", None)
        if H is None:
            raise RuntimeError("Model must expose .hidden_size for RHA adapters.")

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
            # per-modality adapter knobs
            video_use_ln=bool(self.global_config.get("RLA_VIDEO_USE_LN", False)),
            video_use_conf_gain=bool(self.global_config.get("RLA_VIDEO_USE_CONF_GAIN", False)),
            video_conf_init_gain=float(self.global_config.get("RLA_VIDEO_CONF_INIT_GAIN", 3.0)),
            video_alpha_init=float(self.global_config.get("RLA_VIDEO_ALPHA_INIT", 1.0)),
            audio_use_ln=bool(self.global_config.get("RLA_AUDIO_USE_LN", False)),
            audio_use_conf_gain=bool(self.global_config.get("RLA_AUDIO_USE_CONF_GAIN", False)),
            audio_conf_init_gain=float(self.global_config.get("RLA_AUDIO_CONF_INIT_GAIN", 3.0)),
            audio_alpha_init=float(self.global_config.get("RLA_AUDIO_ALPHA_INIT", 1.0)),
        )

    # -------------------------------------------------------------------------
    # Single forward (mirrors validate() forward path, minus loss/metrics)
    # -------------------------------------------------------------------------
    def _single_forward(self, batch):
        """
        EXACT same forward path as in validate():

        prelim_logits, pooled = model(input_ids, attention_mask, domain_ids)
        pooled = apply_hidden_adapters(...)
        logits, _ = model(domain_ids=domain_ids, pooled=pooled)
        """
        if "input_ids" not in batch:
            raise KeyError(f"Batch missing 'input_ids'. Got: {list(batch.keys())}")

        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask", None)

        if "dataset" not in batch:
            raise KeyError("Batch missing 'dataset' needed for domain routing.")

        device = input_ids.device
        domain_ids = self._datasets_to_domain_ids(batch["dataset"], device=device)

        # --- Audio feats (mirror validate) ---
        pooled_audio_feats = None
        if (
            "audio_feats" in batch
            and (self.rla_stage in {"residual_only", "joint", "residual_and_head"})
            and self.use_rla_audio
        ):
            audio_feats = batch["audio_feats"]
            pooled_audio_feats = build_audio_feats_batch(
                audio_feats,
                device=device,
                temporal_mode=self.global_config.get("RLA_AUDIO_TEMPORAL", "none"),
                norm=self.global_config.get("RLA_AUDIO_NORM", "l2"),
                target_dim=self.d_audio_feat,
            )
        # else stays None

        # --- Video feats (mirror validate) ---
        pooled_video_feats = None
        if (
            "video_feats" in batch
            and (self.rla_stage in {"residual_only", "joint", "residual_and_head"})
            and self.use_rla_video
        ):
            video_feats = batch["video_feats"]
            pooled_video_feats = build_video_feats_batch(
                video_feats,
                device=device,
                temporal_mode=self.global_config.get("RLA_VIDEO_TEMPORAL", "meanstd"),
                use_conf=self.global_config.get("RLA_VIDEO_USE_CONF", True),
                target_dim=self.d_video_feat,
            )
        # else stays None

        # --- Base model forward (encoder + global head logits) ---
        prelim_logits, pooled = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            domain_ids=domain_ids,
        )

        # --- RHA fusion (if enabled) ---
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

        # --- Final logits (classification head) ---
        logits, _ = self.model(
            domain_ids=domain_ids,
            pooled=pooled,
        )
        return logits

    # -------------------------------------------------------------------------
    # Public API: profile_forward_cost
    # -------------------------------------------------------------------------
    def profile_forward_cost(self, num_batches: int = 10, description: str = "rha_profile") -> Dict[str, float]:
        """
        Run inference-style forwards over `num_batches` batches from `data_files`:

        - First batch: warmup (not timed)
        - Next `num_batches` batches: timed
        - Reports mean/std latency and peak VRAM (per rank) in MB.
        """
        # Build dataloader on CPU
        dataloader = self._get_dataloader(shuffle=True)

        # Build adapters once (if needed)
        self._build_adapters_if_needed()

        # Prepare model + adapters + dataloader exactly once
        modules = [self.model]
        if self.rla_stage in {"residual_only", "joint", "residual_and_head"}:
            if self.video_adapter is not None:
                modules.append(self.video_adapter)
            if self.audio_adapter is not None:
                modules.append(self.audio_adapter)
        modules.append(dataloader)

        prepared = self.accelerator.prepare(*modules)

        idx = 0
        self.model = prepared[idx]; idx += 1
        if self.rla_stage in {"residual_only", "joint", "residual_and_head"}:
            if self.video_adapter is not None:
                self.video_adapter = prepared[idx]; idx += 1
            if self.audio_adapter is not None:
                self.audio_adapter = prepared[idx]; idx += 1
        dataloader = prepared[idx]

        device = self.accelerator.device

        # Eval mode + no grad
        self.model.eval()
        if self.video_adapter is not None:
            self.video_adapter.eval()
        if self.audio_adapter is not None:
            self.audio_adapter.eval()

        latencies: List[float] = []

        # Single loop: step 0 = warmup; steps 1..num_batches = timed
        for step, batch in enumerate(dataloader):
            if step == 0:
                # Warmup – no timing, no mem stats
                with torch.no_grad():
                    _ = self._single_forward(batch)
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats(device)
                continue

            if step > num_batches:
                break

            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            start = time.perf_counter()

            with torch.no_grad():
                _ = self._single_forward(batch)

            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            end = time.perf_counter()

            latencies.append(end - start)

        if not latencies:
            self.accelerator.print(
                f"[PROFILE] {description}: no batches profiled "
                f"(num_batches={num_batches}, dataset too small?)."
            )
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

        gpu_name = torch.cuda.get_device_name(device) if torch.cuda.is_available() else "CPU"
        self.accelerator.print(
            f"[PROFILE] {description} :: "
            f"GPU={gpu_name}, batch_size={self.batch_size}, mixed_precision={self.accelerator.mixed_precision}\n"
            f"  mean latency: {mean:.4f}s, std: {std:.4f}s over {len(latencies)} batches "
            f"(first batch warmup)\n"
            f"  peak VRAM: {peak_mb:.1f} MB"
        )

        return {
            "mean_latency_s": mean,
            "std_latency_s": std,
            "peak_vram_mb": peak_mb,
        }