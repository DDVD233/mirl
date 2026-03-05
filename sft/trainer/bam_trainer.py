"""
BAM (Residual Hidden Adapter) trainer.

Loads a frozen multi-head classification checkpoint, builds video/audio
hidden adapters on top, and trains them (optionally with base model) using
per-module optimizers.

Inherits all shared infrastructure from BaseMultiHeadTrainer. Unique BAM
methods are defined here.
"""
import os
import json
import time
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from tqdm import tqdm
from transformers import get_scheduler

from trainer.base_trainer import BaseMultiHeadTrainer
from utils.logger import log_batch_training_metrics, log_validation_results, log_epoch_training_metrics
from models.bam_utils import build_video_feats_batch, build_audio_feats_batch, maybe_build_hidden_adapters, apply_hidden_adapters


class BAMTrainer(BaseMultiHeadTrainer):
    """
    Overrides:
      - _extra_wandb_config()
      - save_checkpoint_unified()  — adds model_order to meta
      - load_checkpoint_unified()  — adds bam_resume_diff_cfg support
      - validate()                 — applies adapters before classification head
      - train()                    — builds adapters, per-module optimizers
      - test()                     — builds adapters, registers per-module opts
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        cfg = self.global_config

        self.use_bam_video = cfg.get("USE_BAM_VIDEO", False)
        self.use_bam_audio = cfg.get("USE_BAM_AUDIO", False)

        # Stage: "base_only" | "residual_only" | "joint" | "residual_and_head"
        self.bam_stage = cfg.get("BAM_STAGE", "base_only")
        self.bam_resume_diff_training_stage = bool(cfg.get("BAM_RESUME_DIFF_TRAINING_STAGE", False))

        self.bam_hidden = cfg.get("BAM_HIDDEN", 128)
        self.bam_hidden_video = int(cfg.get("BAM_HIDDEN_VIDEO", self.bam_hidden))
        self.bam_hidden_audio = int(cfg.get("BAM_HIDDEN_AUDIO", self.bam_hidden))

        self.bam_pv = cfg.get("BAM_P_MODDROP_VIDEO", 0.30)
        self.bam_pa = cfg.get("BAM_P_MODDROP_AUDIO", 0.30)

        self.video_temporal = cfg.get("BAM_VIDEO_TEMPORAL", "meanstd")
        self.video_norm = cfg.get("BAM_VIDEO_NORM", None)
        self.audio_temporal = cfg.get("BAM_AUDIO_TEMPORAL", "none")
        self.audio_norm = cfg.get("BAM_AUDIO_NORM", "l2")

        self.d_video_feat = cfg.get("D_VIDEO_FEAT", None)
        self.d_audio_feat = cfg.get("D_AUDIO_FEAT", None)

        self.video_adapter = None
        self.audio_adapter = None

    # -------------------------------------------------------------------------
    # Wandb
    # -------------------------------------------------------------------------

    def _extra_wandb_config(self):
        cfg = self.global_config
        return {
            "bam_use_video": bool(cfg.get("USE_BAM_VIDEO", False)),
            "bam_use_audio": bool(cfg.get("USE_BAM_AUDIO", False)),
            "bam_stage": cfg.get("BAM_STAGE", "base_only"),
            "bam_resume_diff_training_stage": bool(cfg.get("BAM_RESUME_DIFF_TRAINING_STAGE", False)),
            "bam_d_video_feat": cfg.get("D_VIDEO_FEAT", None),
            "bam_d_audio_feat": cfg.get("D_AUDIO_FEAT", None),
            "bam_video_temporal": cfg.get("BAM_VIDEO_TEMPORAL", "meanstd"),
            "bam_video_norm": cfg.get("BAM_VIDEO_NORM", None),
            "bam_audio_temporal": cfg.get("BAM_AUDIO_TEMPORAL", "none"),
            "bam_audio_norm": cfg.get("BAM_AUDIO_NORM", "l2"),
            "bam_hidden_global": cfg.get("BAM_HIDDEN", 128),
            "bam_hidden_video": int(cfg.get("BAM_HIDDEN_VIDEO", self.bam_hidden)),
            "bam_hidden_audio": int(cfg.get("BAM_HIDDEN_AUDIO", self.bam_hidden)),
            "bam_p_moddrop_video": cfg.get("BAM_P_MODDROP_VIDEO", 0.30),
            "bam_p_moddrop_audio": cfg.get("BAM_P_MODDROP_AUDIO", 0.30),
            "bam_video_use_ln": bool(cfg.get("BAM_VIDEO_USE_LN", False)),
            "bam_video_use_conf_gain": bool(cfg.get("BAM_VIDEO_USE_CONF_GAIN", False)),
            "bam_video_conf_init_gain": float(cfg.get("BAM_VIDEO_CONF_INIT_GAIN", 3.0)),
            "bam_video_alpha_init": float(cfg.get("BAM_VIDEO_ALPHA_INIT", 1.0)),
            "bam_audio_use_ln": bool(cfg.get("BAM_AUDIO_USE_LN", False)),
            "bam_audio_use_conf_gain": bool(cfg.get("BAM_AUDIO_USE_CONF_GAIN", False)),
            "bam_audio_conf_init_gain": float(cfg.get("BAM_AUDIO_CONF_INIT_GAIN", 3.0)),
            "bam_audio_alpha_init": float(cfg.get("BAM_AUDIO_ALPHA_INIT", 1.0)),
            "base_lr": float(cfg.get("BASE_LR", self.lr * 0.25)),
            "bam_lr": float(cfg.get("BAM_LR", self.lr * 5.0)),
            "hard_gamma": float(cfg.get("HARD_GAMMA", 0.0)),
        }

    # -------------------------------------------------------------------------
    # Checkpoint — overrides to include model_order and bam_resume_diff_cfg
    # -------------------------------------------------------------------------

    def _current_model_order(self):
        order = ["base"]
        if self.bam_stage in {"residual_only", "joint", "residual_and_head"}:
            if self.video_adapter is not None:
                order.append("video")
            if self.audio_adapter is not None:
                order.append("audio")
        return order

    def save_checkpoint_unified(self, accelerator, model, epoch, batch_idx,
                                len_train_dataloader, training_strategy, base_ckpt_dir):
        global_step = epoch * len_train_dataloader + (batch_idx + 1)
        ckpt_dir = os.path.join(base_ckpt_dir, f"step_{global_step}")
        os.makedirs(ckpt_dir, exist_ok=True)
        accelerator.save_state(ckpt_dir)
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            meta = {
                "epoch": int(epoch),
                "global_step": int(global_step),
                "len_train_dataloader": int(len_train_dataloader),
                "training_strategy": str(training_strategy),
                "model_order": self._current_model_order(),
                "saved_at_unix": time.time(),
            }
            with open(os.path.join(ckpt_dir, "meta.json"), "w") as f:
                json.dump(meta, f, indent=2)
        accelerator.print(f"[save] checkpoint @ step {global_step} → {ckpt_dir}")
        return ckpt_dir

    def load_checkpoint_unified(self, accelerator, model, base_ckpt_dir,
                                explicit_dir=None, expect_training_strategy=None,
                                inference_only=False, bam_resume_diff_cfg=False):
        """
        Extended version: adds bam_resume_diff_cfg.
        When True, resets epoch/offset to 0 (fresh training stage, same weights).
        """
        from math import floor

        ckpt_dir = explicit_dir or None
        if ckpt_dir is None:
            print(f"[load] finding latest checkpoint from {base_ckpt_dir}")
            ckpt_dir = self._latest_checkpoint_dir(base_ckpt_dir)
            print(f"[load] latest checkpoint found: {ckpt_dir}")

        print(f"[load] loading checkpoint from {ckpt_dir}")

        if ckpt_dir is None:
            accelerator.print("[load] no checkpoint found; starting fresh.")
            return 0, 0, 0, None, None

        meta_path = os.path.join(ckpt_dir, "meta.json")
        if not os.path.isfile(meta_path):
            accelerator.print(f"[load] missing meta.json in {ckpt_dir}; starting fresh.")
            return 0, 0, 0, None, None

        with open(meta_path, "r") as f:
            meta = json.load(f)

        if expect_training_strategy and meta.get("training_strategy") != expect_training_strategy:
            accelerator.print(
                f"[warn] strategy mismatch: expected {expect_training_strategy}, "
                f"got {meta.get('training_strategy')}"
            )

        if inference_only:
            from accelerate.utils.fsdp_utils import load_fsdp_model
            load_fsdp_model(accelerator.state.fsdp_plugin, accelerator, model, ckpt_dir, model_index=0)
        else:
            accelerator.load_state(ckpt_dir)

        global_step = int(meta["global_step"])
        len_dl = int(meta["len_train_dataloader"])
        if len_dl <= 0:
            accelerator.print("[load] invalid len_train_dataloader; starting at epoch 0.")
            return 0, 0, 0, meta, ckpt_dir

        if bam_resume_diff_cfg:
            start_epoch = 0
            start_batch_offset = 0
        else:
            start_epoch = floor((global_step - 1) / len_dl)
            start_batch_offset = (global_step - 1) % len_dl

        accelerator.print(
            f"[load] resumed {ckpt_dir} → epoch={start_epoch}, "
            f"step={global_step}, offset={start_batch_offset}"
        )
        return start_epoch, start_batch_offset, global_step, meta, ckpt_dir

    # -------------------------------------------------------------------------
    # Adapter setup helpers
    # -------------------------------------------------------------------------

    def _build_adapters(self):
        H = getattr(self.model, "hidden_size", None)
        if H is None:
            raise RuntimeError("Model must expose .hidden_size for BAM out_dim")
        cfg = self.global_config
        self.video_adapter, self.audio_adapter = maybe_build_hidden_adapters(
            domain_id_to_global_indices=self.domain_id_to_global_indices,
            use_bam_video=self.use_bam_video,
            use_bam_audio=self.use_bam_audio,
            bam_hidden_video=self.bam_hidden_video,
            bam_hidden_audio=self.bam_hidden_audio,
            p_moddrop_video=self.bam_pv,
            p_moddrop_audio=self.bam_pa,
            out_dim_hidden=H,
            d_video_feat=self.d_video_feat,
            d_audio_feat=self.d_audio_feat,
            video_use_ln=bool(cfg.get("BAM_VIDEO_USE_LN", False)),
            video_use_conf_gain=bool(cfg.get("BAM_VIDEO_USE_CONF_GAIN", False)),
            video_conf_init_gain=float(cfg.get("BAM_VIDEO_CONF_INIT_GAIN", 3.0)),
            video_alpha_init=float(cfg.get("BAM_VIDEO_ALPHA_INIT", 1.0)),
            audio_use_ln=bool(cfg.get("BAM_AUDIO_USE_LN", False)),
            audio_use_conf_gain=bool(cfg.get("BAM_AUDIO_USE_CONF_GAIN", False)),
            audio_conf_init_gain=float(cfg.get("BAM_AUDIO_CONF_INIT_GAIN", 3.0)),
            audio_alpha_init=float(cfg.get("BAM_AUDIO_ALPHA_INIT", 1.0)),
        )

    def _prepare_modules(self, train_dl, val_dl, include_adapters: bool):
        """Prepare model (and optionally adapters) + dataloaders with accelerator."""
        modules = [self.model]
        if include_adapters and self.bam_stage in {"residual_only", "joint", "residual_and_head"}:
            if self.video_adapter is not None:
                modules.append(self.video_adapter)
            if self.audio_adapter is not None:
                modules.append(self.audio_adapter)
        modules += [train_dl, val_dl]
        prepared = self.accelerator.prepare(*modules)

        idx = 0
        self.model = prepared[idx]; idx += 1
        if include_adapters and self.bam_stage in {"residual_only", "joint", "residual_and_head"}:
            if self.video_adapter is not None:
                self.video_adapter = prepared[idx]; idx += 1
            if self.audio_adapter is not None:
                self.audio_adapter = prepared[idx]; idx += 1
        train_dl = prepared[idx]; idx += 1
        val_dl = prepared[idx]
        return train_dl, val_dl

    def _prepare_fresh_adapters(self):
        """Prepare freshly built adapters (called after model-only prepare)."""
        adapters = [a for a in (self.video_adapter, self.audio_adapter) if a is not None]
        if not adapters:
            return
        prepared = self.accelerator.prepare(*adapters)
        if not isinstance(prepared, (list, tuple)):
            prepared = [prepared]
        i = 0
        if self.video_adapter is not None:
            self.video_adapter = prepared[i]; i += 1
        if self.audio_adapter is not None and i < len(prepared):
            self.audio_adapter = prepared[i]

    def prepare_params_for_training(self, base_lr=None, bam_lr=None):
        base_lr = self.lr if base_lr is None else base_lr
        bam_lr = self.lr if bam_lr is None else bam_lr

        def _set(module, flag):
            if module is None:
                return
            for p in module.parameters():
                p.requires_grad = flag

        bundles = {"base": None, "video": None, "audio": None}

        if self.bam_stage == "base_only":
            _set(self.video_adapter, False)
            _set(self.audio_adapter, False)
            params = [p for p in self.model.parameters() if p.requires_grad]
            if params:
                bundles["base"] = {"params": params, "lr": base_lr}

        elif self.bam_stage == "residual_only":
            _set(self.model, False)
            _set(self.video_adapter, True)
            _set(self.audio_adapter, True)
            base_params = list(self.model.parameters())
            if base_params:
                bundles["base"] = {"params": base_params, "lr": 0.0, "weight_decay": 0.0}
            if self.video_adapter is not None:
                vp = [p for p in self.video_adapter.parameters() if p.requires_grad]
                if vp:
                    bundles["video"] = {"params": vp, "lr": bam_lr}
            if self.audio_adapter is not None:
                ap = [p for p in self.audio_adapter.parameters() if p.requires_grad]
                if ap:
                    bundles["audio"] = {"params": ap, "lr": bam_lr}

        elif self.bam_stage == "residual_and_head":
            _set(self.model, False)
            _set(self.model.heads, True)
            _set(self.video_adapter, True)
            _set(self.audio_adapter, True)
            hp = [p for p in self.model.heads.parameters() if p.requires_grad]
            if hp:
                bundles["base"] = {"params": hp, "lr": base_lr}
            if self.video_adapter is not None:
                vp = [p for p in self.video_adapter.parameters() if p.requires_grad]
                if vp:
                    bundles["video"] = {"params": vp, "lr": bam_lr}
            if self.audio_adapter is not None:
                ap = [p for p in self.audio_adapter.parameters() if p.requires_grad]
                if ap:
                    bundles["audio"] = {"params": ap, "lr": bam_lr}

        elif self.bam_stage == "joint":
            _set(self.video_adapter, True)
            _set(self.audio_adapter, True)
            params = [p for p in self.model.parameters() if p.requires_grad]
            if params:
                bundles["base"] = {"params": params, "lr": base_lr}
            if self.video_adapter is not None:
                vp = [p for p in self.video_adapter.parameters() if p.requires_grad]
                if vp:
                    bundles["video"] = {"params": vp, "lr": bam_lr}
            if self.audio_adapter is not None:
                ap = [p for p in self.audio_adapter.parameters() if p.requires_grad]
                if ap:
                    bundles["audio"] = {"params": ap, "lr": bam_lr}
        else:
            raise ValueError(f"Unknown BAM stage: {self.bam_stage}")

        return bundles

    def _build_per_module_optimizers(self, bundles):
        opt_base = Adam(bundles["base"]["params"], lr=bundles["base"]["lr"]) if bundles["base"] else None
        opt_video = Adam(bundles["video"]["params"], lr=bundles["video"]["lr"]) if bundles["video"] else None
        opt_audio = Adam(bundles["audio"]["params"], lr=bundles["audio"]["lr"]) if bundles["audio"] else None
        opts = [o for o in (opt_base, opt_video, opt_audio) if o is not None]
        names = [n for n, o in zip(["base", "video", "audio"], (opt_base, opt_video, opt_audio)) if o is not None]
        return opts, names

    def _build_schedulers(self, opts, total_updates):
        if not self.use_scheduler:
            return []
        return [
            get_scheduler(self.scheduler_type, o,
                          num_warmup_steps=self.warmup_steps,
                          num_training_steps=total_updates)
            for o in opts
        ]

    def _prepare_opts_scheds(self, opts, scheds):
        if scheds:
            prepared = self.accelerator.prepare(*opts, *scheds)
            n = len(opts)
            self.prepared_opts = list(prepared[:n])
            self.prepared_scheds = list(prepared[n:])
            for s in self.prepared_scheds:
                self.accelerator.register_for_checkpointing(s)
        else:
            self.prepared_opts = list(self.accelerator.prepare(*opts))
            self.prepared_scheds = []

    def _step_all_opts(self):
        for opt in self.prepared_opts:
            opt.step()

    def _zero_all_opts(self):
        for opt in self.prepared_opts:
            opt.zero_grad(set_to_none=True)

    def _step_all_scheds(self):
        for sch in self.prepared_scheds:
            sch.step()

    def _current_lr(self):
        return self.prepared_opts[0].param_groups[0]["lr"] if self.prepared_opts else self.lr

    # -------------------------------------------------------------------------
    # Feature extraction helpers (called per batch)
    # -------------------------------------------------------------------------

    def _pool_feats(self, batch, device):
        cfg = self.global_config
        use_adapters = self.bam_stage in {"residual_only", "joint", "residual_and_head"}

        pooled_video = None
        if use_adapters and self.use_bam_video and "video_feats" in batch and batch["video_feats"] is not None:
            pooled_video = build_video_feats_batch(
                batch["video_feats"], device=device,
                temporal_mode=cfg.get("BAM_VIDEO_TEMPORAL", "meanstd"),
                use_conf=cfg.get("BAM_VIDEO_USE_CONF", True),
                norm=self.video_norm,
                target_dim=self.d_video_feat,
            )

        pooled_audio = None
        if use_adapters and self.use_bam_audio and "audio_feats" in batch and batch["audio_feats"] is not None:
            pooled_audio = build_audio_feats_batch(
                batch["audio_feats"], device=device,
                temporal_mode=self.audio_temporal,
                norm=self.audio_norm,
                target_dim=self.d_audio_feat,
            )

        return pooled_video, pooled_audio

    def _forward_with_adapters(self, input_ids, attention_mask, domain_ids,
                                pooled_video, pooled_audio, train_mode):
        """Two-pass forward: get pooled hidden, apply adapters, get final logits."""
        prelim_logits, pooled = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            domain_ids=domain_ids,
        )
        use_adapters = (
            self.bam_stage in {"residual_only", "joint", "residual_and_head"}
            and (self.use_bam_video or self.use_bam_audio)
        )
        if use_adapters:
            pooled = apply_hidden_adapters(
                h_base=pooled,
                domain_ids=domain_ids,
                prelim_global_logits=prelim_logits,
                video_hidden_adapter=self.video_adapter,
                audio_hidden_adapter=self.audio_adapter,
                video_feats=pooled_video,
                audio_feats=pooled_audio,
                train_mode=train_mode,
            )
        logits, _ = self.model(domain_ids=domain_ids, pooled=pooled)
        return logits

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    def validate(self, val_dataloader, split_name="validation", current_step=None):
        self.model.eval()
        if self.video_adapter is not None:
            self.video_adapter.eval()
        if self.audio_adapter is not None:
            self.audio_adapter.eval()

        from evaluate.detailed_multi_task_evaluation import evaluate_predictions

        total_loss = 0.0
        all_predictions = []
        all_labels = []
        all_datasets = []
        criterion = CrossEntropyLoss()
        num_classes = self.global_config.get('NUM_CLASSES', 0)

        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validating", total=len(val_dataloader),
                              disable=not self.accelerator.is_main_process):
                if 'input_ids' not in batch or 'labels' not in batch:
                    raise KeyError(f"Batch missing required keys. Got: {list(batch.keys())}")
                if 'dataset' not in batch:
                    raise KeyError("Batch missing 'dataset' needed for domain routing.")

                input_ids = batch['input_ids']
                labels = batch['labels']
                attention_mask = batch.get('attention_mask', None)
                domain_ids = self._datasets_to_domain_ids(batch['dataset'], device=input_ids.device)

                if labels.dim() != 1:
                    if labels.dim() == 2 and labels.size(1) == num_classes:
                        labels = labels.argmax(dim=1)
                    else:
                        raise ValueError(f"Unexpected labels shape {labels.shape}")

                pooled_video, pooled_audio = self._pool_feats(batch, input_ids.device)
                logits = self._forward_with_adapters(
                    input_ids, attention_mask, domain_ids, pooled_video, pooled_audio, train_mode=False
                )

                loss = criterion(logits, labels)
                total_loss += loss.item() * input_ids.size(0)
                preds = logits.argmax(dim=1)

                gathered_preds = self.accelerator.gather_for_metrics(preds)
                gathered_labels = self.accelerator.gather_for_metrics(labels)
                gathered_datasets = self.accelerator.gather_for_metrics(batch.get('dataset'))

                if self.accelerator.is_main_process:
                    all_predictions.extend(gathered_preds.cpu().numpy())
                    all_labels.extend(gathered_labels.cpu().numpy())
                    all_datasets.extend(gathered_datasets)

        avg_loss = total_loss / max(1, len(all_labels)) if self.accelerator.is_main_process else 0.0

        if self.accelerator.is_main_process:
            evaluation_results = evaluate_predictions(
                predictions=all_predictions,
                ground_truths=all_labels,
                datasets=all_datasets or None,
                split_name=split_name,
                save_path=self.validation_result_dir,
                global_steps=current_step,
                label_map_path=self.label_map_path,
            )
            agg = evaluation_results["aggregate_metrics"]
            accuracy = agg.get("micro_accuracy", 0.0)
            f1 = agg.get("micro_f1", 0.0)
            precision = agg.get("micro_precision", 0.0)
            recall = agg.get("micro_recall", 0.0)

            print(f"{split_name.capitalize()} - Loss: {avg_loss:.4f} - Acc: {accuracy:.4f} - F1: {f1:.4f}")
            print(f"  Macro F1: {agg.get('macro_f1', 0.0):.4f} - Weighted F1: {agg.get('weighted_f1', 0.0):.4f}")

            return {
                'loss': avg_loss, 'accuracy': accuracy, 'precision': precision,
                'recall': recall, 'f1': f1, 'predictions': all_predictions,
                'labels': all_labels, 'evaluation_results': evaluation_results,
                'aggregate_metrics': agg,
            }
        return None

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------

    def train(self):
        train_dataloader = self.get_dataloader(
            self.data_files, self.batch_size, num_workers=self.num_workers, shuffle=True
        )
        val_dataloader = self.get_dataloader(
            self.val_data_files, self.val_batch_size, num_workers=self.num_workers, shuffle=False
        )

        self._build_adapters()

        total_updates = self.epochs * len(train_dataloader)
        base_lr = self.global_config.get("BASE_LR", self.lr * 0.25)
        bam_lr = self.global_config.get("BAM_LR", self.lr * 5.0)
        gamma = float(self.global_config.get("HARD_GAMMA", 0.0))
        self.prepared_opts, self.prepared_scheds = [], []

        if self.bam_resume_diff_training_stage:
            # Load base model only from previous stage checkpoint (inference_only to skip
            # optimizer/scheduler count mismatch)
            train_dataloader, val_dataloader = self._prepare_modules(
                train_dataloader, val_dataloader, include_adapters=False
            )
            self.load_checkpoint_unified(
                accelerator=self.accelerator,
                model=self.model,
                base_ckpt_dir=self.checkpoint_dir,
                explicit_dir=self.load_checkpoint_path or None,
                expect_training_strategy=self.global_config.get("TRAINING_STRATEGY"),
                inference_only=True,
                bam_resume_diff_cfg=True,
            )
            # Prepare freshly built adapters after base model is loaded
            self._prepare_fresh_adapters()
            start_epoch, start_batch_offset = 0, 0
        else:
            # Same-regime resume: model + adapters prepared together
            train_dataloader, val_dataloader = self._prepare_modules(
                train_dataloader, val_dataloader, include_adapters=True
            )
            start_epoch, start_batch_offset, _, _, _ = self.load_checkpoint_unified(
                accelerator=self.accelerator,
                model=self.model,
                base_ckpt_dir=self.checkpoint_dir,
                explicit_dir=self.load_checkpoint_path or None,
                expect_training_strategy=self.global_config.get("TRAINING_STRATEGY"),
                bam_resume_diff_cfg=False,
            )

        bundles = self.prepare_params_for_training(base_lr=base_lr, bam_lr=bam_lr)
        opts, _ = self._build_per_module_optimizers(bundles)
        scheds = self._build_schedulers(opts, total_updates)
        self._prepare_opts_scheds(opts, scheds)

        criterion = CrossEntropyLoss()
        validate_every_n_epochs = self.global_config.get('VALIDATE_EVERY_N_EPOCHS', None)
        validate_every_n_steps = self.global_config.get('VALIDATE_EVERY_N_STEPS', None)
        save_every_n_epochs = self.global_config.get('SAVE_EVERY_N_EPOCHS', None)
        save_every_n_steps = self.global_config.get('SAVE_EVERY_N_STEPS', None)
        early_stopping_patience = self.global_config.get('EARLY_STOPPING_PATIENCE', 0)
        use_wandb = self.global_config.get('USE_WANDB', False)
        num_classes = self.global_config.get('NUM_CLASSES', 0)

        for epoch in tqdm(range(start_epoch, self.epochs), desc="Epochs", position=0,
                          disable=not self.accelerator.is_main_process):
            self.model.train()
            if self.video_adapter is not None:
                self.video_adapter.train()
            if self.audio_adapter is not None:
                self.audio_adapter.train()

            if hasattr(train_dataloader, "sampler") and hasattr(train_dataloader.sampler, "set_epoch"):
                train_dataloader.sampler.set_epoch(epoch)

            total_loss = 0.0
            correct = 0
            total = 0
            epoch_start_time = time.time()
            eff_loss = 0.0
            eff_correct = 0
            eff_total = 0

            for batch_idx, batch in tqdm(enumerate(train_dataloader), desc="Training",
                                         total=len(train_dataloader),
                                         disable=not self.accelerator.is_main_process):
                self.model.train()
                if self.video_adapter is not None:
                    self.video_adapter.train()
                if self.audio_adapter is not None:
                    self.audio_adapter.train()

                if epoch == start_epoch and batch_idx < start_batch_offset:
                    continue

                current_step = epoch * len(train_dataloader) + batch_idx + 1

                if 'input_ids' not in batch or 'labels' not in batch:
                    raise KeyError(f"Batch missing required keys. Got: {list(batch.keys())}")
                if 'dataset' not in batch:
                    raise KeyError("Batch missing 'dataset' needed for domain routing.")

                input_ids = batch['input_ids']
                labels = batch['labels']
                attention_mask = batch.get('attention_mask', None)
                domain_ids = self._datasets_to_domain_ids(batch['dataset'], device=input_ids.device)

                if labels.dim() != 1:
                    if labels.dim() == 2 and labels.size(1) == num_classes:
                        labels = labels.argmax(dim=1)
                    else:
                        raise ValueError(f"Unexpected labels shape {labels.shape}")

                pooled_video, pooled_audio = self._pool_feats(batch, input_ids.device)

                with self.accelerator.accumulate(self.model):
                    logits = self._forward_with_adapters(
                        input_ids, attention_mask, domain_ids,
                        pooled_video, pooled_audio, train_mode=True
                    )

                    if gamma > 0:
                        with torch.no_grad():
                            probs = torch.softmax(logits, dim=-1)
                            p_true = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
                            hardness = (1.0 - p_true).clamp_(1e-6, 1.0)
                            weights = hardness.pow(gamma)
                        ce = F.cross_entropy(logits, labels, reduction='none')
                        loss = (weights * ce).sum() / weights.sum().clamp_min(1.0)
                    else:
                        loss = criterion(logits, labels)

                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite loss encountered")

                    self.accelerator.backward(loss)
                    self._step_all_opts()
                    self._step_all_scheds()
                    self._zero_all_opts()

                current_lr = self._current_lr()

                with torch.no_grad():
                    eff_loss += loss.item() * input_ids.size(0)
                    preds = logits.argmax(dim=1)
                    gathered_preds = self.accelerator.gather_for_metrics(preds)
                    gathered_labels = self.accelerator.gather_for_metrics(labels)

                    if self.accelerator.is_main_process:
                        eff_correct += (gathered_preds == gathered_labels).sum().item()
                        eff_total += gathered_labels.size(0)
                        correct += (gathered_preds == gathered_labels).sum().item()
                        total += gathered_labels.size(0)
                    total_loss += loss.item() * input_ids.size(0)

                    log_batch_training_metrics(
                        epoch=epoch, batch_idx=batch_idx, total_batches=len(train_dataloader),
                        loss=eff_loss, correct=eff_correct, total=max(1, eff_total),
                        epoch_start_time=epoch_start_time, start_time=self.start_time,
                        gradient_accumulation_steps=self.gradient_accumulation_steps,
                        batch_size=self.batch_size, epochs=self.epochs,
                        accelerator=self.accelerator, use_wandb=use_wandb,
                        current_lr=current_lr, current_step=current_step,
                    )

                    if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                        eff_loss = 0.0
                        eff_correct = 0
                        eff_total = 0

                    if validate_every_n_steps and current_step % validate_every_n_steps == 0:
                        print(f"\n[STEP {current_step}] Running step-based validation...")
                        val_results = self.validate(val_dataloader, "validation", current_step=current_step)
                        if self.accelerator.is_main_process and val_results is not None:
                            val_f1 = val_results['f1']
                            if val_f1 > self.best_val_acc:
                                self.best_val_acc = val_f1
                                self.steps_without_improvement = 0
                            else:
                                self.steps_without_improvement += 1
                            val_results['best_val_f1'] = self.best_val_acc
                            val_results['steps_without_improvement'] = self.steps_without_improvement
                            log_validation_results(val_results=val_results, current_step=current_step,
                                                   split_name="validation", accelerator=self.accelerator,
                                                   use_wandb=use_wandb)

                    if save_every_n_steps and current_step % save_every_n_steps == 0:
                        print(f"\n[STEP {current_step}] Saving checkpoint...")
                        self.save_checkpoint_unified(
                            accelerator=self.accelerator, model=self.model, epoch=epoch,
                            batch_idx=batch_idx, len_train_dataloader=len(train_dataloader),
                            training_strategy=self.global_config.get("TRAINING_STRATEGY"),
                            base_ckpt_dir=self.checkpoint_dir,
                        )

            avg_train_loss = total_loss / max(1, total)
            train_acc = correct / max(1, total)

            if self.accelerator.is_main_process:
                print(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {avg_train_loss:.4f} - Train Acc: {train_acc:.4f}")

            if validate_every_n_epochs and (epoch + 1) % validate_every_n_epochs == 0:
                val_results = self.validate(val_dataloader, "validation", current_step=current_step)
                if self.accelerator.is_main_process and val_results is not None:
                    val_f1 = val_results['f1']
                    if val_f1 > self.best_val_acc:
                        self.best_val_acc = val_f1
                        self.epochs_without_improvement = 0
                    else:
                        self.epochs_without_improvement += 1
                    val_results['best_val_f1'] = self.best_val_acc
                    val_results['epochs_without_improvement'] = self.epochs_without_improvement
                    log_epoch_training_metrics(epoch=epoch, avg_train_loss=avg_train_loss,
                                               train_acc=train_acc, total_batches=len(train_dataloader),
                                               accelerator=self.accelerator, use_wandb=use_wandb,
                                               current_step=current_step)
                    log_validation_results(val_results=val_results, current_step=current_step,
                                           split_name="validation", accelerator=self.accelerator,
                                           use_wandb=use_wandb)
            else:
                log_epoch_training_metrics(epoch=epoch, avg_train_loss=avg_train_loss,
                                           train_acc=train_acc, total_batches=len(train_dataloader),
                                           accelerator=self.accelerator, use_wandb=use_wandb,
                                           current_step=current_step)

            if save_every_n_epochs and (epoch + 1) % save_every_n_epochs == 0:
                self.save_checkpoint_unified(
                    accelerator=self.accelerator, model=self.model, epoch=epoch,
                    batch_idx=batch_idx, len_train_dataloader=len(train_dataloader),
                    training_strategy=self.global_config.get("TRAINING_STRATEGY"),
                    base_ckpt_dir=self.checkpoint_dir,
                )

            if validate_every_n_steps is not None:
                if self.steps_without_improvement >= early_stopping_patience:
                    if self.accelerator.is_main_process:
                        print(f"Early stopping triggered after {early_stopping_patience} steps")
                    break
            else:
                if self.epochs_without_improvement >= early_stopping_patience:
                    if self.accelerator.is_main_process:
                        print(f"Early stopping triggered after {early_stopping_patience} epochs")
                    break

    # -------------------------------------------------------------------------
    # Test
    # -------------------------------------------------------------------------

    def test(self):
        print("\n" + "=" * 50)
        print("STARTING TESTING PHASE")
        print("=" * 50)

        train_dataloader = self.get_dataloader(
            self.data_files, self.batch_size, num_workers=self.num_workers, shuffle=True
        )
        test_dataloader = self.get_dataloader(
            self.test_data_files, self.test_batch_size, num_workers=self.num_workers, shuffle=False
        )

        self._build_adapters()
        train_dataloader, test_dataloader = self._prepare_modules(
            train_dataloader, test_dataloader,
            include_adapters=(self.bam_stage in {"residual_only", "joint", "residual_and_head"}),
        )

        base_lr = self.global_config.get("BASE_LR", self.lr * 0.25)
        bam_lr = self.global_config.get("BAM_LR", self.lr * 5.0)
        total_updates = max(1, self.epochs * len(train_dataloader))
        bundles = self.prepare_params_for_training(base_lr=base_lr, bam_lr=bam_lr)
        opts, _ = self._build_per_module_optimizers(bundles)
        scheds = self._build_schedulers(opts, total_updates)
        self._prepare_opts_scheds(opts, scheds)

        self.load_checkpoint_unified(
            accelerator=self.accelerator,
            model=self.model,
            base_ckpt_dir=self.checkpoint_dir,
            explicit_dir=self.load_checkpoint_path or None,
            expect_training_strategy=self.global_config.get("TRAINING_STRATEGY"),
            bam_resume_diff_cfg=False,
        )

        test_results = self.validate(test_dataloader, "test", current_step=1)

        if self.accelerator.is_main_process and test_results is not None:
            print(f"\nOverall TEST RESULTS:")
            print(f"Test Loss: {test_results['loss']:.4f}")
            print(f"Test Micro Accuracy: {test_results['accuracy']:.4f}")
            print(f"Test Micro F1: {test_results['f1']:.4f}")

            use_wandb = self.global_config.get('USE_WANDB', False)
            log_validation_results(val_results=test_results, current_step=1, split_name="test",
                                   accelerator=self.accelerator, use_wandb=use_wandb)

        return test_results
