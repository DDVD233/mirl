"""
BAM-on-Qwen3-VL trainer (parallel to bam_trainer.py for Omni).

Trains a single BAM-augmented Qwen3-VL model on one dataset at a time.
  --task_type qa   ->  BAMVLQA  model, teacher-forcing LM loss   (primary; user choice)
  --task_type cls  ->  BAMVLCLS model, multi-head CE loss

Key differences vs BAMTrainer:
  * THREE side-channel adapters (facial / pose / audio) instead of two.
  * Side features are PRE-POOLED to fixed vectors in the dataset (OmniClassifierDataset
    with vl_feat_config), each accompanied by a 0/1 presence mask. So _extract_feats here
    just moves the already-batched [B, D] tensors to device — no per-batch pooling.
  * Native video (pixel_values_videos) is NOT wired in this trainer yet; this is the
    runnable text+BAM QA core. The model forward already accepts video kwargs, so adding
    them later is additive (see VL_USE_NATIVE_VIDEO guard).

Adapters are registered as nn.Module submodules on the model wrapper BEFORE
accelerator.prepare(), so FSDP/DDP wraps model + adapters as one unit.
"""
import os
import json
import time
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from tqdm import tqdm
from transformers import get_scheduler

from trainer.base_trainer import BaseMultiHeadTrainer
from dataset.sft_dataset import OmniClassifierDataset
from dataset.dataset_utils import collate_fn
from torch.utils.data import DataLoader
from utils.logger import log_batch_training_metrics, log_validation_results, log_epoch_training_metrics
from utils.wandb_utils import log_metrics
from models.bam_vl_utils import maybe_build_vl_adapters

STREAMS = ("facial", "pose", "audio")


class BAMVLTrainer(BaseMultiHeadTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = self.global_config

        self.task_type = cfg.get("TASK_TYPE", "qa")

        # Per-stream enable flags / feature dims / adapter hidden sizes / dropout / pooling.
        self.use_bam = {s: bool(cfg.get(f"USE_BAM_{s.upper()}", False)) for s in STREAMS}
        self.d_feat = {s: cfg.get(f"D_{s.upper()}_FEAT", None) for s in STREAMS}
        default_hidden = int(cfg.get("BAM_HIDDEN", 128))
        self.bam_hidden = {s: int(cfg.get(f"BAM_HIDDEN_{s.upper()}", default_hidden)) for s in STREAMS}
        self.p_moddrop = {s: float(cfg.get(f"BAM_P_MODDROP_{s.upper()}", 0.30)) for s in STREAMS}
        # Pooling modes default to the locked choices (facial/pose meanstd, audio none).
        default_modes = {"facial": "meanstd", "pose": "meanstd", "audio": "none"}
        self.temporal = {s: cfg.get(f"BAM_{s.upper()}_TEMPORAL", default_modes[s]) for s in STREAMS}
        self.use_ln = {s: bool(cfg.get(f"BAM_{s.upper()}_USE_LN", False)) for s in STREAMS}
        self.alpha_init = {s: float(cfg.get(f"BAM_{s.upper()}_ALPHA_INIT", 1.0)) for s in STREAMS}

        self.bam_stage = cfg.get("BAM_STAGE", "bam_only")
        self.bam_fresh_start = bool(cfg.get("BAM_FRESH_START", False))
        self.qa_loss_weight = float(cfg.get("QA_LOSS_WEIGHT", 1.0))
        self.qa_datasets = set(d.lower() for d in cfg.get("QA_DATASETS", []))

        # Native video stays off in this trainer (deferred); guard so configs can't silently
        # assume it works yet.
        self.use_native_video = bool(cfg.get("VL_USE_NATIVE_VIDEO", False))
        if self.use_native_video:
            raise NotImplementedError(
                "VL_USE_NATIVE_VIDEO=True is not wired in BAMVLTrainer yet "
                "(native-video collation/teacher-forcing is a separate step). "
                "Run with native video off for the text+BAM QA core."
            )

        self.validate_every_n_epochs = cfg.get("VALIDATE_EVERY_N_EPOCHS", None)
        self.validate_every_n_steps = cfg.get("VALIDATE_EVERY_N_STEPS", None)
        self.save_every_n_epochs = cfg.get("SAVE_EVERY_N_EPOCHS", None)
        self.save_every_n_steps = cfg.get("SAVE_EVERY_N_STEPS", None)
        self.early_stopping_patience = cfg.get("EARLY_STOPPING_PATIENCE", 0)
        self.use_wandb = bool(cfg.get("USE_WANDB", False))
        self.num_classes = int(cfg.get("NUM_CLASSES", 0))
        # Gradient-norm clipping (matches verl fsdp_sft_trainer's config.optim.clip_grad,
        # default 1.0). clip_grad_norm_ also returns the pre-clip norm, which we log.
        self.max_grad_norm = float(cfg.get("MAX_GRAD_NORM", 1.0))
        # When no explicit load_checkpoint_path is given, start fresh by default rather than
        # auto-resuming the latest checkpoint in save_checkpoint_dir. Set RESUME_FROM_LATEST
        # to opt back into auto-resume.
        self.resume_from_latest = bool(cfg.get("RESUME_FROM_LATEST", False))

        self.adapters = {s: None for s in STREAMS}
        self.prepared_opts, self.prepared_scheds = [], []

    # -------------------------------------------------------------------------
    # Wandb / checkpoint metadata
    # -------------------------------------------------------------------------

    def _extra_wandb_config(self):
        # IMPORTANT: BaseMultiHeadTrainer.__init__ calls _init_wandb() -> this method
        # BEFORE this subclass's __init__ body sets self.bam_stage / self.use_bam / etc.
        # So read from self.global_config (set early in the base __init__), not self.*.
        gc = self.global_config
        out = {
            "task_type":       gc.get("TASK_TYPE", "qa"),
            "bam_stage":       gc.get("BAM_STAGE", "bam_only"),
            "bam_fresh_start": bool(gc.get("BAM_FRESH_START", False)),
            "qa_loss_weight":  float(gc.get("QA_LOSS_WEIGHT", 1.0)),
            "base_lr":         float(gc.get("BASE_LR", self.lr * 0.25)),
            "bam_lr":          float(gc.get("BAM_LR", self.lr * 5.0)),
        }
        for s in STREAMS:
            S = s.upper()
            out[f"bam_use_{s}"]      = bool(gc.get(f"USE_BAM_{S}", False))
            out[f"bam_d_{s}_feat"]   = gc.get(f"D_{S}_FEAT", None)
            out[f"bam_hidden_{s}"]   = int(gc.get(f"BAM_HIDDEN_{S}", gc.get("BAM_HIDDEN", 128)))
            out[f"bam_p_moddrop_{s}"] = float(gc.get(f"BAM_P_MODDROP_{S}", 0.30))
            out[f"bam_{s}_temporal"] = gc.get(f"BAM_{S}_TEMPORAL", "meanstd" if s != "audio" else "none")
        return out

    def _current_model_order(self):
        order = ["base"]
        for s in STREAMS:
            if self.adapters[s] is not None:
                order.append(s)
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
                "epoch": int(epoch), "global_step": int(global_step),
                "len_train_dataloader": int(len_train_dataloader),
                "training_strategy": str(training_strategy),
                "model_order": self._current_model_order(),
                "saved_at_unix": time.time(),
            }
            with open(os.path.join(ckpt_dir, "meta.json"), "w") as f:
                json.dump(meta, f, indent=2)
        accelerator.print(f"[save] checkpoint @ step {global_step} -> {ckpt_dir}")
        return ckpt_dir

    def _checkpoint_is_loadable(self, ckpt_dir, inference_only):
        """
        Heuristic guard: does ckpt_dir contain model weights this run can actually load?
        FSDP runs save/load the model as pytorch_model_fsdp{_i}.bin; non-FSDP runs use
        model*.safetensors / pytorch_model*.bin. This catches incomplete saves and
        FSDP/non-FSDP format mismatches (e.g. a single-GPU smoke-test checkpoint left in
        an FSDP run's save dir) before we attempt the load.
        """
        import glob
        using_fsdp = getattr(self.accelerator.state, "fsdp_plugin", None) is not None
        if using_fsdp or inference_only:
            patterns = ["pytorch_model_fsdp*.bin"]
        else:
            patterns = ["*.safetensors", "pytorch_model*.bin"]
        return any(glob.glob(os.path.join(ckpt_dir, p)) for p in patterns)

    def load_checkpoint_unified(self, accelerator, model, base_ckpt_dir,
                                explicit_dir=None, expect_training_strategy=None,
                                inference_only=False, auto_resume=False):
        from math import floor
        # By default (auto_resume=False) we only load when an explicit checkpoint path is
        # given. With no explicit path we start a FRESH run rather than silently picking up
        # the latest checkpoint in the save dir — so a prior training in the same save dir
        # can't bleed into a new one. Set auto_resume=True (RESUME_FROM_LATEST) to opt back
        # into resuming the most recent checkpoint.
        ckpt_dir = explicit_dir
        if ckpt_dir is None and auto_resume:
            ckpt_dir = self._latest_checkpoint_dir(base_ckpt_dir)
        if ckpt_dir is None:
            if not auto_resume:
                accelerator.print("[load] no load_checkpoint_path given; starting a FRESH run "
                                  "(prior checkpoints in the save dir are ignored; set "
                                  "RESUME_FROM_LATEST=true or load_checkpoint_path to resume).")
            else:
                accelerator.print("[load] no checkpoint found; starting fresh.")
            return 0, 0, 0, None, None
        meta_path = os.path.join(ckpt_dir, "meta.json")
        if not os.path.isfile(meta_path):
            accelerator.print(f"[load] missing meta.json in {ckpt_dir}; starting fresh.")
            return 0, 0, 0, None, None
        # Hardening: skip checkpoints with no compatible model weights (incomplete save or
        # FSDP/non-FSDP format mismatch) rather than crashing on the load.
        if not self._checkpoint_is_loadable(ckpt_dir, inference_only):
            accelerator.print(f"[load] {ckpt_dir} has no compatible model weights "
                              f"(incomplete or wrong format); starting fresh.")
            return 0, 0, 0, None, None
        with open(meta_path, "r") as f:
            meta = json.load(f)
        if expect_training_strategy and meta.get("training_strategy") != expect_training_strategy:
            accelerator.print(f"[warn] strategy mismatch: expected {expect_training_strategy}, "
                              f"got {meta.get('training_strategy')}")
        try:
            if inference_only:
                from accelerate.utils.fsdp_utils import load_fsdp_model
                load_fsdp_model(accelerator.state.fsdp_plugin, accelerator, model, ckpt_dir, model_index=0)
            else:
                accelerator.load_state(ckpt_dir)
        except Exception as e:
            # Backstop: any load failure (corruption, partial save, version skew) -> fresh
            # start with a clear message instead of aborting the whole run.
            accelerator.print(f"[load] failed to load checkpoint {ckpt_dir} "
                              f"({type(e).__name__}: {e}); starting fresh.")
            return 0, 0, 0, None, None
        global_step = int(meta["global_step"])
        len_dl = int(meta["len_train_dataloader"])
        if len_dl <= 0:
            return 0, 0, 0, meta, ckpt_dir
        start_epoch = floor((global_step - 1) / len_dl)
        start_batch_offset = (global_step - 1) % len_dl
        accelerator.print(f"[load] resumed {ckpt_dir} -> epoch={start_epoch}, step={global_step}, "
                          f"offset={start_batch_offset}")
        return start_epoch, start_batch_offset, global_step, meta, ckpt_dir

    # -------------------------------------------------------------------------
    # Dataloader (pass vl_feat_config + qa_datasets)
    # -------------------------------------------------------------------------

    def _build_vl_feat_config(self):
        return {s: {"use": self.use_bam[s],
                    "dim": int(self.d_feat[s]) if self.d_feat[s] is not None else None,
                    "mode": self.temporal[s]}
                for s in STREAMS if self.use_bam[s]}

    def get_dataloader(self, data_files, batch_size, num_workers=0, shuffle=True):
        dataset = OmniClassifierDataset(
            data_files=data_files,
            tokenizer=self.tokenizer,
            config=self.config,
            processor=self.processor,
            label_key=self.label_key,
            label_map=self.label_map,
            qa_datasets=list(self.qa_datasets),
            vl_feat_config=self._build_vl_feat_config(),
        )
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn,
            num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0,
        )

    # -------------------------------------------------------------------------
    # Adapter build + preparation
    # -------------------------------------------------------------------------

    def _build_adapters(self, attach_to_model=True):
        H = getattr(self.model, "hidden_size", None)
        if H is None:
            raise RuntimeError("Model must expose .hidden_size for BAM out_dim")
        fa, pa, aa = maybe_build_vl_adapters(
            out_dim_hidden=H,
            use_facial=self.use_bam["facial"], use_pose=self.use_bam["pose"], use_audio=self.use_bam["audio"],
            d_facial_feat=self.d_feat["facial"], d_pose_feat=self.d_feat["pose"], d_audio_feat=self.d_feat["audio"],
            hidden_facial=self.bam_hidden["facial"], hidden_pose=self.bam_hidden["pose"], hidden_audio=self.bam_hidden["audio"],
            p_moddrop_facial=self.p_moddrop["facial"], p_moddrop_pose=self.p_moddrop["pose"], p_moddrop_audio=self.p_moddrop["audio"],
            use_ln_facial=self.use_ln["facial"], use_ln_pose=self.use_ln["pose"], use_ln_audio=self.use_ln["audio"],
        )
        self.adapters = {"facial": fa, "pose": pa, "audio": aa}

        # FSDP flattens each wrapped unit into one tensor and requires a single dtype.
        # The backbone (and heads) are bf16; adapters default to fp32, which makes the
        # root FSDP unit mixed-dtype (and would also break the bf16 forward matmul).
        # Cast adapters to the backbone parameter dtype.
        try:
            bb_dtype = next(self.model.backbone.parameters()).dtype
        except StopIteration:
            bb_dtype = None
        if bb_dtype is not None:
            for ad in self.adapters.values():
                if ad is not None:
                    ad.to(dtype=bb_dtype)

        if attach_to_model:
            # nn.Module.__setattr__ auto-registers these as submodules so
            # accelerator.prepare(model) wraps them with the backbone.
            self.model.facial_adapter = self.adapters["facial"]
            self.model.pose_adapter = self.adapters["pose"]
            self.model.audio_adapter = self.adapters["audio"]

    def _prepare_modules(self, train_dl, val_dl):
        model, train_dl, val_dl = self.accelerator.prepare(self.model, train_dl, val_dl)
        self.model = model
        return train_dl, val_dl

    def _prepare_fresh_adapters_separately(self):
        present = [(s, self.adapters[s]) for s in STREAMS if self.adapters[s] is not None]
        if not present:
            return
        prepared = self.accelerator.prepare(*[a for _, a in present])
        if not isinstance(prepared, (list, tuple)):
            prepared = [prepared]
        inner = getattr(self.model, "module", self.model)
        for (s, _), pa in zip(present, prepared):
            self.adapters[s] = pa
            setattr(inner, f"{s}_adapter", pa)

    # -------------------------------------------------------------------------
    # Parameter groups + optimizers
    # -------------------------------------------------------------------------

    def prepare_params_for_training(self, base_lr=None, bam_lr=None):
        base_lr = self.lr if base_lr is None else base_lr
        bam_lr = self.lr if bam_lr is None else bam_lr

        def _set(module, flag):
            if module is None:
                return
            for p in module.parameters():
                p.requires_grad = flag

        # The adapters are submodules of self.model, so self.model.parameters() INCLUDES
        # them. Track adapter param ids so the "base" bundle can exclude them — otherwise
        # adapter params land in both the base group and their own per-stream group, which
        # a single consolidated optimizer rejects ("params appear in >1 param group").
        adapter_param_ids = set()
        for s in STREAMS:
            if self.adapters[s] is not None:
                adapter_param_ids.update(id(p) for p in self.adapters[s].parameters())

        def _base_params(require_grad_only):
            out = []
            for p in self.model.parameters():
                if id(p) in adapter_param_ids:
                    continue
                if require_grad_only and not p.requires_grad:
                    continue
                out.append(p)
            return out

        bundles = {"base": None, "facial": None, "pose": None, "audio": None}

        if self.bam_stage == "bam_only":
            _set(self.model, False)
            for s in STREAMS:
                _set(self.adapters[s], True)
            base_params = _base_params(require_grad_only=False)  # frozen backbone+heads, lr 0
            if base_params:
                bundles["base"] = {"params": base_params, "lr": 0.0, "weight_decay": 0.0}
        elif self.bam_stage == "bam_and_classifier_heads_only":
            _set(self.model, False)
            if hasattr(self.model, "heads"):
                _set(self.model.heads, True)
            for s in STREAMS:
                _set(self.adapters[s], True)
            hp = _base_params(require_grad_only=True)  # heads only (adapters excluded)
            if hp:
                bundles["base"] = {"params": hp, "lr": base_lr}
        elif self.bam_stage == "bam_and_full_model":
            for s in STREAMS:
                _set(self.adapters[s], True)
            params = _base_params(require_grad_only=True)  # backbone+heads (adapters excluded)
            if params:
                bundles["base"] = {"params": params, "lr": base_lr}
        else:
            raise ValueError(f"Unknown BAM stage: {self.bam_stage}")

        for s in STREAMS:
            if self.adapters[s] is not None:
                sp = [p for p in self.adapters[s].parameters() if p.requires_grad]
                if sp:
                    bundles[s] = {"params": sp, "lr": bam_lr}
        return bundles

    def _build_per_module_optimizers(self, bundles):
        """
        Build optimizer(s) over the per-module parameter bundles.

        Normal path (adapters are submodules of the single wrapped model): return ONE
        optimizer with one param group per bundle, each keeping its own lr. This is
        required because accelerate's FSDP save_state pairs optimizer[i] with model[i];
        with a single model, N separate optimizers -> IndexError on save.

        bam_fresh_start path (adapters are prepared as separate models): return one
        optimizer per bundle, in [base, *STREAMS] order so they line up 1:1 with the
        separately-prepared models.
        """
        ordered = [(name, bundles[name]) for name in (["base"] + list(STREAMS)) if bundles.get(name)]
        if not ordered:
            return [], []
        if self.bam_fresh_start:
            opts = [Adam(b["params"], lr=b["lr"]) for _, b in ordered]
            names = [name for name, _ in ordered]
            return opts, names
        # Consolidated single optimizer with per-group learning rates.
        param_groups = [{"params": b["params"], "lr": b["lr"]} for _, b in ordered]
        return [Adam(param_groups)], ["+".join(name for name, _ in ordered)]

    def _build_schedulers(self, opts, total_updates):
        if not self.use_scheduler:
            return []
        return [get_scheduler(self.scheduler_type, o, num_warmup_steps=self.warmup_steps,
                              num_training_steps=total_updates) for o in opts]

    def _prepare_opts_scheds(self, opts, scheds):
        # accelerator.prepare returns a single object for one arg and a tuple for many,
        # so normalize to a list (the consolidated path passes exactly one optimizer).
        def _as_list(x):
            return list(x) if isinstance(x, (list, tuple)) else [x]

        if scheds:
            prepared = _as_list(self.accelerator.prepare(*opts, *scheds))
            n = len(opts)
            self.prepared_opts = prepared[:n]
            self.prepared_scheds = prepared[n:]
            for s in self.prepared_scheds:
                self.accelerator.register_for_checkpointing(s)
        elif opts:
            self.prepared_opts = _as_list(self.accelerator.prepare(*opts))
            self.prepared_scheds = []
        else:
            self.prepared_opts = []
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
        # Report the largest group lr (the adapter/bam_lr) rather than param_groups[0],
        # which under bam_only is the frozen "base" group at lr 0.
        lrs = [g["lr"] for opt in self.prepared_opts for g in opt.param_groups]
        return max(lrs) if lrs else self.lr

    # -------------------------------------------------------------------------
    # Feature extraction (dataset already pooled to fixed [B, D] + [B] mask)
    # -------------------------------------------------------------------------

    def _extract_feats(self, batch, device):
        """Return {stream: (feats[B,D] or None, mask[B] or None)} moved to device."""
        out = {}
        for s in STREAMS:
            fk, mk = f"{s}_feats", f"{s}_mask"
            if self.use_bam[s] and isinstance(batch.get(fk), torch.Tensor):
                feats = batch[fk].to(device)
                mask = batch[mk].to(device) if isinstance(batch.get(mk), torch.Tensor) else None
                out[s] = (feats, mask)
            else:
                out[s] = (None, None)
        return out

    def _model_feat_kwargs(self, feats):
        return {
            "facial_feats": feats["facial"][0], "facial_mask": feats["facial"][1],
            "pose_feats": feats["pose"][0], "pose_mask": feats["pose"][1],
            "audio_feats": feats["audio"][0], "audio_mask": feats["audio"][1],
        }

    # -------------------------------------------------------------------------
    # Teacher-forcing input builder (QA)
    # -------------------------------------------------------------------------

    def _build_tf_inputs_and_labels(self, batch, seq_len, device):
        ids_all = batch["input_ids"]
        attn_all = batch.get("attention_mask", None)
        B, T = ids_all.size(0), seq_len
        pad_id = self.tokenizer.pad_token_id
        eos_id = self.tokenizer.eos_token_id

        qa_input_ids = torch.full((B, T), pad_id, dtype=torch.long, device=device)
        qa_attn = torch.zeros((B, T), dtype=torch.long, device=device)
        lm_labels = torch.full((B, T), -100, dtype=torch.long, device=device)

        for j in range(B):
            prompt_ids = ids_all[j]
            prompt_len = int(attn_all[j].sum().item()) if attn_all is not None else int((prompt_ids != pad_id).sum().item())
            prompt_len = min(prompt_len, T)
            qa_input_ids[j, :prompt_len] = prompt_ids[:prompt_len]
            qa_attn[j, :prompt_len] = 1

            ans = batch["lm_labels"][j]
            if isinstance(ans, np.generic):
                ans = ans.item()
            if isinstance(ans, bytes):
                ans = ans.decode("utf-8", errors="ignore")
            ans = "" if ans is None else str(ans)

            ans_tok = self.tokenizer.encode(ans, add_special_tokens=False)
            if len(ans_tok) == 0 or ans_tok[-1] != eos_id:
                ans_tok = ans_tok + [eos_id]

            rem = T - prompt_len
            if rem > 0:
                ans_tok = ans_tok[:rem]
                end = prompt_len + len(ans_tok)
                qa_input_ids[j, prompt_len:end] = torch.tensor(ans_tok, device=device)
                qa_attn[j, prompt_len:end] = 1
                lm_labels[j, prompt_len:end] = qa_input_ids[j, prompt_len:end]
        return qa_input_ids, qa_attn, lm_labels

    # -------------------------------------------------------------------------
    # Validation (CLS only; QA generation eval is a separate step)
    # -------------------------------------------------------------------------

    def validate(self, val_dataloader, split_name="validation", current_step=None):
        if self.task_type == "qa":
            # QA generation eval is intentionally deferred; skip to avoid misleading CLS metrics.
            if self.accelerator.is_main_process:
                print(f"[validate] task_type=qa: generation eval not wired; skipping {split_name}.")
            return None

        self.model.eval()
        from evaluate.detailed_multi_task_evaluation import evaluate_predictions
        total_loss = 0.0
        all_predictions, all_labels, all_datasets = [], [], []
        criterion = CrossEntropyLoss()

        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validating", total=len(val_dataloader),
                              disable=not self.accelerator.is_main_process):
                input_ids = batch["input_ids"]
                labels = batch["labels"]
                attention_mask = batch.get("attention_mask", None)
                device = input_ids.device
                domain_ids = self._datasets_to_domain_ids(batch["dataset"], device=device)
                if labels.dim() != 1:
                    if labels.dim() == 2 and labels.size(1) == self.num_classes:
                        labels = labels.argmax(dim=1)
                    else:
                        raise ValueError(f"Unexpected labels shape {labels.shape}")
                feats = self._extract_feats(batch, device)
                out = self.model(input_ids=input_ids, attention_mask=attention_mask,
                                 domain_ids=domain_ids, train_mode=False, **self._model_feat_kwargs(feats))
                logits = out[0] if isinstance(out, tuple) else out["cls_logits"]
                loss = criterion(logits, labels)
                total_loss += loss.item() * labels.size(0)
                preds = logits.argmax(dim=1)
                g_preds = self.accelerator.gather_for_metrics(preds)
                g_labels = self.accelerator.gather_for_metrics(labels)
                g_datasets = self.accelerator.gather_for_metrics(batch["dataset"])
                if self.accelerator.is_main_process:
                    all_predictions.extend(g_preds.cpu().numpy())
                    all_labels.extend(g_labels.cpu().numpy())
                    all_datasets.extend(g_datasets)

        if self.accelerator.is_main_process:
            avg_loss = total_loss / max(1, len(all_labels))
            evaluation_results = evaluate_predictions(
                predictions=all_predictions, ground_truths=all_labels,
                datasets=all_datasets or None, split_name=split_name,
                save_path=self.validation_result_dir, global_steps=current_step,
                label_map_path=self.label_map_path,
            )
            agg = evaluation_results["aggregate_metrics"]
            print(f"{split_name.capitalize()} - Loss: {avg_loss:.4f} - Acc: {agg.get('micro_accuracy',0.0):.4f} "
                  f"- F1: {agg.get('micro_f1',0.0):.4f}")
            return {"loss": avg_loss, "accuracy": agg.get("micro_accuracy", 0.0),
                    "f1": agg.get("micro_f1", 0.0), "aggregate_metrics": agg,
                    "evaluation_results": evaluation_results,
                    "predictions": all_predictions, "labels": all_labels}
        return None

    # -------------------------------------------------------------------------
    # Setup + dispatch
    # -------------------------------------------------------------------------

    def _setup_training(self, train_dl, val_dl):
        total_updates = self.epochs * len(train_dl)
        base_lr = self.global_config.get("BASE_LR", self.lr * 0.25)
        bam_lr = self.global_config.get("BAM_LR", self.lr * 5.0)
        self.prepared_opts, self.prepared_scheds = [], []

        if self.bam_fresh_start:
            train_dl, val_dl = self._prepare_modules(train_dl, val_dl)
            self.load_checkpoint_unified(
                accelerator=self.accelerator, model=self.model, base_ckpt_dir=self.checkpoint_dir,
                explicit_dir=self.load_checkpoint_path or None,
                expect_training_strategy=self.global_config.get("TRAINING_STRATEGY"),
                inference_only=True, auto_resume=self.resume_from_latest,
            )
            self._build_adapters(attach_to_model=False)
            self._prepare_fresh_adapters_separately()
            start_epoch, start_batch_offset = 0, 0
        else:
            self._build_adapters()
            train_dl, val_dl = self._prepare_modules(train_dl, val_dl)
            start_epoch, start_batch_offset, _, _, _ = self.load_checkpoint_unified(
                accelerator=self.accelerator, model=self.model, base_ckpt_dir=self.checkpoint_dir,
                explicit_dir=self.load_checkpoint_path or None,
                expect_training_strategy=self.global_config.get("TRAINING_STRATEGY"),
                auto_resume=self.resume_from_latest,
            )

        bundles = self.prepare_params_for_training(base_lr=base_lr, bam_lr=bam_lr)
        opts, _ = self._build_per_module_optimizers(bundles)
        scheds = self._build_schedulers(opts, total_updates)
        self._prepare_opts_scheds(opts, scheds)
        return train_dl, val_dl, start_epoch, start_batch_offset

    def train(self):
        train_dl = self.get_dataloader(self.data_files, self.batch_size,
                                       num_workers=self.num_workers, shuffle=True)
        val_dl = self.get_dataloader(self.val_data_files, self.val_batch_size,
                                     num_workers=self.num_workers, shuffle=False)
        if self.task_type == "qa":
            self._train_qa(train_dl, val_dl)
        else:
            self._train_cls(train_dl, val_dl)

    # -------------------------------------------------------------------------
    # QA training loop (primary)
    # -------------------------------------------------------------------------

    def _train_qa(self, train_dataloader, val_dataloader):
        train_dataloader, val_dataloader, start_epoch, start_batch_offset = \
            self._setup_training(train_dataloader, val_dataloader)
        device = self.accelerator.device

        # Per-optimizer-step accumulators (one logged point per weight update, aggregated
        # over the gradient-accumulation window — i.e. per step, not per micro-batch).
        win_loss_sum, win_micro = 0.0, 0
        win_correct = torch.zeros((), device=device)
        win_total = torch.zeros((), device=device)

        for epoch in tqdm(range(start_epoch, self.epochs), desc="Epochs", position=0,
                          disable=not self.accelerator.is_main_process):
            self.model.train()
            if hasattr(train_dataloader, "sampler") and hasattr(train_dataloader.sampler, "set_epoch"):
                train_dataloader.sampler.set_epoch(epoch)

            total_loss, total_samples = 0.0, 0

            for batch_idx, batch in tqdm(enumerate(train_dataloader), desc="Training (QA)",
                                         total=len(train_dataloader),
                                         disable=not self.accelerator.is_main_process):
                self.model.train()
                if epoch == start_epoch and batch_idx < start_batch_offset:
                    continue
                current_step = epoch * len(train_dataloader) + batch_idx + 1

                if "input_ids" not in batch or "lm_labels" not in batch:
                    raise KeyError(f"QA batch missing required keys. Got: {list(batch.keys())}")

                input_ids = batch["input_ids"]
                B, T = input_ids.shape
                domain_ids = torch.full((B,), -1, dtype=torch.long, device=input_ids.device)

                tf_input_ids, tf_attn, lm_labels = self._build_tf_inputs_and_labels(batch, T, input_ids.device)
                feats = self._extract_feats(batch, input_ids.device)

                with self.accelerator.accumulate(self.model):
                    out = self.model(
                        input_ids=tf_input_ids, attention_mask=tf_attn, domain_ids=domain_ids,
                        lm_labels=lm_labels, train_mode=True, **self._model_feat_kwargs(feats),
                    )
                    lm_loss = out["lm_loss"]
                    if lm_loss is None or not torch.isfinite(lm_loss):
                        raise FloatingPointError("Non-finite or missing lm_loss")
                    loss = self.qa_loss_weight * lm_loss
                    self.accelerator.backward(loss)

                    # grad_norm + clip at the optimizer-step boundary (mirrors verl
                    # fsdp_sft_trainer: clip_grad_norm_ returns the pre-clip norm; skip the
                    # update if non-finite). accelerate routes this through FSDP.
                    grad_norm = None
                    if self.accelerator.sync_gradients:
                        grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    if grad_norm is not None and not torch.isfinite(grad_norm):
                        if self.accelerator.is_main_process:
                            print(f"[warn] non-finite grad_norm ({grad_norm}); skipping optimizer step")
                        self._zero_all_opts()
                    else:
                        self._step_all_opts()
                        self._step_all_scheds()
                        self._zero_all_opts()

                # --- accumulate per-step stats over the gradient-accumulation window ---
                win_loss_sum += lm_loss.item()
                win_micro += 1
                if out["lm_token_correct"] is not None:
                    win_correct += out["lm_token_correct"].detach()
                    win_total += out["lm_token_total"].detach()
                total_loss += lm_loss.item() * B
                total_samples += B

                # --- log once per optimizer step (train/loss, grad_norm, token_acc, lr) ---
                if self.accelerator.sync_gradients:
                    tok_correct = self.accelerator.reduce(win_correct.clone(), reduction="sum")
                    tok_total = self.accelerator.reduce(win_total.clone(), reduction="sum")
                    step_loss = self.accelerator.reduce(
                        torch.tensor(win_loss_sum / max(1, win_micro), device=device), reduction="mean")
                    if self.accelerator.is_main_process and self.use_wandb:
                        log_metrics("train", {
                            "loss": step_loss.item(),
                            "grad_norm": float(grad_norm) if grad_norm is not None else 0.0,
                            "token_accuracy": (tok_correct / tok_total.clamp_min(1)).item(),
                            "learning_rate": self._current_lr(),
                        }, step=current_step)
                    win_loss_sum, win_micro = 0.0, 0
                    win_correct.zero_(); win_total.zero_()

                if self.validate_every_n_steps and current_step % self.validate_every_n_steps == 0:
                    self._validate_qa(val_dataloader, current_step, split_name="validation")
                    self.model.train()

                if self.save_every_n_steps and current_step % self.save_every_n_steps == 0:
                    self.save_checkpoint_unified(
                        accelerator=self.accelerator, model=self.model, epoch=epoch,
                        batch_idx=batch_idx, len_train_dataloader=len(train_dataloader),
                        training_strategy=self.global_config.get("TRAINING_STRATEGY"),
                        base_ckpt_dir=self.checkpoint_dir,
                    )

            avg_train_loss = total_loss / max(1, total_samples)
            if self.accelerator.is_main_process:
                print(f"Epoch {epoch+1}/{self.epochs} - QA Train Loss: {avg_train_loss:.4f}")
                if self.use_wandb:
                    log_metrics("train", {"epoch_avg_loss": avg_train_loss}, step=current_step)

            # Validate at epoch end too (if a step cadence isn't set).
            if self.validate_every_n_epochs and (epoch + 1) % self.validate_every_n_epochs == 0:
                self._validate_qa(val_dataloader, current_step, split_name="validation")
                self.model.train()

            if self.save_every_n_epochs and (epoch + 1) % self.save_every_n_epochs == 0:
                self.save_checkpoint_unified(
                    accelerator=self.accelerator, model=self.model, epoch=epoch,
                    batch_idx=batch_idx, len_train_dataloader=len(train_dataloader),
                    training_strategy=self.global_config.get("TRAINING_STRATEGY"),
                    base_ckpt_dir=self.checkpoint_dir,
                )

    # -------------------------------------------------------------------------
    # QA validation (teacher-forcing val loss + token accuracy)
    # -------------------------------------------------------------------------

    def _validate_qa(self, val_dataloader, current_step, split_name="validation"):
        """
        Run a full teacher-forcing pass over the val set and log val/loss + val/token_accuracy.
        Mirrors verl fsdp_sft_trainer.validation_step (eval mode, no backward, cross-rank
        reduce), extended with token accuracy.
        """
        self.model.eval()
        device = self.accelerator.device
        loss_sum, n_batches = 0.0, 0
        correct = torch.zeros((), device=device)
        total = torch.zeros((), device=device)

        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f"Validating @ step {current_step}",
                              total=len(val_dataloader), disable=not self.accelerator.is_main_process):
                if "input_ids" not in batch or "lm_labels" not in batch:
                    raise KeyError(f"QA val batch missing keys. Got: {list(batch.keys())}")
                input_ids = batch["input_ids"]
                B, T = input_ids.shape
                domain_ids = torch.full((B,), -1, dtype=torch.long, device=input_ids.device)
                tf_input_ids, tf_attn, lm_labels = self._build_tf_inputs_and_labels(batch, T, input_ids.device)
                feats = self._extract_feats(batch, input_ids.device)
                out = self.model(
                    input_ids=tf_input_ids, attention_mask=tf_attn, domain_ids=domain_ids,
                    lm_labels=lm_labels, train_mode=False, **self._model_feat_kwargs(feats),
                )
                if out["lm_loss"] is not None:
                    loss_sum += out["lm_loss"].item()
                    n_batches += 1
                if out["lm_token_correct"] is not None:
                    correct += out["lm_token_correct"].detach()
                    total += out["lm_token_total"].detach()

        # Reduce across ranks: loss averaged, token counts summed.
        tok_correct = self.accelerator.reduce(correct, reduction="sum")
        tok_total = self.accelerator.reduce(total, reduction="sum")
        avg_loss = self.accelerator.reduce(
            torch.tensor(loss_sum / max(1, n_batches), device=device), reduction="mean")
        token_acc = (tok_correct / tok_total.clamp_min(1)).item()

        if self.accelerator.is_main_process:
            print(f"[val @ step {current_step}] loss={avg_loss.item():.4f} token_acc={token_acc:.4f}")
            if self.use_wandb:
                log_metrics(split_name, {"loss": avg_loss.item(), "token_accuracy": token_acc},
                            step=current_step)
        return {"loss": avg_loss.item(), "token_accuracy": token_acc}

    # -------------------------------------------------------------------------
    # CLS training loop (parity / sanity)
    # -------------------------------------------------------------------------

    def _train_cls(self, train_dataloader, val_dataloader):
        train_dataloader, val_dataloader, start_epoch, start_batch_offset = \
            self._setup_training(train_dataloader, val_dataloader)
        criterion = CrossEntropyLoss()

        for epoch in tqdm(range(start_epoch, self.epochs), desc="Epochs", position=0,
                          disable=not self.accelerator.is_main_process):
            self.model.train()
            if hasattr(train_dataloader, "sampler") and hasattr(train_dataloader.sampler, "set_epoch"):
                train_dataloader.sampler.set_epoch(epoch)
            total_loss, correct, total = 0.0, 0, 0
            epoch_start_time = time.time()

            for batch_idx, batch in tqdm(enumerate(train_dataloader), desc="Training (CLS)",
                                         total=len(train_dataloader),
                                         disable=not self.accelerator.is_main_process):
                self.model.train()
                if epoch == start_epoch and batch_idx < start_batch_offset:
                    continue
                current_step = epoch * len(train_dataloader) + batch_idx + 1

                input_ids = batch["input_ids"]
                labels = batch["labels"]
                attention_mask = batch.get("attention_mask", None)
                device = input_ids.device
                B = input_ids.size(0)
                domain_ids = self._datasets_to_domain_ids(batch["dataset"], device=device)
                if labels.dim() != 1:
                    if labels.dim() == 2 and labels.size(1) == self.num_classes:
                        labels = labels.argmax(dim=1)
                    else:
                        raise ValueError(f"Unexpected labels shape {labels.shape}")
                feats = self._extract_feats(batch, device)

                with self.accelerator.accumulate(self.model):
                    logits, _ = self.model(
                        input_ids=input_ids, attention_mask=attention_mask, domain_ids=domain_ids,
                        train_mode=True, **self._model_feat_kwargs(feats),
                    )
                    loss = criterion(logits, labels)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite loss encountered")
                    self.accelerator.backward(loss)
                    self._step_all_opts()
                    self._step_all_scheds()
                    self._zero_all_opts()

                with torch.no_grad():
                    total_loss += loss.item() * B
                    preds = logits.argmax(dim=1)
                    g_preds = self.accelerator.gather_for_metrics(preds)
                    g_labels = self.accelerator.gather_for_metrics(labels)
                    if self.accelerator.is_main_process:
                        correct += (g_preds == g_labels).sum().item()
                        total += g_labels.size(0)
                    log_batch_training_metrics(
                        epoch=epoch, batch_idx=batch_idx, total_batches=len(train_dataloader),
                        loss=total_loss, correct=correct, total=max(1, total),
                        epoch_start_time=epoch_start_time, start_time=self.start_time,
                        gradient_accumulation_steps=self.gradient_accumulation_steps,
                        batch_size=self.batch_size, epochs=self.epochs,
                        accelerator=self.accelerator, use_wandb=self.use_wandb,
                        current_lr=self._current_lr(), current_step=current_step,
                    )
                    if self.validate_every_n_steps and current_step % self.validate_every_n_steps == 0:
                        val_results = self.validate(val_dataloader, "validation", current_step=current_step)
                        if self.accelerator.is_main_process and val_results is not None:
                            log_validation_results(val_results=val_results, current_step=current_step,
                                                   split_name="validation", accelerator=self.accelerator,
                                                   use_wandb=self.use_wandb)
                    if self.save_every_n_steps and current_step % self.save_every_n_steps == 0:
                        self.save_checkpoint_unified(
                            accelerator=self.accelerator, model=self.model, epoch=epoch,
                            batch_idx=batch_idx, len_train_dataloader=len(train_dataloader),
                            training_strategy=self.global_config.get("TRAINING_STRATEGY"),
                            base_ckpt_dir=self.checkpoint_dir,
                        )

            if self.save_every_n_epochs and (epoch + 1) % self.save_every_n_epochs == 0:
                self.save_checkpoint_unified(
                    accelerator=self.accelerator, model=self.model, epoch=epoch,
                    batch_idx=batch_idx, len_train_dataloader=len(train_dataloader),
                    training_strategy=self.global_config.get("TRAINING_STRATEGY"),
                    base_ckpt_dir=self.checkpoint_dir,
                )
