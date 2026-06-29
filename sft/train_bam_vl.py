"""
Entry point for BAM-on-Qwen3-VL training (parallel to train_bam.py for Omni).

  --task_type qa   ->  BAMVLQA  model (teacher-forcing LM loss)   [primary]
  --task_type cls  ->  BAMVLCLS model (multi-head CE loss)

Reads configs/config_bam_vl_accelerate.yaml. Most settings live in the YAML; only a
few high-level overrides are exposed on the CLI for convenience. Native video is
supported via vl_use_native_video=true (requires modalities="videos" in dataset_config).
"""
import os
import sys
import json
import torch
import argparse
from datetime import datetime
from transformers import AutoTokenizer, AutoProcessor
from omegaconf import OmegaConf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.bam_wrapped_qwen3_vl import BAMVLCLS, BAMVLQA
from trainer.bam_vl_trainer import BAMVLTrainer

STREAMS = ("facial", "pose", "audio")
TORCH_DTYPE_MAP = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}


def _parse_n(v):
    """YAML may carry None / 'None' / ints for the every_n_* knobs."""
    if v is None or v == "None":
        return None
    return int(v)


def parse_parameters():
    parser = argparse.ArgumentParser(description="BAM-on-Qwen3.5 training")
    parser.add_argument("--config", type=str, default="configs/config_bam_vl_accelerate.yaml")
    parser.add_argument("--mode", type=str, choices=["train", "test"])
    parser.add_argument("--task_type", type=str, choices=["cls", "qa"])
    parser.add_argument("--train_file", type=str)
    parser.add_argument("--val_file", type=str)
    parser.add_argument("--test_file", type=str)
    parser.add_argument("--label_map_path", type=str)
    parser.add_argument("--save_checkpoint_dir", type=str)
    parser.add_argument("--load_checkpoint_path", type=str)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--bam_stage", type=str,
                        choices=["bam_only", "bam_and_classifier_heads_only", "bam_and_full_model"])
    parser.add_argument("--use_wandb", action="store_true")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    # --- light CLI overrides ---
    if args.mode is not None:                cfg.mode = args.mode
    if args.task_type is not None:           cfg.bam.task_type = args.task_type
    if args.train_file is not None:          cfg.data.train_file = args.train_file
    if args.val_file is not None:            cfg.data.val_file = args.val_file
    if args.test_file is not None:           cfg.data.test_file = args.test_file
    if args.label_map_path is not None:      cfg.data.label_map_path = args.label_map_path
    if args.save_checkpoint_dir is not None: cfg.train.save_checkpoint_dir = args.save_checkpoint_dir
    if args.load_checkpoint_path is not None: cfg.train.load_checkpoint_path = args.load_checkpoint_path
    if args.epochs is not None:              cfg.train.epochs = args.epochs
    if args.bam_stage is not None:           cfg.bam.bam_stage = args.bam_stage
    if args.use_wandb:                       cfg.wandb.use = True

    with open(cfg.data.label_map_path, "r") as f:
        label_config = json.load(f)

    return cfg, label_config


def build_global_config(cfg, label_config):
    bam = cfg.bam
    train = cfg.train

    def b(key, default=None):
        return bam.get(key, default) if hasattr(bam, "get") else getattr(bam, key, default)

    gc = {
        # identity / scheme
        "TASK_TYPE":               bam.get("task_type", "qa"),
        "DATASET_NAME":            bam.get("dataset_name", "default"),
        "TRAINING_STRATEGY":       cfg.model.training_strategy,
        "FULL_LABEL_SCHEME":       label_config,
        "LABEL_MAP":               label_config["label_mapping"],
        "LABEL_MAP_PATH":          cfg.data.label_map_path,
        "NUM_CLASSES":             label_config["num_classes"],
        "label_config":            label_config,
        "LORA_CONFIG":             OmegaConf.to_container(cfg.model.lora_config, resolve=True),
        # scheduler / checkpoint cadence
        "USE_SCHEDULER":           bool(train.use_scheduler),
        "SCHEDULER_TYPE":          train.scheduler_type,
        "WARMUP_STEPS":            train.warmup_steps,
        "VALIDATION_RESULT_DIR":   train.validation_result_dir,
        "VALIDATE_EVERY_N_EPOCHS": _parse_n(train.validate_every_n_epochs),
        "VALIDATE_EVERY_N_STEPS":  _parse_n(train.validate_every_n_steps),
        "SAVE_EVERY_N_EPOCHS":     _parse_n(train.save_every_n_epochs),
        "SAVE_EVERY_N_STEPS":      _parse_n(train.save_every_n_steps),
        "MAX_STEPS":               _parse_n(train.get("max_steps", None)),
        "EARLY_STOPPING_PATIENCE": int(train.early_stopping_patience),
        "BASE_LR":                 float(train.get("base_lr", float(train.lr) * 0.25)),
        "BAM_LR":                  float(train.get("bam_lr", float(train.lr) * 5.0)),
        "MAX_GRAD_NORM":           float(train.get("max_grad_norm", 1.0)),
        # Default: no explicit checkpoint -> fresh run. Set true to auto-resume the latest.
        "RESUME_FROM_LATEST":      bool(train.get("resume_from_latest", False)),
        # wandb
        "USE_WANDB":               bool(cfg.wandb.use),
        "WANDB_PROJECT":           cfg.wandb.project,
        "WANDB_ENTITY":            cfg.wandb.entity,
        # Qwen3-VL is bf16-native; match the bf16 Accelerate launch config.
        "MIXED_PRECISION":         cfg.model.get("mixed_precision", "bf16"),
        # BAM stage / QA
        "BAM_STAGE":               b("bam_stage", "bam_only"),
        "BAM_FRESH_START":         bool(b("bam_fresh_start", False)),
        "QA_LOSS_WEIGHT":          float(b("qa_loss_weight", 1.0)),
        "QA_DATASETS":             list(b("qa_datasets", []) or []),
        "VL_USE_NATIVE_VIDEO":     bool(b("vl_use_native_video", False)),
        "BAM_HIDDEN":              int(b("bam_hidden", 128)),
    }

    # Per-stream BAM settings (facial / pose / audio).
    for s in STREAMS:
        S = s.upper()
        gc[f"USE_BAM_{S}"]        = bool(b(f"use_bam_{s}", False))
        gc[f"D_{S}_FEAT"]         = b(f"d_{s}_feat", None)
        gc[f"BAM_HIDDEN_{S}"]     = int(b(f"bam_hidden_{s}", gc["BAM_HIDDEN"]))
        gc[f"BAM_P_MODDROP_{S}"]  = float(b(f"bam_p_moddrop_{s}", 0.30))
        gc[f"BAM_{S}_TEMPORAL"]   = b(f"bam_{s}_temporal", "meanstd" if s != "audio" else "none")
        gc[f"BAM_{S}_USE_LN"]     = bool(b(f"bam_{s}_use_ln", False))
        gc[f"BAM_{S}_ALPHA_INIT"] = float(b(f"bam_{s}_alpha_init", 1.0))

    # wandb run name: use the explicit wandb.run_name from the config if given; otherwise
    # fall back to a descriptive auto-name (data split + task + stage + which streams are on)
    # so runs in the same project stay comparable.
    explicit_run_name = cfg.wandb.get("run_name", None)
    if explicit_run_name:
        gc["RUN_NAME"] = str(explicit_run_name)
    else:
        split = os.path.splitext(os.path.basename(str(cfg.data.train_file)))[0]
        on_streams = "".join(s[0] for s in STREAMS if gc[f"USE_BAM_{s.upper()}"]) or "none"
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        gc["RUN_NAME"] = f"bamvl_{gc['TASK_TYPE']}_{gc['BAM_STAGE']}_{split}_feats-{on_streams}_{ts}"

    return gc


def main():
    cfg, label_config = parse_parameters()
    gc = build_global_config(cfg, label_config)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.tokenizer_name)
    processor = AutoProcessor.from_pretrained(cfg.model.processor_name)

    task_type = gc["TASK_TYPE"]
    model_cls = BAMVLCLS if task_type == "cls" else BAMVLQA
    torch_dtype = TORCH_DTYPE_MAP.get(cfg.model.torch_dtype, torch.bfloat16)

    print(f"[INFO] Initializing {model_cls.__name__} on {cfg.model.backbone_name} "
          f"(task_type={task_type}, stage={gc['BAM_STAGE']}, num_classes={gc['NUM_CLASSES']})")
    model = model_cls(
        full_label_scheme=label_config,
        freeze_backbone=cfg.model.training_strategy,
        backbone_name=cfg.model.backbone_name,
        backbone_class=cfg.model.get("backbone_class", None),
        attn_implementation=cfg.model.get("attn_implementation", "flash_attention_2"),
        lora_config=gc["LORA_CONFIG"] if cfg.model.training_strategy == "lora" else None,
        device_map=cfg.model.device_map,
        torch_dtype=torch_dtype,
    )

    dataset_config = OmegaConf.create(dict(cfg.dataset_config))

    trainer = BAMVLTrainer(
        data_files=cfg.data.train_file,
        val_data_files=cfg.data.val_file,
        test_data_files=None,
        tokenizer=tokenizer,
        processor=processor,
        config=dataset_config,
        batch_size=int(cfg.train.train_batch_size),
        val_batch_size=int(cfg.train.val_batch_size),
        test_batch_size=int(cfg.train.test_batch_size),
        lr=float(cfg.train.lr),
        epochs=int(cfg.train.epochs),
        save_checkpoint_dir=cfg.train.save_checkpoint_dir,
        load_checkpoint_path=cfg.train.load_checkpoint_path,
        model=model,
        gradient_accumulation_steps=int(cfg.train.gradient_accumulation_steps),
        num_workers=int(cfg.train.num_workers),
        global_config=gc,
    )

    mode = getattr(cfg, "mode", "train")
    if mode == "train":
        trainer.train()
    elif mode == "test":
        trainer.test()
    else:
        raise ValueError(f"Invalid mode: {mode}")


if __name__ == "__main__":
    main()
