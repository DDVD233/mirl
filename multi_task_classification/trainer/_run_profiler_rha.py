# profile_rha_overhead.py

import copy
from rha_profiler_trainer import RHAMultiHeadOmniClassifierProfiler
from your_config_loader import load_global_config
from your_model_builder import build_model_tokenizer_processor

def build_profiler_trainer(overrides: dict):
    base_cfg = load_global_config()
    cfg = copy.deepcopy(base_cfg)
    cfg.update(overrides)

    tokenizer, processor, model = build_model_tokenizer_processor(cfg)

    trainer = RHAMultiHeadOmniClassifierProfiler(
        data_files=cfg["TRAIN_FILES"],
        val_data_files=cfg["VAL_FILES"],
        test_data_files=cfg.get("TEST_FILES", []),
        tokenizer=tokenizer,
        processor=processor,
        config=cfg["DATASET_CONFIG"],
        batch_size=cfg["TRAIN_BATCH_SIZE"],
        val_batch_size=cfg["VAL_BATCH_SIZE"],
        test_batch_size=cfg.get("TEST_BATCH_SIZE", cfg["VAL_BATCH_SIZE"]),
        lr=cfg["LR"],
        epochs=1,
        save_checkpoint_dir=cfg["SAVE_DIR"],
        load_checkpoint_path=cfg.get("LOAD_CKPT", None),
        model=model,
        gradient_accumulation_steps=cfg["GRAD_ACCUM_STEPS"],
        num_workers=cfg.get("NUM_WORKERS", 0),
        use_lora=cfg.get("USE_LORA", False),
        global_config=cfg,
    )

    return trainer

def safe_delta(a, b):
    if a is None or b is None:
        return None
    return b - a

def main():
    # -------- baseline (no adapters) ----------
    trainer_base = build_profiler_trainer({
        "USE_RLA_VIDEO": False,
        "USE_RLA_AUDIO": False,
        "RLA_STAGE": "base_only",
    })
    base_stats = trainer_base.profile_forward_cost(
        num_batches=10,
        split="val",
        description="base_only",
    )

    # -------- with adapters ----------
    trainer_rha = build_profiler_trainer({
        "USE_RLA_VIDEO": True,
        "USE_RLA_AUDIO": True,          # or your actual combo
        "RLA_STAGE": "residual_only",   # or "joint" / "residual_and_head"
    })
    rha_stats = trainer_rha.profile_forward_cost(
        num_batches=10,
        split="val",
        description="with_rha",
    )

    print("\n=== Adapter overhead (RHA vs base) ===")
    print(f"Δ mean latency (s): {safe_delta(base_stats['mean_latency_s'], rha_stats['mean_latency_s'])}")
    print(f"Δ peak VRAM (MiB):  {safe_delta(base_stats['peak_vram_mb'], rha_stats['peak_vram_mb'])}")

if __name__ == "__main__":
    main()