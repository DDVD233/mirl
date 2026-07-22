# ChildPlay ADOS-2 video training (SFT → GRPO)

Trains **Qwen3-VL-8B-Instruct** to score ADOS-2 Module 1 items from 1-minute video
chunks of the ChildPlay dataset. One row of training data = one (video chunk, ADOS
item) pair; the model answers `Reasoning: ... / Score: [x]` and the RL reward is
exact score match (+ small format bonus), see
`verl/utils/reward_score/childplay_ados.py` (`data_source="childplay_ados"`).

Ground-truth scores/justifications come from the per-chunk Gemini-generated
annotation JSONs (`*_ados.json`); the train/val split is grouped by YouTube id
(no video shared across splits). Score `8` ("not assessable") and non-integer
labels are dropped, mirroring the upstream ChildPlay pipeline.

## One-click usage (docker)

```bash
# full pipeline: prep data -> SFT on justification traces -> merge -> GRPO RL
bash scripts/video_training/docker_run.sh all

# RL only, from the base model
bash scripts/video_training/docker_run.sh rl_only

# quick end-to-end smoke test (tiny subsets, few steps, console logging)
SMOKE=1 CUDA_VISIBLE_DEVICES=4,5,6,7 bash scripts/video_training/docker_run.sh rl_only
```

The wrapper builds `mirl-video-training:latest` from
`docker/Dockerfile.video_training` if missing (base:
`zjdavid/verl-selfevolving:cu130-vllm0.22.1`), mounts `/scratch/dvdai` and the
HF cache, and bind-mounts the repo over `/workspace/mirl` so code edits apply
without a rebuild.

## Stages (also runnable bare on the host)

| Stage | What it does |
|---|---|
| `prep` | rsync chunks/annotations/split from `/scratch/emorfin/childplay_dataset` to `/scratch/dvdai/childplay_dataset`, build RL JSONL + SFT parquet + smoke/mini subsets + `stats.json` |
| `sft` | `verl.trainer.sft_trainer` (FSDP engine), assistant target = `Reasoning: <annotation justification>\nScore: [x]` |
| `rl` | GRPO from the merged SFT HF checkpoint |
| `rl_only` | GRPO from the base HF model |
| `all` | prep → sft → `verl.model_merger` → rl |

Key env knobs (all optional): `CUDA_VISIBLE_DEVICES`, `NUM_GPUS`, `SMOKE=1`,
`DATA_DIR`, `RUN_DIR` (checkpoints/logs, default `/scratch/dvdai/childplay_ados`),
`MODEL_PATH`, `TRAIN_BS`, `GRPO_N`, `ROLLOUT_TP`, `GPU_UTIL`, `SFT_LR`, `RL_LR`,
`LOGGER='["console"]'` to disable wandb.

Dataset prep options (`prepare_childplay.py`): `--max-frames 16` (per 1-min chunk),
`--balance-cap N` (cap per-(item,score) train rows; score 0 is ~58% of data),
`--skip-rsync`.

## Files

- `prepare_childplay.py` — one-click dataset processing (copy + format into the
  repo-standard RL JSONL / SFT parquet schemas; relative `chunks/*.mp4` video paths
  resolve against the JSONL's directory both on host and in docker)
- `ados2_module1.json` — ADOS-2 Module 1 rubric bank (34 items) used to build prompts
- `run_childplay_sft.sh` / `run_childplay_grpo.sh` — stage scripts
- `run_pipeline.sh` — stage driver (logs to `$RUN_DIR/logs_pipeline_*.log`)
- `docker_run.sh` — docker wrapper (build-if-missing + mounts)
