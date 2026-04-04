#!/bin/bash
# ── Paths (edit these) ────────────────────────────────────────────────────────
CLS_JSON="/scratch/keane/human_behaviour/HARPO_Hier_Ema_Rerun/global_step_400_test/cls_val_generations_400.json"
LLM_JSON="/scratch/keane/human_behaviour/HARPO_Hier_Ema_Rerun/global_step_400_test/full_test_or_val_generation_outputs/step_400_llm_grading_results.json"
LABEL_MAP="/home/keaneong/human-behavior/verl/sft/label_maps/unified_label_map.json"
OUTPUT_JSON="/home/keaneong/human-behavior/verl/examples/bootstrap/harpo_400_test_eval_metrics_bootstrap.json"
N_BOOT=1000
WANDB_PROJECT="hb_rebuttal"           # set to your W&B project name to upload artifact, e.g. "human-behavior"
WANDB_ENTITY=""            # set to your W&B entity (team/user), or leave blank
WANDB_RUN_NAME="harpo_test_step_400_bootstrap"          # optional run name, e.g. "grpo_step600_bootstrap"
WANDB_ARTIFACT_NAME="harpo_test_step_400_bootstrap_eval"
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

WANDB_ARGS=""
[ -n "$WANDB_PROJECT" ] && WANDB_ARGS="--wandb_project $WANDB_PROJECT"
[ -n "$WANDB_ENTITY" ]  && WANDB_ARGS="$WANDB_ARGS --wandb_entity $WANDB_ENTITY"
[ -n "$WANDB_RUN_NAME" ] && WANDB_ARGS="$WANDB_ARGS --wandb_run_name $WANDB_RUN_NAME"
[ -n "$WANDB_ARTIFACT_NAME" ] && WANDB_ARGS="$WANDB_ARGS --wandb_artifact_name $WANDB_ARTIFACT_NAME"

python "$SCRIPT_DIR/eval_metrics_bootstrap.py" \
  --cls_json  "$CLS_JSON"  \
  --llm_json  "$LLM_JSON"  \
  --label_map "$LABEL_MAP" \
  --output    "$OUTPUT_JSON" \
  --n_boot    "$N_BOOT" \
  $WANDB_ARGS
