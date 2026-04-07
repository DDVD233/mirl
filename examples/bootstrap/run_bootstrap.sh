#!/bin/bash
# ── Global settings ────────────────────────────────────────────────────────────
LABEL_MAP="/home/keaneong/human-behavior/verl/sft/label_maps/unified_label_map.json"
OUTPUT_JSON="/home/keaneong/human-behavior/verl/examples/bootstrap/v3_multi_model_eval_metrics_bootstrap.json"
OUTPUT_MD="/home/keaneong/human-behavior/verl/examples/bootstrap/v3_multi_model_eval_metrics_bootstrap.md"
N_BOOT=100000
# Use 0.10 only if you want to label exploratory/lenient McNemar significance.
MCNEMAR_ALPHA=0.10

# Which model name (must match a "name" entry below) is YOUR method for McNemar
METHOD_NAME="harpo_method"

# W&B settings (leave blank to skip upload)
WANDB_PROJECT="hb_rebuttal"
WANDB_ENTITY=""
WANDB_RUN_NAME="v4_multi_model_bootstrap"
WANDB_ARTIFACT_NAME="v3_multi_model_bootstrap_eval"
# ──────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/bootstrap_config_tmp.json"

# ── Model definitions ──────────────────────────────────────────────────────────
# Add or remove entries below. Each entry needs:
#   "name"     : display name used in tables / McNemar output
#   "cls_json" : path to classifier predictions JSON
#   "llm_json" : path to LLM grading results JSON
#
# The model whose "name" matches METHOD_NAME is YOUR method;
# all others are treated as baselines for the McNemar test.
cat > "$CONFIG_FILE" << EOF
{
  "method": "${METHOD_NAME}",
  "models": [
    {
      "name": "harpo_method",
      "cls_json": "/scratch/keane/human_behaviour/HARPO_Hier_Ema_Rerun/global_step_400_test/cls_val_generations_400.json",
      "llm_json": "/scratch/keane/human_behaviour/HARPO_Hier_Ema_Rerun/global_step_400_test/full_test_or_val_generation_outputs/step_400_llm_grading_results.json"
    },
    {
      "name": "rloo_baseline",
      "cls_json": "/scratch/keane/human_behaviour/rloo_global_step_350_test/cls_val_generations_350.json",
      "llm_json": "/scratch/keane/human_behaviour/rloo_global_step_350_test/full_test_or_val_generation_outputs/step_350_llm_grading_results.json"
    },
    {
      "name": "gpg_baseline",
      "cls_json": "/scratch/keane/human_behaviour/gpg_global_step_300_test/global_step_300_test/cls_val_generations_300.json",
      "llm_json": "/scratch/keane/human_behaviour/gpg_global_step_300_test/global_step_300_test/full_test_or_val_generation_outputs/step_300_llm_grading_results.json"
    },
    {
      "name": "grpo_baseline",
      "cls_json": "/scratch/keane/human_behaviour/grpo_full/global_step_600_test/cls_val_generations_600.json",
      "llm_json": "/scratch/keane/human_behaviour/grpo_full/global_step_600_test/full_test_or_val_generation_outputs/step_600_llm_grading_results.json"
    }
  ]
}
EOF
# ──────────────────────────────────────────────────────────────────────────────

WANDB_ARGS=""
[ -n "$WANDB_PROJECT" ]       && WANDB_ARGS="--wandb_project $WANDB_PROJECT"
[ -n "$WANDB_ENTITY" ]        && WANDB_ARGS="$WANDB_ARGS --wandb_entity $WANDB_ENTITY"
[ -n "$WANDB_RUN_NAME" ]      && WANDB_ARGS="$WANDB_ARGS --wandb_run_name $WANDB_RUN_NAME"
[ -n "$WANDB_ARTIFACT_NAME" ] && WANDB_ARGS="$WANDB_ARGS --wandb_artifact_name $WANDB_ARTIFACT_NAME"

python "$SCRIPT_DIR/eval_metrics_bootstrap.py" \
  --config    "$CONFIG_FILE" \
  --label_map "$LABEL_MAP"   \
  --output    "$OUTPUT_JSON" \
  --output_md "$OUTPUT_MD"   \
  --n_boot    "$N_BOOT"      \
  --mcnemar_alpha "$MCNEMAR_ALPHA" \
  $WANDB_ARGS

# Clean up temp config
rm -f "$CONFIG_FILE"
