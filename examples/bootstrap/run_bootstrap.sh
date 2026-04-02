#!/bin/bash
# ── Paths (edit these) ────────────────────────────────────────────────────────
CLS_JSON="/scratch/keane/human_behaviour/grpo_full/global_step_600_test/cls_val_generations_600.json"
LLM_JSON="/scratch/keane/human_behaviour/grpo_full/global_step_600_test/full_test_or_val_generation_outputs/step_600_llm_grading_results.json"
LABEL_MAP="/home/keaneong/human-behavior/verl/sft/label_maps/unified_label_map.json"
OUTPUT_JSON="/home/keaneong/human-behavior/verl/examples/bootstrap/grpo_test_eval_metrics_bootstrap.json"
N_BOOT=1000
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

python "$SCRIPT_DIR/eval_metrics_bootstrap.py" \
  --cls_json  "$CLS_JSON"  \
  --llm_json  "$LLM_JSON"  \
  --label_map "$LABEL_MAP" \
  --output    "$OUTPUT_JSON" \
  --n_boot    "$N_BOOT"
