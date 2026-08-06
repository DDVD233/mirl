#!/usr/bin/env bash
# Qwen3.5-9B iteration track.
#
# Why 9B: the last five blockers were infrastructure, not capability — a chat-template
# incompatibility, a reward-manager routing bug (verl has TWO dapo.py files), a
# decision gradient that never fired, and an untruncated sampling tail that filled 49%
# of rollouts with CJK/Cyrillic garbage. Each cost a 30-40 min load-and-validate cycle
# at 27B. The open question is now sharp and needs many iterations to answer:
#
#     v13 training reward climbed 0.45 -> 0.53 while HealthBench val sat at
#     0.371 -> 0.360 across 20 steps. The policy improves at our SELF-GENERATED
#     tasks without improving on the benchmark.
#
# That is either a task-distribution transfer failure or a gameable self-judge, and
# 9B lets us test it ~3x faster.
#
# WHAT TRANSFERS from a 9B result: the machinery (decision samples earning real
# rewards, selective retrieval, dumps, rubric quality, judge behaviour).
# WHAT DOES NOT: absolute scores (the 0.3827/0.3752 bars are 27B-only — hence the
# baseline stage below), sampling fragility (model-specific), and above all whether
# retrieval HELPS: a 9B has bigger knowledge gaps so retrieval should help it more,
# meaning a positive 9B result would not certify 27B (a negative one would be damning).
#
# Stages (run one at a time; each needs the 4 GPUs):
#   baseline_noret   stock 9B, retrieval disabled   -> the "can it just answer" bar
#   baseline_ret     stock 9B, retrieval enabled    -> does the tool help at 9B
#   sft              SFT the same 901 GPT-5.6 traces onto 9B, merge to HF
#   rl               RL from the SFT checkpoint with every fix applied
#
#   bash scripts/self_evolving/train/run_9b_track.sh baseline_noret
set -uo pipefail

STAGE="${1:-}"
REPO=${REPO:-/scratch/sheng/self_evolving/verl_healthbench}
cd "$REPO"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-9B}"
TRACES_JSONL="${TRACES_JSONL:-/scratch/sheng/self_evolving/gpt56_sft_traces.jsonl}"
SFT_EXP="${SFT_EXP:-gpt56_sft_qwen35_9b}"
SFT_BASE="/scratch/sheng/self_evolving/checkpoints/self_evolving_medical/$SFT_EXP"
LOGDIR=/scratch/sheng/self_evolving/logs_healthbench_rubric
mkdir -p "$LOGDIR"

# 9B fits comfortably; TP=2 leaves room for a larger rollout batch than the 27B's TP=4.
export ROLLOUT_TP="${ROLLOUT_TP:-2}"
# Truncate the sampling tail. This is what took 27B rollouts from 49.4% garbage to
# 0.2% and training reward from 0.07 to 0.53 — do not run without it.
export ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-0.95}"
export ROLLOUT_TOP_K="${ROLLOUT_TOP_K:-20}"
export VAL_JUDGE_MODEL="${VAL_JUDGE_MODEL:-gpt-chat-latest_2026-05-28}"
export REWARD_JUDGE_CONCURRENCY="${REWARD_JUDGE_CONCURRENCY:-6}"

require_free_gpus() {
  if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; then
    echo "FATAL: GPUs busy — stop the running job first." >&2
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv >&2
    exit 1
  fi
}

case "$STAGE" in
  baseline_noret|baseline_ret)
    require_free_gpus
    EXP="healthbench_9b_${STAGE}"
    LOG="$LOGDIR/${EXP}.log"
    EXTRA=(trainer.val_only=True
           actor_rollout_ref.rollout.top_p="$ROLLOUT_TOP_P"
           actor_rollout_ref.rollout.top_k="$ROLLOUT_TOP_K"
           actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP")
    if [[ "$STAGE" == "baseline_noret" ]]; then
      # Plain single-turn answering: no agent loop, no tool schema in the prompt.
      EXTRA+=(actor_rollout_ref.rollout.multi_turn.enable=False)
      EXTRA+=(~actor_rollout_ref.rollout.agent.default_agent_loop)
    fi
    ACTOR_MODEL_PATH="$BASE_MODEL" EXP="$EXP" \
      bash scripts/self_evolving/train/run_qwen36_27b_healthbench_rubric_v7_retrieval.sh \
      "${EXTRA[@]}" 2>&1 | tee "$LOG"
    echo "== $STAGE done; read acc_len_adj_signed from $LOG =="
    ;;

  sft)
    require_free_gpus
    MODEL_PATH="$BASE_MODEL" TRACES_JSONL="$TRACES_JSONL" EXP="$SFT_EXP" \
      TRAIN_PARQUET=/scratch/sheng/self_evolving/gpt56_sft_traces.parquet \
      SFT_MAX_LENGTH="${SFT_MAX_LENGTH:-12288}" LR="${LR:-1e-5}" \
      bash scripts/self_evolving/train/run_gpt56_sft.sh 2>&1 | tee "$LOGDIR/9b_sft.log"
    LAST=$(ls -d "$SFT_BASE"/global_step_* 2>/dev/null | sort -t_ -k3 -n | tail -1)
    test -n "$LAST" || { echo "FATAL: no 9B SFT checkpoint"; exit 1; }
    python3 -m verl.model_merger merge --backend fsdp \
      --local_dir "$LAST" --target_dir "$LAST/hf_merged" 2>&1 | tee "$LOGDIR/9b_merge.log"
    test -f "$LAST/hf_merged/config.json" || { echo "FATAL: merge produced no config.json"; exit 1; }
    # model_merger writes weights+config only; without the tokenizer files vLLM cannot
    # load the checkpoint at RL start.
    for f in tokenizer.json tokenizer_config.json chat_template.jinja processor_config.json \
             vocab.json merges.txt special_tokens_map.json; do
      [ -f "$LAST/huggingface/$f" ] && cp -n "$LAST/huggingface/$f" "$LAST/hf_merged/$f"
    done
    echo "== 9B SFT merged -> $LAST/hf_merged =="
    ;;

  rl)
    require_free_gpus
    LAST=$(ls -d "$SFT_BASE"/global_step_*/hf_merged 2>/dev/null | sort -t_ -k3 -n | tail -1)
    test -n "$LAST" || { echo "FATAL: run the sft stage first"; exit 1; }
    SFT_CKPT="$LAST" VAL_ONLY=0 \
      EXP="${EXP:-healthbench_9b_rl}" \
      LOG="$LOGDIR/9b_rl.log" \
      ROLLOUT_TP="$ROLLOUT_TP" \
      bash scripts/self_evolving/train/run_v11_gpt56_sft_val.sh
    ;;

  *)
    echo "usage: $0 {baseline_noret|baseline_ret|sft|rl}" >&2
    exit 2
    ;;
esac
