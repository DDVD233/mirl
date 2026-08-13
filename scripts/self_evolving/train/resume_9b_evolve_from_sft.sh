#!/usr/bin/env bash
# Resume the 9B mimic-rare RL run (wandb 555gr3yi / EXP
# mimiciv_rare_qwen35_9b_evolve_from_sft) after a crash, on the 2-GPU pod (:2333,
# david-evolving-gljtx-2-gpus-0).
#
# WHY THIS FILE EXISTS. Resuming it once meant reconstructing the launch from a
# `set -x` trace in a dead log, and two things silently break a naive restart:
#
#   1. THE GEN SERVER ON :8006 MUST BE UP FIRST, and its start script defaults to
#      MODEL_NAME=Qwen/Qwen3.6-27B while the teacher endpoint (:18184) now serves
#      ONLY Qwen/Qwen3.5-9B. Starting it with the default leaves a healthy-looking
#      server whose every generation call 404s against a model that is not there.
#
#   2. THE TRAINER SCRIPT HARDCODES 4 GPUs (run_qwen36_27b_selfimprove_from_sft.sh
#      sets trainer.n_gpus_per_node=4 and CUDA_VISIBLE_DEVICES=0,1,2,3) but this
#      pod has 2, so it dies at init with "Total available GPUs 2.0 is less than
#      total desired GPUs 4". The original run passed n_gpus_per_node=2 as a
#      TRAILING hydra override -- hydra takes the last occurrence -- along with
#      nine others. All ten are reproduced below; dropping any of them silently
#      changes the config the run was training under.
#
# LOG-PROB MICRO-BATCH: 12288, which is max_prompt(8192) + max_response(4096).
# That is a HARD FLOOR, not a tuning knob: rearrange_micro_batches asserts
# max_token_len >= the longest ACTUAL sequence, so 8192 died with "Got
# max_token_len=8192 and max_seq_len=8215" after 433 otherwise-healthy steps. The
# memory that made 16384 OOM is fixed properly in transformer_impl.py instead --
# the non-remove-padding branch now honors entropy_from_logits_with_chunking, which
# it had been silently ignoring.
#
# (Historical note: 8192 was tried, down from the original 16384.) The first resume OOMed
# in entropy_from_logits during the log-prob INFERENCE pass, asking for 52.54 GiB
# with 21.76 free. At a ~152k vocab a single logits tensor for 16384 tokens is
# ~10 GB in fp32 and the entropy op needs several of them at once. Note the
# original run's ++actor.entropy_from_logits_with_chunking=True does NOT cover
# this path -- it applies to the ACTOR UPDATE, while this OOM is in
# compute_log_prob -> infer_batch, so the only lever that bites here is the
# micro-batch length.
#
# STEP BUDGET: effectively unbounded by default. verl has no -1 sentinel; when
# total_training_steps is set it simply wins over epochs*len(dataloader), and a
# run that hits it exits as "finished" (which is how the 60-step spec-gap run
# ended). warmup_style is null so the LR is constant -- a large budget does not
# stretch any decay schedule, which is what would otherwise make this unsafe.
#
# resume_mode=auto picks up the latest checkpoint by itself. WANDB_RUN_ID is
# pinned so the curve continues the existing run instead of starting a new one;
# steps already logged before the crash are skipped by wandb and recording
# resumes past them.
#
#   bash scripts/self_evolving/train/resume_9b_evolve_from_sft.sh
set -xeuo pipefail

S=/scratch/sheng/self_evolving
REPO=${REPO:-$S/verl_healthbench}
cd "$REPO"

WANDB_RUN_ID="${WANDB_RUN_ID:-555gr3yi}"
export WANDB_RUN_ID WANDB_RESUME=allow
export WANDB_API_KEY="${WANDB_API_KEY:-$(cat $S/.wandb_key_dvd)}"
export TOTAL_STEPS="${TOTAL_STEPS:-1000000}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-10000}"
export ACTOR_MODEL_PATH="${ACTOR_MODEL_PATH:-$S/checkpoints/self_evolving_medical/mimiciv_rare_qwen35_9b_sft_distill/global_step_90/actor/huggingface}"
CKPT_DIR=$S/checkpoints/self_evolving_medical/mimiciv_rare_qwen35_9b_evolve_from_sft
TEACHER_BASE="${TEACHER_BASE:-http://point.dd.works:18184/v1}"

[ -d "$ACTOR_MODEL_PATH" ] || { echo "FATAL: SFT init dir missing: $ACTOR_MODEL_PATH" >&2; exit 1; }
echo "resuming from checkpoint iteration: $(cat "$CKPT_DIR/latest_checkpointed_iteration.txt" 2>/dev/null || echo NONE)"

# The teacher/judge must be up AND must serve the model the gen server will ask
# for. Read the served id rather than assuming it.
TEACHER_MODEL=$(curl -sf -m 10 "$TEACHER_BASE/models" \
    | /usr/local/bin/python -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])') \
    || { echo "FATAL: teacher/judge unreachable at $TEACHER_BASE" >&2; exit 1; }
echo "teacher/judge serving: $TEACHER_MODEL"

if curl -sf -m 5 localhost:8006/healthz 2>/dev/null | grep -q '"ok":true'; then
    echo "gen server already healthy on :8006 - reusing"
else
    echo "starting gen server on :8006 with MODEL_NAME=$TEACHER_MODEL"
    tmux kill-window -t pipe:gensrv 2>/dev/null || true
    tmux has-session -t pipe 2>/dev/null || tmux new-session -d -s pipe -n idle
    tmux new-window -t pipe -n gensrv \
        "cd $REPO && MODEL_NAME='$TEACHER_MODEL' bash scripts/self_evolving/serve/start_gen_server_mimic_s1.sh \
         > $S/logs_9b_phase1/gen_server_resume.log 2>&1"
    for i in $(seq 1 180); do
        curl -sf -m 3 localhost:8006/healthz 2>/dev/null | grep -q '"ok":true' && break
        sleep 5
    done
    curl -sf -m 5 localhost:8006/healthz 2>/dev/null | grep -q '"ok":true' \
        || { echo "FATAL: gen server not healthy after ~15m" >&2; exit 1; }
fi

# The ten trailing overrides the original launch used, recovered from its set -x
# trace. n_gpus_per_node=2 MUST come after the script's own =4.
exec bash scripts/self_evolving/train/run_qwen35_9b_evolve_from_sft.sh \
    trainer.n_gpus_per_node=2 \
    actor_rollout_ref.actor.policy_loss.loss_mode=vanilla \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_custom_all_reduce=True \
    ++actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=12288 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${LOGPROB_MAX_TOKEN_LEN:-12288}" \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${LOGPROB_MAX_TOKEN_LEN:-12288}" \
    ++reward.custom_reward_function.reward_kwargs.model_name="$TEACHER_MODEL" \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    "$@"
