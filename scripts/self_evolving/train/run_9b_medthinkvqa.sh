#!/usr/bin/env bash
# 9B multi-image RL on MedThinkVQA (bio-nlp-umass/MedThinkVQA).
#
#   RETRIEVAL=0 bash scripts/self_evolving/train/run_9b_medthinkvqa.sh   # baseline: plain GRPO
#   RETRIEVAL=1 bash scripts/self_evolving/train/run_9b_medthinkvqa.sh   # + medical KB retrieval
#
# WHY THIS RUN IS SIMPLER THAN THE HEALTHBENCH ONES, and why that is the point.
# The answer is one of five letters, so the reward is a string comparison:
# verl/utils/reward_score/medthinkvqa.py, no judge, no rubric, no API cost. Every
# failure mode that has eaten HealthBench runs -- judge outages that look like bad
# answers, rubrics a curriculum can quietly make easier, a reward that drifts
# because the grader drifts -- simply does not exist here. The baseline needs no
# gen server and no TRAPI key at all.
#
# WHAT THE NUMBER HAS TO BEAT. Five options means random guessing scores 0.20, and
# the train answers are not uniform (D is 25.4%), so a policy that learns the
# label prior alone scores ~0.254. `reward/guess_rate/mean` is logged so that is
# visible rather than inferred. Any claim of learning has to clear 0.254, not 0.
#
# IMAGES. The dataset caps at 4 per case (see build_medthinkvqa.py), which drops
# 55% of available train images and 45% of val ones -- the benchmark's own README
# reports accuracy rising with image count, so this configuration's ceiling sits
# below the published one. That is a context-budget trade, and it is recorded per
# row (n_images_total vs n_images_used) rather than assumed away.
#
# Image token cost is bounded by construction: images are pre-resized to a 768px
# long side, so each costs ~190 tokens and four ~760 — small against the prompt
# budget. That is why max_prompt_length can stay at 8192 despite the clinical
# history being long.
#
# 9B quirks carried over from the HealthBench runs: head_dim=256 breaks the
# FlashAttention varlen kernel -> use_remove_padding=False + sdpa.
set -xeuo pipefail

S=/scratch/sheng/self_evolving
REPO=${REPO:-$S/verl_healthbench}
cd "$REPO"

RETRIEVAL="${RETRIEVAL:?set RETRIEVAL=1 (retrieval arm) or RETRIEVAL=0 (baseline)}"
DATA=${DATA:-/scratch/sheng/medthinkvqa}
EXP="${EXP:-mtv9b_$([ "$RETRIEVAL" = 1 ] && echo retrieval || echo base)}"
LOGDIR=$S/logs_mtv; mkdir -p "$LOGDIR"

TRAIN=$DATA/medthinkvqa_train.parquet
VAL=$DATA/medthinkvqa_val.parquet
[ -f "$TRAIN" ] || { echo "FATAL: $TRAIN missing — run build_medthinkvqa.py first" >&2; exit 1; }
[ -f "$VAL" ]   || { echo "FATAL: $VAL missing — run build_medthinkvqa.py first" >&2; exit 1; }

GEN_PORT="${GEN_PORT:-8042}"
SUMM_PORT="${SUMM_PORT:-8198}"
SUMM_BASE="${SUMM_BASE:-http://localhost:$SUMM_PORT/v1}"
SUMM_MODEL="${SUMM_MODEL:-Qwen/Qwen3.5-9B}"
EMBED_BASE="${EMBED_BASE:-http://mib.media.mit.edu:18001/v1}"
MILVUS_URI="${MILVUS_URI:-http://mib.media.mit.edu:19531}"

export HF_HOME=$S/hf_cache
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0
export RAY_ADDRESS=local
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_API_KEY="${WANDB_API_KEY:?export WANDB_API_KEY first}"
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=600
export VLLM_ENGINE_ITERATION_TIMEOUT_S=600

# wandb id policy: resume only when a checkpoint actually exists, else a fresh id
# (resume=allow on an existing id resumes FROM ITS LAST STEP, so a from-scratch
# restart under the old id has every metric silently rejected).
_CKPT_DIR="$S/checkpoints/mtv9b/$EXP"
_RUNID_FILE="$LOGDIR/.wandb_runid.$EXP"
if [ -f "$_CKPT_DIR/latest_checkpointed_iteration.txt" ] && [ -s "$_RUNID_FILE" ]; then
    export WANDB_RUN_ID="$(cat "$_RUNID_FILE")"
else
    export WANDB_RUN_ID="${EXP}_$(date +%m%d_%H%M)"
    printf '%s' "$WANDB_RUN_ID" > "$_RUNID_FILE"
fi
export WANDB_RESUME=allow

MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-8192}"     # 4 images ~760 tok + history + options
MAX_RESP_LEN="${MAX_RESP_LEN:-4096}"
ROLLOUT_MAX_LEN="${ROLLOUT_MAX_LEN:-12288}"  # prompt + response
# HARD RULE: >= max_prompt + max_response, else rearrange_micro_batches asserts on
# the first long rollout (the assert is on the longest ACTUAL sequence).
PPO_MAX_TOKEN_LEN="${PPO_MAX_TOKEN_LEN:-12288}"
LOGPROB_MAX_TOKEN_LEN="${LOGPROB_MAX_TOKEN_LEN:-12288}"

cleanup() { kill ${GEN_PID:-} ${SUMM_PID:-} 2>/dev/null || true; }
trap cleanup EXIT INT TERM

AGENT_ARGS=()
if [ "$RETRIEVAL" = 1 ]; then
    curl -sf -m 10 "$EMBED_BASE/models" >/dev/null \
        || { echo "FATAL: embed server unreachable at $EMBED_BASE" >&2; exit 1; }
    export RETRIEVAL_URL="http://localhost:$GEN_PORT/retrieve"
    export VERL_MAX_SEARCHES="${VERL_MAX_SEARCHES:-2}"
    export VERL_SEARCH_THINK_BUDGET="${VERL_SEARCH_THINK_BUDGET:-768}"
    export VERL_THINK_BUDGET_TOKENS="${VERL_THINK_BUDGET_TOKENS:-2048}"
    export VERL_ANSWER_RESERVE_TOKENS="${VERL_ANSWER_RESERVE_TOKENS:-1024}"
    export VERL_MIN_ANSWER_TOKENS="${VERL_MIN_ANSWER_TOKENS:-256}"
    VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.30}"
    AGENT_ARGS=(
        actor_rollout_ref.rollout.multi_turn.enable=True
        actor_rollout_ref.rollout.multi_turn.format=qwen3_coder
        actor_rollout_ref.rollout.multi_turn.max_tool_response_length="${MAX_TOOL_RESP:-6000}"
        actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side=right
        actor_rollout_ref.rollout.multi_turn.tool_config_path=scripts/self_evolving/train/config/medical_retrieval_tool.yaml
        actor_rollout_ref.rollout.agent.default_agent_loop=retrieval_tool_agent
    )
    # Frozen 9B for the retrieval summarizer (same weights the actor starts from,
    # never the live actor: a summarizer that tracks the policy makes the evidence
    # brief non-stationary).
    if ! curl -sf -m 5 "$SUMM_BASE/models" >/dev/null 2>&1; then
        CUDA_VISIBLE_DEVICES="${SUMM_GPU:-3}" MODEL="$SUMM_MODEL" PORT="$SUMM_PORT" \
        TP=1 MEM="${SUMM_MEM:-0.16}" MAXSEQS=96 MAXLEN=16384 \
            bash scripts/self_evolving/serve/serve_summarizer.sh \
            > "$LOGDIR/frozen9b_${EXP}.log" 2>&1 &
        SUMM_PID=$!
        start=$SECONDS
        until curl -sf -m 5 "$SUMM_BASE/models" >/dev/null; do
            kill -0 "$SUMM_PID" 2>/dev/null || { echo "FATAL: frozen 9B died" >&2; tail -40 "$LOGDIR/frozen9b_${EXP}.log"; exit 1; }
            (( SECONDS - start > 1800 )) && { echo "FATAL: frozen 9B unhealthy" >&2; exit 1; }
            sleep 5
        done
    fi
    # Retrieval server. --rubric_mode is NOT set: this run generates no tasks, it
    # only needs /retrieve over the medical KB.
    if curl -sf -m 5 "localhost:$GEN_PORT/healthz" >/dev/null 2>&1; then
        echo "FATAL: port $GEN_PORT already serving (orphan?)" >&2; exit 1
    fi
    /usr/local/bin/python scripts/self_evolving/generation_server.py \
        --embed_api_base "$EMBED_BASE" --embed_model Qwen/Qwen3-VL-Embedding-2B \
        --milvus_uri "$MILVUS_URI" --milvus_token root:Milvus \
        --milvus_collection medical_knowledge_v2 \
        --retrieve_top_k "${RETRIEVE_TOP_K:-5}" --retrieve_total "${RETRIEVE_TOTAL:-16}" \
        --summarizer_api_base "$SUMM_BASE" --summarizer_model "$SUMM_MODEL" \
        --summarizer_provider vllm \
        --workers "${GEN_WORKERS:-4}" --log_dir "$LOGDIR/$EXP" \
        --host 0.0.0.0 --port "$GEN_PORT" \
        > "$LOGDIR/gen_server_${EXP}.log" 2>&1 &
    GEN_PID=$!
    start=$SECONDS
    until curl -sf -m 5 "localhost:$GEN_PORT/healthz" >/dev/null; do
        kill -0 "$GEN_PID" 2>/dev/null || { echo "FATAL: retrieval server died" >&2; tail -40 "$LOGDIR/gen_server_${EXP}.log"; exit 1; }
        (( SECONDS - start > 900 )) && { echo "FATAL: retrieval server unhealthy" >&2; exit 1; }
        sleep 5
    done
    # Prove retrieval works BEFORE spending a step: a silent fallback to raw
    # passages changes the whole run's context distribution and would read as a
    # modelling result.
    curl -sf -m 180 -X POST "localhost:$GEN_PORT/retrieve" -H 'content-type: application/json' \
        -d '{"question":"bony protuberance dorsal carpometacarpal joint on MRI",
             "queries":["carpal boss carpometacarpal bossing imaging",
                        "dorsal wrist mass MRI differential osteophyte"]}' \
      | /usr/local/bin/python -c '
import json, sys
d = json.load(sys.stdin)
assert d["n_merged"] >= 4, d
assert d["summarized"] is True, "summarizer NOT active: %s" % (d.get("fallback_reason"),)
print("retrieve smoke OK: %sq -> %s passages -> %s chars"
      % (d["n_queries"], d["n_merged"], d["chars"]))
' || { echo "FATAL: /retrieve smoke failed" >&2; exit 1; }
else
    VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.45}"
fi

/usr/local/bin/python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    algorithm.norm_adv_by_std_in_grpo=False \
    data.train_files="$TRAIN" \
    data.val_files="$VAL" \
    data.train_batch_size="${TRAIN_BS:-32}" \
    data.max_prompt_length="$MAX_PROMPT_LEN" \
    data.max_response_length="$MAX_RESP_LEN" \
    +data.apply_chat_template_kwargs.enable_thinking=True \
    data.shuffle=True \
    ++data.val_max_samples="${VAL_MAX:--1}" \
    data.truncation=left \
    data.return_raw_chat=True \
    data.dataloader_num_workers=8 \
    data.image_key=images \
    reward.custom_reward_function.path=verl/utils/reward_score/medthinkvqa.py \
    reward.custom_reward_function.name=compute_score \
    reward.reward_manager.name=dapo \
    actor_rollout_ref.model.path="${ACTOR_MODEL_PATH:-Qwen/Qwen3.5-9B}" \
    actor_rollout_ref.model.use_remove_padding=False \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr="${LR:-1e-6}" \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$PPO_MAX_TOKEN_LEN" \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.ref.use_torch_compile=False \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.2 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    +trainer.filter_zero_variance_groups=True \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP:-2}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="$VLLM_GPU_UTIL" \
    actor_rollout_ref.rollout.max_model_len="$ROLLOUT_MAX_LEN" \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.val_kwargs.n="${VAL_N:-1}" \
    actor_rollout_ref.rollout.val_kwargs.do_sample="${VAL_DO_SAMPLE:-True}" \
    actor_rollout_ref.rollout.val_kwargs.temperature="${VAL_TEMP:-1.0}" \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.attention_config.flash_attn_version=2 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.attention_config.use_trtllm_attention=False \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_encoder_attn_backend=TORCH_SDPA \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$LOGPROB_MAX_TOKEN_LEN" \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="$LOGPROB_MAX_TOKEN_LEN" \
    critic.enable=False \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.total_epochs=100 \
    trainer.total_training_steps="${STEPS:-60}" \
    trainer.test_freq=5 \
    trainer.save_freq=20 \
    trainer.val_before_train=True \
    +trainer.max_actor_ckpt_to_keep=1 \
    trainer.resume_mode=auto \
    trainer.default_local_dir="$_CKPT_DIR" \
    +trainer.rollout_data_dir="$LOGDIR/rollouts/$EXP" \
    +trainer.validation_data_dir="$LOGDIR/val_generations/$EXP" \
    trainer.project_name=medthinkvqa \
    trainer.experiment_name="$EXP" \
    'trainer.logger=["console","wandb"]' \
    +ray_init.address=local \
    "${AGENT_ARGS[@]}" \
    "$@"
rc=$?
kill "${GEN_PID:-}" "${SUMM_PID:-}" 2>/dev/null || true
exit $rc
