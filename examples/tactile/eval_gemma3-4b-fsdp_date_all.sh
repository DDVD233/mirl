#!/usr/bin/env bash
# Evaluate the gemma-3-4b FSDP checkpoint (latest step) on the 4 split-date
# test variants in a single launch:
#   1. annotation_verl_split_date_test.json             (full)
#   2. annotation_verl_split_date_test_no_tactile.json  (no tactile)
#   3. annotation_verl_split_date_test_no_video.json    (no video)
#   4. annotation_verl_split_date_test_single_view.json (single view)
#
# Steps:
#   - merge the FSDP shards under checkpoints/.../actor into an HF dir (once)
#   - loop through the 4 test sets and run val_only=True against each
#
# Usage (inside the apptainer container):
#   bash examples/tactile/eval_gemma3-4b-fsdp_date_all.sh
# Or via the apptainer wrapper:
#   IMAGE_TAG=verlai/verl:vllm017.latest \
#     bash examples/tactile/run_apptainer.sh examples/tactile/eval_gemma3-4b-fsdp_date_all.sh

set -x
set -e

PROJECT_NAME="tactile"
EXPERIMENT_NAME="gemma3_4b_dapo_split_date_fsdp"
CKPT_ROOT="checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}"
LATEST_STEP=$(cat "${CKPT_ROOT}/latest_checkpointed_iteration.txt")
FSDP_DIR="${CKPT_ROOT}/global_step_${LATEST_STEP}/actor"
HF_OUT="outputs/eval_gemma3_4b_dapo_split_date_step${LATEST_STEP}"

# 1) merge FSDP shards -> HF directory (once)
if [ ! -f "${HF_OUT}/model.safetensors" ] && [ ! -f "${HF_OUT}/model.safetensors.index.json" ]; then
    mkdir -p "${HF_OUT}"
    # the merger writes weights into target_dir; keep the existing HF configs/tokenizer too
    cp -n "${FSDP_DIR}/huggingface/"* "${HF_OUT}/" 2>/dev/null || true
    python3 -m verl.model_merger merge \
        --backend fsdp \
        --local_dir "${FSDP_DIR}" \
        --target_dir "${HF_OUT}"
fi

DATA_DIR="${HOME}/scratch/raofu/3DHaptic"
declare -a VARIANTS=("full" "no_tactile" "no_video" "single_view")
declare -A VARIANT_PATHS=(
    [full]="${DATA_DIR}/annotation_verl_split_date_test.json"
    [no_tactile]="${DATA_DIR}/annotation_verl_split_date_test_no_tactile.json"
    [no_video]="${DATA_DIR}/annotation_verl_split_date_test_no_video.json"
    [single_view]="${DATA_DIR}/annotation_verl_split_date_test_single_view.json"
)

ROLLOUT_TP=${ROLLOUT_TP:-2}

for VARIANT in "${VARIANTS[@]}"; do
    TEST_PATH="${VARIANT_PATHS[$VARIANT]}"
    EVAL_NAME="eval_gemma3_4b_dapo_split_date_step${LATEST_STEP}_${VARIANT}"
    EVAL_OUT="outputs/${EVAL_NAME}"
    mkdir -p "${EVAL_OUT}"
    echo "============================================"
    echo "Evaluating variant: ${VARIANT}"
    echo "  test file: ${TEST_PATH}"
    echo "  output:    ${EVAL_OUT}"
    echo "============================================"

    python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        algorithm.use_kl_in_reward=False \
        data.train_files="${TEST_PATH}" \
        data.val_files="${TEST_PATH}" \
        data.train_batch_size=64 \
        data.val_batch_size=32 \
        data.max_prompt_length=8192 \
        data.max_response_length=4096 \
        data.filter_overlong_prompts=False \
        data.truncation='left' \
        +data.video_as_frames=4 \
        actor_rollout_ref.model.path="${HF_OUT}" \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.ppo_mini_batch_size=64 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
        actor_rollout_ref.actor.use_dynamic_bsz=True \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=12288 \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.01 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.actor.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
        actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=1 \
        +actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[Gemma3DecoderLayer] \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP} \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
        actor_rollout_ref.rollout.enable_chunked_prefill=False \
        actor_rollout_ref.rollout.enforce_eager=False \
        actor_rollout_ref.rollout.load_format=auto \
        actor_rollout_ref.rollout.n=5 \
        actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
        actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
        actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=12288 \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
        actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
        actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=12288 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        actor_rollout_ref.ref.fsdp_config.ulysses_sequence_parallel_size=1 \
        +actor_rollout_ref.ref.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[Gemma3DecoderLayer] \
        reward_model.reward_manager=dapo \
        +reward_model.reward_kwargs.overlong_buffer_cfg.enable=True \
        +reward_model.reward_kwargs.overlong_buffer_cfg.len=512 \
        +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
        +reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
        +reward_model.reward_kwargs.max_resp_len=2048 \
        trainer.critic_warmup=0 \
        trainer.balance_batch=True \
        trainer.logger='["console","wandb"]' \
        trainer.project_name="${PROJECT_NAME}" \
        trainer.experiment_name="${EVAL_NAME}" \
        trainer.n_gpus_per_node=4 \
        trainer.nnodes=1 \
        trainer.save_freq=-1 \
        trainer.test_freq=1 \
        trainer.val_only=True \
        trainer.log_val_generations=10 \
        trainer.validation_data_dir="${EVAL_OUT}" \
        trainer.total_epochs=1
done

echo ""
echo "All four evaluations done."
