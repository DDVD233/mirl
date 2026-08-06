#!/usr/bin/env bash
# Overnight chained fix: SELECTIVE-SFT -> merge -> v7 retrieval RL.
#
# Run AFTER make_selective_traces.py has finished writing
# /scratch/sheng/self_evolving/selective_sft_traces.jsonl. Frees the GPUs held by
# the local teacher (port 8200), SFTs stock Qwen3.6-27B on the selective traces,
# merges to HF, and launches the v7-selective RL run (which does val_before_train =
# the decisive step-0 number). Idempotent-ish; each stage tees its own log.
set -xeuo pipefail

REPO=/scratch/sheng/self_evolving/verl_healthbench
cd "$REPO"

DIRECT_JSONL=/scratch/sheng/self_evolving/direct_sft_traces.jsonl        # new: direct (no-search) traces
RETR_JSONL=/scratch/sheng/self_evolving/retrieval_sft_traces.jsonl        # existing: 600 retrieval traces
TRACES_JSONL=/scratch/sheng/self_evolving/selective_sft_traces.jsonl      # combined mix (built below)
N_RETR=${N_RETR:-200}                                                     # retrieval traces to mix in
SFT_EXP=retrieval_sft_qwen36_27b_selective
SFT_CKPT=/scratch/sheng/self_evolving/checkpoints/self_evolving_medical/$SFT_EXP
LOGDIR=/scratch/sheng/self_evolving/logs_healthbench_rubric

# ---- build the SELECTIVE mix: all direct traces + N_RETR sampled retrieval traces ----
# Teaches the CHOICE: ~60% direct (answer without searching) + ~40% retrieval. Breaks
# the always-retrieve habit baked in by the old 100%-retrieval SFT (v7_wsfix).
test -s "$DIRECT_JSONL" || { echo "FATAL: no direct traces at $DIRECT_JSONL"; exit 1; }
python3 - "$DIRECT_JSONL" "$RETR_JSONL" "$TRACES_JSONL" "$N_RETR" <<'PY'
import sys, json, random
random.seed(0)
direct = [l for l in open(sys.argv[1])]
retr   = [l for l in open(sys.argv[2])]
n_retr = int(sys.argv[4])
random.shuffle(retr)
mix = direct + retr[:n_retr]
random.shuffle(mix)
with open(sys.argv[3], "w") as f:
    for l in mix:
        f.write(l if l.endswith("\n") else l + "\n")
print(f"mix: {len(direct)} direct + {min(n_retr,len(retr))} retrieval = {len(mix)} traces")
PY
echo "traces: $(wc -l < "$TRACES_JSONL")"

# ---- 0. free GPUs: local teacher (vllm serve, port 8200) must be down for SFT ----
pkill -9 -f "vllm serve Qwen" 2>/dev/null || true
sleep 8
# hard-free any residual GPU procs (gen-server holds no local GPU -> safe)
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    kill -9 "$pid" 2>/dev/null || true
done
sleep 8
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

# ---- 1. SFT on selective traces (stock init, 1 epoch, final ckpt only) ----
TRACES_JSONL="$TRACES_JSONL" \
TRAIN_PARQUET=/scratch/sheng/self_evolving/selective_sft_traces.parquet \
EXP="$SFT_EXP" MODEL_PATH=Qwen/Qwen3.6-27B WANDB_MODE=online \
bash scripts/self_evolving/train/run_retrieval_sft.sh 2>&1 | tee "$LOGDIR/selective_sft.log"

# ---- 2. merge the final FSDP checkpoint -> HF ----
LAST=$(ls -d "$SFT_CKPT"/global_step_* 2>/dev/null | sort -t_ -k3 -n | tail -1)
test -n "$LAST" || { echo "FATAL: no SFT checkpoint under $SFT_CKPT"; exit 1; }
echo "merging $LAST"
python3 -m verl.model_merger merge --backend fsdp \
    --local_dir "$LAST" --target_dir "$LAST/hf_merged" 2>&1 | tee "$LOGDIR/selective_merge.log"
test -f "$LAST/hf_merged/config.json" || { echo "FATAL: merge produced no config.json"; exit 1; }
# GOTCHA: model_merger writes model+config only. Copy tokenizer/chat_template from the
# checkpoint's huggingface/ subdir, else vLLM can't load the tokenizer at RL start.
for f in tokenizer.json tokenizer_config.json chat_template.jinja processor_config.json \
         vocab.json merges.txt special_tokens_map.json; do
    [ -f "$LAST/huggingface/$f" ] && cp -n "$LAST/huggingface/$f" "$LAST/hf_merged/$f"
done

# ---- 3. launch v7-selective RL (val_before_train = decisive step-0 acc) ----
ACTOR_MODEL_PATH="$LAST/hf_merged" \
EXP=healthbench_rubric_qwen36_27b_v7_selective \
RETRIEVAL_URL=http://localhost:8006/retrieve \
bash scripts/self_evolving/train/run_qwen36_27b_healthbench_rubric_v7_selective.sh 2>&1 | tee "$LOGDIR/v7_selective_train.log"
