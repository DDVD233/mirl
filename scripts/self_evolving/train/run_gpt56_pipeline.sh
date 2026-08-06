#!/usr/bin/env bash
# Chained: GPT-5.6-trace SFT -> merge to HF -> v11 step-0 validation.
#
# Run on the node whose 4 GPUs you are willing to give up (server1). It does NOT
# free them for you: the previous pipeline kill -9'd every PID reported by
# `nvidia-smi --query-compute-apps`, which is fine on a dedicated pod and
# destructive anywhere else. Stop the current trainer yourself, then run this.
#
# Stage 3 launches with VAL_ONLY=1 by default: the point of this pipeline is to
# find out whether the warm start helps BEFORE spending GPU-days on RL.
#
# Judge the result on `acc_len_adj_signed` (length-adjusted AND signed) — NOT
# `acc_raw`, which runs ~0.18 higher on the same responses. Bar, same judge
# (gpt-chat-latest): 0.3827 = stock + fixed retrieval, 0.3752 = no tool,
# 0.399 = v9's step 0 (best start observed). Set VAL_ONLY=0 to continue straight
# into RL if it clears.
set -xeuo pipefail

REPO=${REPO:-/scratch/sheng/self_evolving/verl_healthbench}
cd "$REPO"

TRACES_JSONL=${TRACES_JSONL:-/scratch/sheng/self_evolving/gpt56_sft_traces.jsonl}
SFT_EXP=${SFT_EXP:-gpt56_sft_qwen36_27b}
SFT_CKPT=/scratch/sheng/self_evolving/checkpoints/self_evolving_medical/$SFT_EXP
LOGDIR=/scratch/sheng/self_evolving/logs_healthbench_rubric
mkdir -p "$LOGDIR"

test -s "$TRACES_JSONL" || { echo "FATAL: no traces at $TRACES_JSONL"; exit 1; }

# Refuse to start if the GPUs are still busy — an FSDP run that lands on occupied
# GPUs OOMs halfway through and wastes the whole stage.
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; then
    echo "FATAL: GPUs still busy. Stop the running trainer first:" >&2
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv >&2
    exit 1
fi

echo "=== trace mix ==="
python3 - "$TRACES_JSONL" <<'PY'
import json, sys
from collections import Counter
kinds, scores, think = Counter(), [], []
for line in open(sys.argv[1]):
    if not line.strip():
        continue
    d = json.loads(line)
    kinds[d["extra_info"]["type"]] += 1
    scores.append(d["extra_info"].get("score", 0))
    a = d["messages"][-1]["content"]
    think.append(len(a.split("</think>")[0]))
print(f"  {sum(kinds.values())} traces: {dict(kinds)}")
print(f"  mean rubric score {sum(scores)/max(len(scores),1):.3f}")
print(f"  mean think chars  {sum(think)/max(len(think),1):,.0f}   (the previous warm "
      f"start collapsed to ~110 — anything under ~800 is a red flag)")
PY

# ---- 1. SFT ----
TRACES_JSONL="$TRACES_JSONL" EXP="$SFT_EXP" \
    bash scripts/self_evolving/train/run_gpt56_sft.sh 2>&1 | tee "$LOGDIR/gpt56_sft.log"

# ---- 2. merge the final FSDP checkpoint -> HF ----
LAST=$(ls -d "$SFT_CKPT"/global_step_* 2>/dev/null | sort -t_ -k3 -n | tail -1)
test -n "$LAST" || { echo "FATAL: no SFT checkpoint under $SFT_CKPT"; exit 1; }
echo "merging $LAST"
python3 -m verl.model_merger merge --backend fsdp \
    --local_dir "$LAST" --target_dir "$LAST/hf_merged" 2>&1 | tee "$LOGDIR/gpt56_merge.log"
test -f "$LAST/hf_merged/config.json" || { echo "FATAL: merge produced no config.json"; exit 1; }
# GOTCHA (carried over): model_merger writes model+config only. Without the tokenizer
# files copied from the checkpoint's huggingface/ subdir, vLLM cannot load the
# tokenizer at RL start and the run dies immediately.
for f in tokenizer.json tokenizer_config.json chat_template.jinja processor_config.json \
         vocab.json merges.txt special_tokens_map.json; do
    [ -f "$LAST/huggingface/$f" ] && cp -n "$LAST/huggingface/$f" "$LAST/hf_merged/$f"
done

# ---- 3. the decisive step-0 validation ----
SFT_CKPT="$LAST/hf_merged" VAL_ONLY="${VAL_ONLY:-1}" \
    bash scripts/self_evolving/train/run_v11_gpt56_sft_val.sh
