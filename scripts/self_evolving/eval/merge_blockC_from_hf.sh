#!/usr/bin/env bash
# Rebuild the block-C (every role served by the frozen 9B) checkpoints as merged HF
# weights from their raw FSDP shards on the Hub, so they can be evaluated without
# tools like the other surviving checkpoints. The NFS copies were deleted in the
# 2026-09-11 cleanup; only the shard backups remain.
#
#   bash scripts/self_evolving/eval/merge_blockC_from_hf.sh   (on an MSR pod)
set -uo pipefail
S=/scratch/sheng/self_evolving
RAW=$S/checkpoints/raw_from_hf
OUT=$S/checkpoints/hf_merged
mkdir -p "$RAW" "$OUT"
export HF_HOME=$S/hf_cache HF_HUB_ENABLE_HF_TRANSFER=0
declare -A TARGET=(
  [hb9b_specgap_ship_retrieval_self9b_websearch_global_step_200]=hb9bC_ser_step200
  [hb9b_specgap_simple_retrieval_self9b_websearch_global_step_80]=hb9bC_fixed_step80
)
for repo in "${!TARGET[@]}"; do
    tgt=$OUT/${TARGET[$repo]}
    if ls "$tgt"/model*.safetensors >/dev/null 2>&1; then echo "=== $tgt already merged"; continue; fi
    echo "=== $(date -u +%FT%TZ) download ddvd233/$repo"
    for i in 1 2 3; do
        python3 -c "from huggingface_hub import snapshot_download; snapshot_download('ddvd233/$repo', local_dir='$RAW/$repo', max_workers=8)" && break
        sleep 30
    done
    ls "$RAW/$repo/actor" | head -3
    echo "=== $(date -u +%FT%TZ) merge -> $tgt"
    ( cd "$S/verl_specgap" && python3 -m verl.model_merger merge --backend fsdp \
        --local_dir "$RAW/$repo/actor" --target_dir "$tgt" ) 2>&1 | tail -5
    if ls "$tgt"/model*.safetensors >/dev/null 2>&1; then
        echo "=== merged OK: $(du -sh "$tgt" | cut -f1)"; rm -rf "$RAW/$repo"
    else
        echo "=== MERGE FAILED for $repo (raw kept at $RAW/$repo)"
    fi
done
echo "BLOCKC_MERGE_DONE $(date -u +%FT%TZ)"
