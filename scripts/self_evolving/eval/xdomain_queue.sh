#!/usr/bin/env bash
# Zero-shot cross-domain evaluation (dvd 2026-09-14): the checkpoints trained on the
# HealthBench-Professional description, scored on the non-medical held-out benchmarks
# PRBench Hard and ProfBench exactly as the held-out arms are validated -- that
# benchmark's domain bundle, the web-search tool only (no clinical corpus), no length
# term, gpt-chat-latest grader with three votes -- next to their untrained base models.
# Same lock/skip protocol as onoff_queue.sh; dumps land in
# $S/logs_hb9b/val_generations/<exp>/0.jsonl and the score is the val-core line of
# $S/paper_refresh/xdom_logs/<exp>.log (collect with xdomain_collect.py).
#
#   NGPU=2 bash scripts/self_evolving/eval/xdomain_queue.sh     (pod 2333)
set -uo pipefail
S=/scratch/sheng/self_evolving
CK=$S/checkpoints/hf_merged
W=$S/verl_specgap/scripts/self_evolving/eval/run_hbpro_valonly.sh
LOG=$S/paper_refresh/xdom_logs; mkdir -p "$LOG"
NGPU="${NGPU:-2}"
for a in 19 20; do [ -f "$S/kb/search_cache_xdom_arm$a.sqlite" ] || cp "$S/kb/search_cache_arm$a.sqlite" "$S/kb/search_cache_xdom_arm$a.sqlite"; done

run() {  # run <exp> <model> <expected rows> [ENV=VAL ...]
    local exp=$1 model=$2 want=$3; shift 3
    local dump=$S/logs_hb9b/val_generations/$exp/0.jsonl
    if [ -f "$dump" ] && [ "$(wc -l < "$dump")" -ge "$want" ]; then echo "=== skip $exp (dump complete)"; return 0; fi
    if ! mkdir "$LOG/$exp.lock" 2>/dev/null; then echo "=== skip $exp (claimed by $(cat "$LOG/$exp.lock/owner" 2>/dev/null))"; return 0; fi
    echo "$(hostname) $(date -u +%FT%TZ)" > "$LOG/$exp.lock/owner"
    echo "=== $(date -u +%FT%TZ) START $exp on $(hostname)"
    env "$@" MODEL="$model" EXP="$exp" RETRIEVAL=0 WEB_SEARCH_TOOL=1 WEB_ONLY=1 N_GPUS="$NGPU" ROLLOUT_TP=2 \
        HB_VAL_LENGTH_PENALTY_PER_500=0 HB_LENGTH_CENTER=4000 HB_KB_ANCHOR_SHARE=0 HB_STYLE_SEED_SHARE=0 \
        bash "$W" > "$LOG/$exp.log" 2>&1
    local rc=$?
    for pat in '^VLLM::EngineCore' '^ray::' '^/usr/local/bin/python scripts/self_evolving/generation_server.py' \
               '^/usr/local/bin/python scripts/self_evolving/kb/serper_cache_server.py'; do pkill -f "$pat" 2>/dev/null || true; done
    sleep 15
    if [ -f "$dump" ]; then echo "=== $(date -u +%FT%TZ) DONE $exp rc=$rc rows=$(wc -l < "$dump")"
    else echo "=== $(date -u +%FT%TZ) FAILED $exp rc=$rc (no dump); tail:"; tail -5 "$LOG/$exp.log"; fi
}

PRB=(SE_DOMAIN=prbench VAL_PARQUET=$S/prbench_hard_val.parquet MAX_PROMPT_LEN=16384 ROLLOUT_MAX_LEN=24576 VERL_MAX_SEARCHES=4
     SEARCH_SNAPSHOT=$S/kb/search_cache_xdom_arm19.sqlite)
PRF=(SE_DOMAIN=profbench VAL_PARQUET=$S/profbench_val.parquet SEARCH_SNAPSHOT=$S/kb/search_cache_xdom_arm20.sqlite)

# ProfBench first (40 items, minutes each), then PRBench (550 items).
for pair in Qwen/Qwen3.6-27B:27b_base $CK/hb27b_fixed_step60:27b_fixed60 $CK/hb27b_ser_step200:27b_ser200 \
            Qwen/Qwen3.5-9B:9b_base $CK/hb9b_noretr_fixed_step860:9bD_fixed860 $CK/hb9b_noretr_ser_step480:9bD_ser480 \
            $CK/hb9bC_fixed_step80:9bC_fixed80 $CK/hb9bC_ser_step200:9bC_ser200; do
    model=${pair%%:*}; tag=${pair##*:}
    run "xdom_profbench_$tag" "$model" 40 "${PRF[@]}"
done
for pair in Qwen/Qwen3.6-27B:27b_base $CK/hb27b_fixed_step60:27b_fixed60 $CK/hb27b_ser_step200:27b_ser200 \
            Qwen/Qwen3.5-9B:9b_base $CK/hb9b_noretr_fixed_step860:9bD_fixed860 $CK/hb9b_noretr_ser_step480:9bD_ser480 \
            $CK/hb9bC_fixed_step80:9bC_fixed80 $CK/hb9bC_ser_step200:9bC_ser200; do
    model=${pair%%:*}; tag=${pair##*:}
    run "xdom_prbench_$tag" "$model" 550 "${PRB[@]}"
done
echo "XDOM_QUEUE_DONE on $(hostname) $(date -u +%FT%TZ)"
