#!/usr/bin/env bash
# Stand up the generation server exactly as ARM=23 does (general bundle, no style
# seeds, curated-KB anchors, staged image tasks), let it mint a small pool, print
# the accepted tasks so a person can read them, then stop it. Needs TRAPI, the mib
# KB endpoints and the NFS; no GPU. Run on an MSR pod from the repo root:
#   bash scripts/self_evolving/tests/general_domain_gen_smoke.sh
set -uo pipefail
S=/scratch/sheng/self_evolving
PORT="${PORT:-8077}"
WANT="${WANT:-8}"
OUT="${OUT:-$S/paper_refresh/general_gen_smoke}"
rm -rf "$OUT"; mkdir -p "$OUT/prompts"
KEY=$(cat "$S/.trapi_key")
cd "$S/verl_specgap"
env SE_DOMAIN=general CHAT_PROVIDER=trapi \
    HB_STYLE_SEED_SHARE=0 HB_KB_ANCHOR_SHARE=0.40 HB_NONENGLISH_SHARE=0.10 \
    HB_MM_SHARE=0.35 HB_MM_MANIFEST="$S/mm_media/manifest.jsonl" HB_MM_ROOT="$S/mm_media/images" \
    HF_HOME="$S/hf_cache" \
    /usr/local/bin/python scripts/self_evolving/generation_server.py \
    --rubric_mode --prompt_dir "$OUT/prompts" \
    --coverage_prompt_file "$OUT/prompts/coverage_prompt.json" \
    --api_base http://point.dd.works:18890/v1 --api_key "$KEY" --model_name gpt-chat-latest_2026-05-28 \
    --embed_api_base http://mib.media.mit.edu:18001/v1 --embed_model Qwen/Qwen3-VL-Embedding-2B \
    --milvus_uri http://mib.media.mit.edu:19531 --milvus_token root:Milvus \
    --milvus_collection medical_knowledge_v2 --milvus_top_k 8 \
    --retrieve_top_k 5 --retrieve_total 16 \
    --n_queries 6 --questions_per_query 1 \
    --accuracy_window 64 --max_pool_size 40 \
    --workers 4 --log_dir "$OUT" \
    --host 0.0.0.0 --port "$PORT" > "$OUT/gen_server.log" 2>&1 &
PID=$!
start=$SECONDS
until curl -sf -m 5 "localhost:$PORT/healthz" >/dev/null; do
    kill -0 $PID 2>/dev/null || { echo "gen server died"; tail -30 "$OUT/gen_server.log"; exit 1; }
    (( SECONDS - start > 600 )) && { echo "gen server never healthy"; kill $PID; exit 1; }
    sleep 5
done
echo "healthy after $((SECONDS-start))s; waiting for $WANT accepted tasks"
while :; do
    n=$(cat "$OUT"/server_accepted_*.jsonl 2>/dev/null | wc -l)
    [ "${n:-0}" -ge "$WANT" ] && break
    (( SECONDS - start > 1500 )) && { echo "only $n tasks after 25 min"; break; }
    sleep 15
done
echo "=== stats"; curl -s -m 10 "localhost:$PORT/stats" | head -c 1500; echo
kill $PID 2>/dev/null; sleep 2
echo "=== accepted tasks: $(cat "$OUT"/server_accepted_*.jsonl 2>/dev/null | wc -l)"
/usr/local/bin/python - "$OUT" <<'EOF'
import glob, json, sys, collections
out = sys.argv[1]
rows = [json.loads(l) for f in glob.glob(f"{out}/server_accepted_*.jsonl") for l in open(f) if l.strip()]
c = collections.Counter()
for r in rows:
    e = r.get("entry", r)
    ei = e.get("extra_info", {})
    c[(ei.get("use_case"), bool(e.get("images") or ei.get("images")))] += 1
print("use_case x has_image:", dict(c))
for r in rows[:4]:
    e = r.get("entry", r); ei = e.get("extra_info", {})
    print("\n---", ei.get("use_case"), "|", ei.get("specialty"), "| images:", e.get("images") or ei.get("images"))
    for m in e.get("prompt", []):
        print(f"  [{m.get('role')}] {str(m.get('content'))[:700]}")
    for it in (ei.get("rubric_items") or [])[:6]:
        print(f"  [{it.get('points')}] {it.get('criterion_text', it.get('criterion'))}")
EOF
