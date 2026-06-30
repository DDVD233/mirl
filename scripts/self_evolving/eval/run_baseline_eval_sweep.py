"""Generate MIMIC-IV rare-disease test responses for off-the-shelf baseline
models (un-finetuned bases + open medical MLLMs), using the SAME eval protocol
as run_checkpoint_eval_sweep.py so the numbers are head-to-head comparable with
our trained checkpoints.

Per baseline:
  1. serve : vLLM OpenAI server on the HF repo (multimodal, TP/DP by size).
             No merge / no norm-fill — these are stock HF weights.
  2. gen   : scripts/self_evolving/eval_sota.py --provider vllm on test.jsonl,
             --vllm_thinking False (same as trained-actor rollout), full set.
             GENERATION-ONLY: --api_base "" --embed_api_base "" so we do NOT
             call any judge here (keeps load off the gcr proxy that the
             28-checkpoint rejudge is using). compute_score still extracts the
             final answer into each row's `extracted_answer`.
  3. stop  : tear down vLLM, free GPUs, next model.

Then rejudge the dumps with gpt-chat-latest via rejudge_all_models.py (identical
judge call as the trained checkpoints) -> directly comparable accuracy.

REWARD_EVAL_LENIENT_ONLY=1 is set so extract_final_answer falls back boxed ->
cue -> last-200-chars; base instruct models that don't reliably emit \\boxed{}
still get their actual final diagnosis extracted (fair extraction for everyone;
trained actors emit boxed anyway so they're unaffected).

Resumable: a baseline whose summary.json exists is skipped; eval_sota itself
resumes a partial dump by hadm_id.

Run on a 4xB200 pod (server4):
    HF_HOME=/scratch/sheng/self_evolving/hf_cache \\
    /usr/local/bin/python scripts/self_evolving/eval/run_baseline_eval_sweep.py
    # --only lingshu   to run a subset
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = "/scratch/sheng/self_evolving/verl"
PY = "/usr/local/bin/python"
VLLM = "/usr/local/bin/vllm"

# Each baseline is a dict: tag, repo, dp, tp + optional per-model overrides
# (max_num_seqs, max_model_len, extra serve flags, text_only). Big (>=27B) ->
# TP=4 (gen-bound); small -> data-parallel replicas for throughput. All are
# multimodal (incl. the 397B teacher + Kimi-K2.6, which have vision towers), so
# they get the X-ray/ECG images just like the trained checkpoints.
# llava-med is the LLaVA-1.5 codebase format; vLLM may reject it (flagged, not fatal).
BASELINES = [
    {"tag": "qwen36_27b_base", "repo": "Qwen/Qwen3.6-27B",                    "dp": 1, "tp": 4},
    {"tag": "gemma4_31b_base", "repo": "google/gemma-4-31B-it",              "dp": 1, "tp": 4},
    {"tag": "qwen35_9b_base",  "repo": "Qwen/Qwen3.5-9B",                    "dp": 2, "tp": 2},
    {"tag": "medgemma_1.5_4b", "repo": "google/medgemma-1.5-4b-it",         "dp": 4, "tp": 1},
    {"tag": "lingshu_7b",      "repo": "lingshu-medical-mllm/Lingshu-7B",   "dp": 2, "tp": 2},
    {"tag": "lingshu_32b",     "repo": "lingshu-medical-mllm/Lingshu-32B",  "dp": 1, "tp": 4},
    {"tag": "llava_med_7b",    "repo": "microsoft/llava-med-v1.5-mistral-7b", "dp": 1, "tp": 1},
    # Frontier MoE references (huge; run last). FP8 / NVFP4 weights -> TP=4,
    # smaller batch + (Kimi) shorter ctx to fit KV cache on 4xB200.
    {"tag": "qwen397_a17b_fp8", "repo": "Qwen/Qwen3.5-397B-A17B-FP8", "dp": 1, "tp": 4,
     "max_num_seqs": 64},
    {"tag": "kimi_k2.6_nvfp4",  "repo": "nvidia/Kimi-K2.6-NVFP4",     "dp": 1, "tp": 4,
     "max_num_seqs": 32, "max_model_len": 24576, "extra": ["--quantization", "modelopt_fp4"]},
]


def log(msg: str) -> None:
    print(f"[baseline {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def wait_health(port: int, proc: subprocess.Popen, timeout_s: int) -> bool:
    url = f"http://127.0.0.1:{port}/health"
    start = time.time()
    while time.time() - start < timeout_s:
        if proc.poll() is not None:
            return False  # server died (bad arch / OOM / download fail)
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(5)
    return False


def serve(spec: dict, served_name: str, port: int, logf: str, args) -> subprocess.Popen:
    dp, tp = spec["dp"], spec["tp"]
    max_model_len = spec.get("max_model_len", args.max_model_len)
    max_num_seqs = spec.get("max_num_seqs", args.max_num_seqs)
    cmd = [
        VLLM, "serve", spec["repo"],
        "--served-model-name", served_name,
        "--data-parallel-size", str(dp),
        "--tensor-parallel-size", str(tp),
        "--host", "0.0.0.0", "--port", str(port),
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", str(args.gpu_mem_util),
        "--limit-mm-per-prompt", json.dumps({"image": args.max_images}),
        "--max-num-seqs", str(max_num_seqs),
        "--trust-remote-code",
    ] + list(spec.get("extra", []))
    log(f"serving {served_name} <- {spec['repo']} (dp={dp}, tp={tp}, port={port}, "
        f"max_len={max_model_len}, max_seqs={max_num_seqs}, extra={spec.get('extra', [])})")
    lf = open(logf, "w")
    return subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, preexec_fn=os.setsid)


def _kill_gpu_apps() -> None:
    try:
        pids = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
        ).split()
        for p in pids:
            if p.strip().isdigit():
                subprocess.call(["kill", "-9", p.strip()])
    except Exception:
        pass


def wait_gpu_free(timeout: int = 180, threshold_mb: int = 2000) -> bool:
    for _ in range(max(1, timeout // 5)):
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
            used = [int(x) for x in out.split() if x.strip().isdigit()]
            if used and all(u < threshold_mb for u in used):
                return True
        except Exception:
            pass
        time.sleep(5)
    _kill_gpu_apps()
    time.sleep(6)
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
        used = [int(x) for x in out.split() if x.strip().isdigit()]
        return bool(used) and all(u < threshold_mb for u in used)
    except Exception:
        return False


def stop(proc: subprocess.Popen) -> None:
    if proc is not None and proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=30)
        except Exception:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass
    subprocess.call(["pkill", "-9", "-f", "vllm serve"])
    subprocess.call(["pkill", "-9", "-f", "EngineCore"])
    subprocess.call(["pkill", "-9", "-f", "VLLM::Worker"])
    time.sleep(2)


def run_eval(served_name: str, port: int, out_jsonl: str, summary_json: str, args) -> bool:
    # GENERATION-ONLY: empty judge/embed bases so compute_score extracts the
    # final answer but issues no external judge calls. CHAT_PROVIDER unused here.
    env = {**os.environ, "REWARD_EVAL_LENIENT_ONLY": "1"}
    cmd = [
        PY, f"{REPO}/scripts/self_evolving/eval_sota.py",
        "--provider", "vllm",
        "--val_file", args.val_file,
        "--model_name", served_name,
        "--openai_base_url", f"http://127.0.0.1:{port}/v1",
        "--openai_api_key", "EMPTY",
        "--api_base", "",            # no judge during generation
        "--embed_api_base", "",      # no embed during generation
        "--vllm_thinking", "False",
        "--vllm_max_tokens", str(args.vllm_max_tokens),
        "--concurrency", str(args.concurrency),
        "--limit", str(args.limit),
        "--output_jsonl", out_jsonl,
        "--summary_json", summary_json,
    ]
    rc = subprocess.call(cmd, env=env)
    return rc == 0 and Path(summary_json).exists()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="/scratch/sheng/self_evolving/eval_baselines")
    ap.add_argument("--val_file", default="/scratch/sheng/self_evolving/mimiciv_rare/test.jsonl")
    ap.add_argument("--port", type=int, default=8300)
    ap.add_argument("--max_model_len", type=int, default=32768)
    ap.add_argument("--gpu_mem_util", type=float, default=0.9)
    ap.add_argument("--max_images", type=int, default=4)
    ap.add_argument("--max_num_seqs", type=int, default=256)
    ap.add_argument("--vllm_max_tokens", type=int, default=8192)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--serve_timeout", type=int, default=5400,
                    help="health timeout incl. first-time HF download (default 90 min)")
    ap.add_argument("--only", default="", help="only run baselines whose tag contains this substring")
    args = ap.parse_args()

    out = Path(args.out_dir)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    (out / "out").mkdir(parents=True, exist_ok=True)

    todo = [b for b in BASELINES if not args.only or args.only in b["tag"]]
    log(f"{len(todo)} baselines to run: {[b['tag'] for b in todo]}")

    done, failed = [], []
    for spec in todo:
        tag = spec["tag"]
        summary_json = str(out / "out" / f"{tag}.summary.json")
        out_jsonl = str(out / "out" / f"{tag}.jsonl")
        if Path(summary_json).exists():
            log(f"SKIP {tag} (summary exists)")
            done.append(tag)
            continue

        ok = False
        if not wait_gpu_free():
            log(f"  GPUs not free before serving {tag} — forcing teardown")
            stop(None)
            wait_gpu_free()
        proc = serve(spec, tag, args.port, str(out / "logs" / f"serve_{tag}.log"), args)
        try:
            if not wait_health(args.port, proc, args.serve_timeout):
                log(f"  serve UNHEALTHY for {tag} ({spec['repo']}) — likely unsupported arch or "
                    f"download/OOM; see logs/serve_{tag}.log. Skipping (build a custom env later).")
            else:
                log(f"server healthy; generating for {tag}")
                ok = run_eval(tag, args.port, out_jsonl, summary_json, args)
        finally:
            stop(proc)
        (done if ok else failed).append(tag)
        log(f"finished {tag}: {'OK' if ok else 'FAILED'}")

    log(f"DONE. ok={len(done)} failed={len(failed)}")
    if failed:
        log(f"failed (need custom env / unsupported): {failed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
