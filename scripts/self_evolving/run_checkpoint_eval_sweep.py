"""Evaluate every retained verl checkpoint of the MIMIC-IV rare-disease runs and
record per-ICD-category metrics, judged by a TRAPI model (default gpt-5.3-chat).

For each experiment under CKPT_ROOT we evaluate every `global_step_*/actor`
that still has FSDP weights (verl prunes all but the last few). Per checkpoint:

  1. merge  : FSDP shards -> HF safetensors via `verl.model_merger`
  2. serve  : vLLM OpenAI server on the merged dir (multimodal, TP by size)
  3. eval   : scripts/self_evolving/eval_sota.py --provider vllm on test.jsonl,
              scored by verl's compute_score (same pipeline as training val),
              with the judge = TRAPI gpt-5.3-chat via the AAD-refreshing proxy.
              eval_sota writes <out>.summary.json with overall + per-category
              (per data_source) metrics.
  4. cleanup: stop vLLM, delete the merged dir (unless --keep-merged).

Resumable: a checkpoint whose summary.json already exists is skipped. The
per-sample eval JSONL is itself resumable (eval_sota skips done hadm_ids), so
an interrupted eval continues where it stopped.

The judge endpoint is the trapi_proxy (http://point.dd.works:18890/v1) which
injects fresh Entra/AAD tokens; we send CHAT_PROVIDER=trapi so the judge path
uses max_completion_tokens + reasoning_effort=none (no temperature, which
gpt-5.3-chat rejects).

Run on the GPU pod (server 1):
    /usr/local/bin/python scripts/self_evolving/run_checkpoint_eval_sweep.py \
        --only e4b_rl_step400      # one checkpoint (validation)
    /usr/local/bin/python scripts/self_evolving/run_checkpoint_eval_sweep.py
        # full sweep over all retained checkpoints
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = "/scratch/sheng/self_evolving/verl"
DEFAULT_CKPT_ROOT = f"{REPO}/checkpoints/self_evolving_medical"
DEFAULT_OUT = "/scratch/sheng/self_evolving/eval_gpt53"
PY = "/usr/local/bin/python"
VLLM = "/usr/local/bin/vllm"

# Base HF models (in the local cache) per experiment-name token. verl's FSDP
# save drops some pretrained 1-D norm weights (e.g. Gemma3n/E4B's per-layer
# k_norm on the KV-shared upper layers); vLLM strict-loads and refuses the
# merged checkpoint. We backfill ONLY those missing 1-D tensors from the base
# model (they're pretrained norms RL barely moves, and this matches what the
# training-time vLLM rollout used). 2-D projections that are architecturally
# shared/absent are intentionally left out.
HF_CACHE = "/scratch/sheng/self_evolving/hf_cache/hub"
BASE_MODELS = [
    ("gemma4_31b", "models--google--gemma-4-31B-it"),
    ("gemma4_e4b", "models--google--gemma-4-E4B-it"),
    ("qwen35_9b", "models--Qwen--Qwen3.5-9B"),
    ("qwen36_27b", "models--Qwen--Qwen3.6-27B"),
]


def resolve_base_safetensors(exp: str) -> list[str]:
    n = exp.lower()
    repo = next((r for tok, r in BASE_MODELS if tok in n), None)
    if repo is None:
        return []
    snaps = sorted(glob.glob(f"{HF_CACHE}/{repo}/snapshots/*"))
    if not snaps:
        return []
    return sorted(glob.glob(f"{snaps[-1]}/*.safetensors"))


def fill_missing_norms(merged_dir: str, base_files: list[str]) -> None:
    """Backfill 1-D tensors present in the base model but missing from the merged
    checkpoint (handles both single-file and sharded safetensors layouts)."""
    if not base_files:
        log("  norm-fill: no base model resolved, skipping")
        return
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    merged_st = sorted(glob.glob(f"{merged_dir}/*.safetensors"))
    merged_keys: set = set()
    for f in merged_st:
        with safe_open(f, framework="pt") as h:
            merged_keys |= set(h.keys())
    to_add: dict = {}
    for bf in base_files:
        with safe_open(bf, framework="pt") as h:
            for k in h.keys():
                if k in merged_keys or k in to_add:
                    continue
                t = h.get_tensor(k)
                if t.dim() == 1:  # norms / biases only — never inject big matrices
                    to_add[k] = t.to(torch.bfloat16)
    if not to_add:
        log("  norm-fill: nothing missing")
        return
    log(f"  norm-fill: adding {len(to_add)} missing 1-D tensors from base")
    idx_path = f"{merged_dir}/model.safetensors.index.json"
    if os.path.exists(idx_path):
        fill_name = "model-fill.safetensors"
        save_file(to_add, f"{merged_dir}/{fill_name}", metadata={"format": "pt"})
        idx = json.load(open(idx_path))
        wm = idx.setdefault("weight_map", {})
        for k in to_add:
            wm[k] = fill_name
        extra = sum(t.numel() * t.element_size() for t in to_add.values())
        idx.setdefault("metadata", {})
        idx["metadata"]["total_size"] = idx["metadata"].get("total_size", 0) + extra
        json.dump(idx, open(idx_path, "w"))
    else:
        single = f"{merged_dir}/model.safetensors"
        sd = load_file(single)
        sd.update(to_add)
        save_file(sd, single, metadata={"format": "pt"})


def log(msg: str) -> None:
    print(f"[sweep {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parallel_for(name: str) -> tuple[int, int]:
    """Return (data_parallel_size, tensor_parallel_size), always using all 4
    B200s. Big models (27B/31B) are generation-bound, so TP=4 is fastest.
    Small models (e4b/9B) are ~2.5x slower at TP=4 than TP=2 (comm overhead),
    so run 2 data-parallel TP=2 replicas behind one load-balanced endpoint —
    all 4 GPUs AND ~2x the throughput of a single TP=2 replica."""
    n = name.lower()
    if "27b" in n or "31b" in n or "32b" in n:
        return (1, 4)
    return (2, 2)


def tag_for(exp: str, step: int) -> str:
    """Compact, filesystem-safe label for a checkpoint."""
    short = re.sub(r"_20\d{6}_\d{6}$", "", exp)  # strip trailing timestamp
    short = short.replace("mimiciv_rare_", "")
    return f"{short}__step{step}"


def discover(ckpt_root: str) -> list[dict]:
    """All (experiment, global_step, actor_dir) with intact FSDP actor weights."""
    out = []
    root = Path(ckpt_root)
    for exp_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for gs in sorted(exp_dir.glob("global_step_*")):
            actor = gs / "actor"
            # an evaluable actor has the sharded model weights
            if actor.is_dir() and list(actor.glob("model_world_size_*_rank_0.pt")):
                m = re.search(r"global_step_(\d+)", gs.name)
                step = int(m.group(1)) if m else -1
                out.append({"exp": exp_dir.name, "step": step, "actor": str(actor)})
    return out


def wait_health(port: int, proc: subprocess.Popen, timeout_s: int) -> bool:
    url = f"http://127.0.0.1:{port}/health"
    start = time.time()
    while time.time() - start < timeout_s:
        if proc.poll() is not None:
            return False  # server died
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(5)
    return False


def merge(actor: str, merged: str, logf: str) -> bool:
    if Path(merged).exists() and list(Path(merged).glob("*.safetensors")):
        log(f"merged dir already present, skipping merge: {merged}")
        return True
    shutil.rmtree(merged, ignore_errors=True)
    log(f"merging {actor} -> {merged}")
    env = {**os.environ, "PYTHONPATH": REPO}
    with open(logf, "w") as lf:
        rc = subprocess.call(
            [PY, "-m", "verl.model_merger", "merge", "--backend", "fsdp",
             "--local_dir", actor, "--target_dir", merged],
            cwd=REPO, env=env, stdout=lf, stderr=subprocess.STDOUT,
        )
    ok = rc == 0 and bool(list(Path(merged).glob("*.safetensors")))
    if not ok:
        log(f"MERGE FAILED (rc={rc}) — see {logf}")
    return ok


def serve(merged: str, served_name: str, port: int, dp: int, tp: int, logf: str, args) -> subprocess.Popen:
    cmd = [
        VLLM, "serve", merged,
        "--served-model-name", served_name,
        "--data-parallel-size", str(dp),
        "--tensor-parallel-size", str(tp),
        "--host", "0.0.0.0", "--port", str(port),
        "--max-model-len", str(args.max_model_len),
        "--gpu-memory-utilization", str(args.gpu_mem_util),
        "--limit-mm-per-prompt", json.dumps({"image": args.max_images}),
        "--max-num-seqs", str(args.max_num_seqs),
        "--trust-remote-code",
    ]
    log(f"serving {served_name} (dp={dp}, tp={tp}, port={port})")
    lf = open(logf, "w")
    return subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, preexec_fn=os.setsid)


def _kill_gpu_apps() -> None:
    """Hard-kill every process still holding GPU memory (last resort for
    orphaned VLLM::Worker_TP* processes whose launcher died without reaping
    them)."""
    try:
        pids = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
        ).split()
        for p in pids:
            if p.strip().isdigit():
                subprocess.call(["kill", "-9", p.strip()])
    except Exception:
        pass


def wait_gpu_free(timeout: int = 150, threshold_mb: int = 2000) -> bool:
    """Poll until every GPU is below threshold_mb used. vLLM TP teardown lags
    a few seconds after the process exits; serving the next model before the
    prior workers release causes NCCL/TCPStore crashes. If the GPUs never free
    (orphaned workers), hard-kill the GPU-holding PIDs and re-check."""
    for _ in range(max(1, timeout // 5)):
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                text=True,
            )
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
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True
        )
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
    # belt-and-suspenders: vLLM TP worker (EngineCore) subprocesses can outlive
    # a SIGTERM to the launcher and keep holding GPUs/NCCL -> kill stragglers so
    # the next serve starts from a clean slate.
    subprocess.call(["pkill", "-9", "-f", "vllm serve"])
    subprocess.call(["pkill", "-9", "-f", "EngineCore"])
    subprocess.call(["pkill", "-9", "-f", "VLLM::Worker"])  # TP worker subprocesses
    time.sleep(2)


def run_eval(served_name: str, port: int, out_jsonl: str, summary_json: str, args) -> bool:
    env = {**os.environ, "CHAT_PROVIDER": "trapi"}
    cmd = [
        PY, f"{REPO}/scripts/self_evolving/eval_sota.py",
        "--provider", "vllm",
        "--val_file", args.val_file,
        "--model_name", served_name,
        "--openai_base_url", f"http://127.0.0.1:{port}/v1",
        "--openai_api_key", "EMPTY",
        "--judge_model_name", args.judge_model,
        "--api_base", args.judge_base,
        "--judge_api_key", args.judge_key,
        "--embed_api_base", args.embed_base,
        "--embed_model", args.embed_model,
        "--vllm_thinking", str(args.vllm_thinking),
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
    ap.add_argument("--ckpt_root", default=DEFAULT_CKPT_ROOT)
    ap.add_argument("--out_dir", default=DEFAULT_OUT)
    ap.add_argument("--val_file", default="/scratch/sheng/self_evolving/mimiciv_rare/test.jsonl")
    ap.add_argument("--judge_model", default="gpt-5.3-chat_2026-03-03")
    ap.add_argument("--judge_base", default="http://point.dd.works:18890/v1")
    ap.add_argument("--judge_key", default=os.environ.get("TRAPI_KEY", ""))
    ap.add_argument("--judge_key_file", default="/scratch/sheng/self_evolving/.trapi_key")
    ap.add_argument("--embed_base", default="http://mib.media.mit.edu:18001/v1")
    ap.add_argument("--embed_model", default="Qwen/Qwen3-VL-Embedding-2B")
    ap.add_argument("--port", type=int, default=8200)
    # vLLM caps each request at --max-model-len (NOT the model's 256K ceiling).
    # Must fit the longest prompt (<=6144 text + vision tokens) PLUS
    # vllm_max_tokens output. 32768 = 2x headroom over the ~14K real max; going
    # to the full 256K is unnecessary and would waste KV cache / cut batch size.
    ap.add_argument("--max_model_len", type=int, default=32768)
    ap.add_argument("--gpu_mem_util", type=float, default=0.9)
    ap.add_argument("--max_images", type=int, default=4)
    ap.add_argument("--max_num_seqs", type=int, default=256)
    # enable_thinking=True makes these actors ramble in a thinking channel and
    # never emit the required \boxed{} (format_ok ~0.12, acc ~0.10); with it
    # OFF they give brief reasoning + the boxed answer (format_ok ~1.0, acc
    # ~0.5), matching how they were trained/rolled out. Keep OFF.
    ap.add_argument("--vllm_thinking", default="False")
    ap.add_argument("--vllm_max_tokens", type=int, default=8192)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="0 = full test set")
    ap.add_argument("--serve_timeout", type=int, default=1200)
    ap.add_argument("--serve_retries", type=int, default=3,
                    help="serve+eval attempts per checkpoint (attempt 1 = configured DP/TP; "
                         "later attempts fall back to a single plain-TP replica)")
    ap.add_argument("--only", default="", help="only run checkpoints whose tag contains this substring")
    ap.add_argument("--keep_merged", action="store_true")
    ap.add_argument("--tp", type=int, default=0, help="override TP (0 = auto by model size)")
    args = ap.parse_args()

    if not args.judge_key and Path(args.judge_key_file).exists():
        args.judge_key = Path(args.judge_key_file).read_text().strip()

    out = Path(args.out_dir)
    (out / "merged").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    (out / "out").mkdir(parents=True, exist_ok=True)

    ckpts = discover(args.ckpt_root)
    if args.only:
        ckpts = [c for c in ckpts if args.only in tag_for(c["exp"], c["step"])]
    log(f"discovered {len(ckpts)} evaluable checkpoints"
        f"{' (filtered)' if args.only else ''}")
    for c in ckpts:
        log(f"  {tag_for(c['exp'], c['step'])}")

    done, failed = [], []
    for c in ckpts:
        tag = tag_for(c["exp"], c["step"])
        summary_json = str(out / "out" / f"{tag}.summary.json")
        out_jsonl = str(out / "out" / f"{tag}.jsonl")
        if Path(summary_json).exists():
            log(f"SKIP {tag} (summary exists)")
            done.append(tag)
            continue

        merged = str(out / "merged" / tag)
        if not merge(c["actor"], merged, str(out / "logs" / f"merge_{tag}.log")):
            failed.append(tag)
            continue
        try:
            fill_missing_norms(merged, resolve_base_safetensors(c["exp"]))
        except Exception as e:
            log(f"  norm-fill failed (continuing, vLLM may reject): {type(e).__name__}: {e}")

        if args.tp:
            dp0, tp0 = 1, args.tp
        else:
            dp0, tp0 = parallel_for(c["exp"])
        # Serve+eval with retries: consecutive serves can hit transient
        # NCCL/TCPStore crashes if the prior workers haven't released. Each
        # attempt waits for the GPUs to drain, then re-serves; run_eval resumes
        # from out_jsonl so a mid-eval crash continues where it stopped. If a
        # data-parallel serve fails to come up, later attempts fall back to a
        # single plain TP replica (slower, 2 idle GPUs, but reliable).
        ok = False
        for attempt in range(1, args.serve_retries + 1):
            dp, tp = (dp0, tp0) if attempt == 1 else (1, tp0)
            if not wait_gpu_free():
                log(f"  GPUs not free before serving {tag} (attempt {attempt}) — forcing teardown")
                stop(None)
                wait_gpu_free()
            proc = serve(merged, tag, args.port, dp, tp, str(out / "logs" / f"serve_{tag}.log"), args)
            try:
                if not wait_health(args.port, proc, args.serve_timeout):
                    log(f"  serve unhealthy for {tag} (attempt {attempt}/{args.serve_retries}, dp={dp} tp={tp})")
                    continue
                log(f"server healthy; running eval for {tag} (attempt {attempt}, dp={dp} tp={tp})")
                ok = run_eval(tag, args.port, out_jsonl, summary_json, args)
            finally:
                stop(proc)
            if ok:
                break
            log(f"  eval did not complete for {tag} (attempt {attempt}/{args.serve_retries})")
        (done if ok else failed).append(tag)
        if not args.keep_merged:
            shutil.rmtree(merged, ignore_errors=True)
        log(f"finished {tag}: {'OK' if ok else 'FAILED'}")

    log(f"DONE. ok={len(done)} failed={len(failed)}")
    if failed:
        log(f"failed: {failed}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
