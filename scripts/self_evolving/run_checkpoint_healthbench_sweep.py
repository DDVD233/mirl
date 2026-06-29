"""Evaluate every retained verl checkpoint on HealthBench Professional.

Sibling of run_checkpoint_eval_sweep.py (which evaluates on the MIMIC-IV rare
task). Reuses that driver's merge + norm-fill + resume machinery, but:
  - eval  : scripts/self_evolving/eval/healthbench_professional_eval.py, graded
            by TRAPI gpt-5.3-chat (reasoning_effort=none), full 525 examples,
            logged to W&B (run name hbpro-ckpt-<tag>).
  - serve : adds --mm-encoder-attn-backend TORCH_SDPA. HealthBench Professional
            is TEXT-ONLY, so the vision tower is never invoked — SDPA-vs-cute on
            the ViT has ZERO effect on the numbers; we use SDPA purely to dodge
            the B200 cutlass-cute crash for both gemma and Qwen checkpoints.
  - GPUs  : pinned to CUDA_VISIBLE_DEVICES (default 2,3) and TP=2, and ALL
            cleanup (stop / wait_gpu_free) is scoped to ONLY those GPUs, so this
            can run concurrently with another job on the node's other GPUs
            (e.g. the base-model re-run on GPUs 0,1) without killing it.

Resumable: a checkpoint whose <tag>.summary.json exists is skipped.

Run on server4 (2337), GPUs 2,3:
    /usr/local/bin/python scripts/self_evolving/run_checkpoint_healthbench_sweep.py
    /usr/local/bin/python scripts/self_evolving/run_checkpoint_healthbench_sweep.py --only selfimprove
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
DEFAULT_OUT = "/scratch/sheng/self_evolving/eval_healthbench"
HF_HOME = "/scratch/sheng/self_evolving/hf_cache"
HBEVAL = "/scratch/sheng/self_evolving/healthbench_eval/healthbench_professional_eval.py"
PY = "/usr/local/bin/python"
VLLM = "/usr/local/bin/vllm"
HF_CACHE = f"{HF_HOME}/hub"

# Base HF models per experiment-name token (for norm backfill — see the MIMIC
# sweep's comment; verl's FSDP save drops some pretrained 1-D norms vLLM needs).
BASE_MODELS = [
    ("gemma4_31b", "models--google--gemma-4-31B-it"),
    ("gemma4_e4b", "models--google--gemma-4-E4B-it"),
    ("qwen35_9b", "models--Qwen--Qwen3.5-9B"),
    ("qwen36_27b", "models--Qwen--Qwen3.6-27B"),
]


def log(msg: str) -> None:
    print(f"[hbsweep {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def resolve_base_safetensors(exp: str) -> list[str]:
    n = exp.lower()
    repo = next((r for tok, r in BASE_MODELS if tok in n), None)
    if repo is None:
        return []
    snaps = sorted(glob.glob(f"{HF_CACHE}/{repo}/snapshots/*"))
    return sorted(glob.glob(f"{snaps[-1]}/*.safetensors")) if snaps else []


def fill_missing_norms(merged_dir: str, base_files: list[str]) -> None:
    """Backfill 1-D tensors present in the base model but missing from the merged
    checkpoint (verl FSDP drops some pretrained norms; vLLM strict-loads)."""
    if not base_files:
        log("  norm-fill: no base model resolved, skipping")
        return
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    merged_keys: set = set()
    for f in sorted(glob.glob(f"{merged_dir}/*.safetensors")):
        with safe_open(f, framework="pt") as h:
            merged_keys |= set(h.keys())
    to_add: dict = {}
    for bf in base_files:
        with safe_open(bf, framework="pt") as h:
            for k in h.keys():
                if k in merged_keys or k in to_add:
                    continue
                t = h.get_tensor(k)
                if t.dim() == 1:
                    to_add[k] = t.to(torch.bfloat16)
    if not to_add:
        log("  norm-fill: nothing missing")
        return
    log(f"  norm-fill: adding {len(to_add)} missing 1-D tensors from base")
    idx_path = f"{merged_dir}/model.safetensors.index.json"
    if os.path.exists(idx_path):
        save_file(to_add, f"{merged_dir}/model-fill.safetensors", metadata={"format": "pt"})
        idx = json.load(open(idx_path))
        wm = idx.setdefault("weight_map", {})
        for k in to_add:
            wm[k] = "model-fill.safetensors"
        extra = sum(t.numel() * t.element_size() for t in to_add.values())
        idx.setdefault("metadata", {})
        idx["metadata"]["total_size"] = idx["metadata"].get("total_size", 0) + extra
        json.dump(idx, open(idx_path, "w"))
    else:
        single = f"{merged_dir}/model.safetensors"
        sd = load_file(single)
        sd.update(to_add)
        save_file(sd, single, metadata={"format": "pt"})


def tag_for(exp: str, step: int) -> str:
    short = re.sub(r"_20\d{6}_\d{6}$", "", exp).replace("mimiciv_rare_", "")
    return f"{short}__step{step}"


def discover(ckpt_root: str) -> list[dict]:
    out = []
    for exp_dir in sorted(p for p in Path(ckpt_root).iterdir() if p.is_dir()):
        for gs in sorted(exp_dir.glob("global_step_*"), key=lambda p: int(re.search(r"\d+", p.name).group())):
            actor = gs / "actor"
            if actor.is_dir() and list(actor.glob("model_world_size_*_rank_0.pt")):
                step = int(re.search(r"global_step_(\d+)", gs.name).group(1))
                out.append({"exp": exp_dir.name, "step": step, "actor": str(actor)})
    return out


def gpu_indices(args) -> list[int]:
    return [int(x) for x in args.gpus.split(",") if x.strip().isdigit()]


def gpu_used_mb(idx_list: list[int]) -> list[int]:
    out = []
    for i in idx_list:
        try:
            v = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", str(i)],
                text=True).strip()
            out.append(int(v))
        except Exception:
            out.append(999999)
    return out


def wait_gpu_free(args, timeout: int = 150, threshold_mb: int = 3000) -> bool:
    """Wait until OUR GPUs (args.gpus) are free — never looks at the other GPUs,
    so a co-tenant job on them doesn't block us."""
    idx = gpu_indices(args)
    for _ in range(max(1, timeout // 5)):
        if all(u < threshold_mb for u in gpu_used_mb(idx)):
            return True
        time.sleep(5)
    return False


def kill_our_gpu_procs(args) -> None:
    """Kill compute procs holding ONLY our GPUs (scoped straggler cleanup — does
    NOT pkill all vLLM, so a co-tenant server on other GPUs survives)."""
    for i in gpu_indices(args):
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader", "-i", str(i)],
                text=True)
            for pid in [p.strip() for p in out.split() if p.strip().isdigit()]:
                subprocess.call(["kill", "-9", pid])
        except Exception:
            pass


def stop(proc, args) -> None:
    if proc is not None and proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=30)
        except Exception:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass
    kill_our_gpu_procs(args)
    time.sleep(2)


def merge(actor: str, merged: str, logf: str) -> bool:
    if Path(merged).exists() and list(Path(merged).glob("*.safetensors")):
        log(f"merged dir already present, skipping merge: {merged}")
        return True
    shutil.rmtree(merged, ignore_errors=True)
    log(f"merging {actor} -> {merged}")
    with open(logf, "w") as lf:
        rc = subprocess.call(
            [PY, "-m", "verl.model_merger", "merge", "--backend", "fsdp",
             "--local_dir", actor, "--target_dir", merged],
            cwd=REPO, env={**os.environ, "PYTHONPATH": REPO}, stdout=lf, stderr=subprocess.STDOUT)
    ok = rc == 0 and bool(list(Path(merged).glob("*.safetensors")))
    if not ok:
        log(f"MERGE FAILED (rc={rc}) — see {logf}")
    return ok


def serve(merged: str, served_name: str, logf: str, args):
    cmd = [
        VLLM, "serve", merged, "--served-model-name", served_name,
        "--tensor-parallel-size", str(args.tp), "--host", "0.0.0.0", "--port", str(args.port),
        "--max-model-len", str(args.max_model_len), "--gpu-memory-utilization", str(args.gpu_mem_util),
        "--mm-encoder-attn-backend", "TORCH_SDPA",  # inert for text-only HealthBench; dodges B200 cute crash
        "--max-num-seqs", str(args.max_num_seqs), "--trust-remote-code",
    ]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": args.gpus, "HF_HOME": HF_HOME, "VLLM_LOGGING_LEVEL": "WARNING"}
    log(f"serving {served_name} (tp={args.tp}, gpus={args.gpus}, port={args.port})")
    lf = open(logf, "w")
    return subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, preexec_fn=os.setsid, env=env)


def wait_health(port: int, proc, timeout_s: int) -> bool:
    start = time.time()
    while time.time() - start < timeout_s:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(5)
    return False


def run_eval(served_name: str, tag: str, args) -> bool:
    think = ["--disable-thinking"] if "qwen" in tag.lower() else []  # qwen3 thinking off; gemma has none
    eval_out = f"{args.out_dir}/files/{tag}"
    os.makedirs(eval_out, exist_ok=True)
    cmd = [
        PY, HBEVAL,
        "--model-base", f"http://127.0.0.1:{args.port}/v1", "--model-name", served_name,
        "--model-provider", "vllm", *think,
        "--grader", "trapi", "--grader-base", args.judge_base, "--grader-key", args.judge_key,
        # gpt-chat-latest rejects reasoning_effort="none" (only "medium"); omit effort for it.
        "--grader-model", args.judge_model,
        "--grader-effort", ("" if "chat-latest" in args.judge_model else "none"),
        "--concurrency", str(args.concurrency), "--max-tokens", "2048", "--limit", str(args.limit),
        "--wandb-project", args.wandb_project, "--wandb-run-name", f"hbpro-ckpt-{tag}",
        "--output-dir", eval_out,
    ]
    rc = subprocess.call(cmd, env={**os.environ, "HF_HOME": HF_HOME})
    metrics = sorted(glob.glob(f"{eval_out}/metrics_*.json"))
    if rc == 0 and metrics:
        shutil.copy(metrics[-1], f"{args.out_dir}/out/{tag}.summary.json")
        return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_root", default=DEFAULT_CKPT_ROOT)
    ap.add_argument("--out_dir", default=DEFAULT_OUT)
    ap.add_argument("--judge_model", default="gpt-5.3-chat_2026-03-03")
    ap.add_argument("--judge_base", default="http://point.dd.works:18890/v1")
    ap.add_argument("--judge_key", default="")
    ap.add_argument("--judge_key_file", default="/scratch/sheng/self_evolving/.trapi_key")
    ap.add_argument("--wandb_project", default="healthbench-professional-eval")
    ap.add_argument("--gpus", default="2,3", help="CUDA_VISIBLE_DEVICES for serving (cleanup scoped to these only)")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--port", type=int, default=8210)
    ap.add_argument("--max_model_len", type=int, default=16384)
    ap.add_argument("--gpu_mem_util", type=float, default=0.85)
    ap.add_argument("--max_num_seqs", type=int, default=256)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="0 = full 525")
    ap.add_argument("--serve_timeout", type=int, default=1200)
    ap.add_argument("--serve_retries", type=int, default=2)
    ap.add_argument("--only", default="")
    ap.add_argument("--keep_merged", action="store_true")
    ap.add_argument("--reverse", action="store_true", help="evaluate latest steps first")
    ap.add_argument("--shard", default="0/1",
                    help="N/M: process only checkpoints with index%M==N (disjoint "
                         "parallel workers, race-free merges). Default 0/1 = all.")
    args = ap.parse_args()

    if not args.judge_key and Path(args.judge_key_file).exists():
        args.judge_key = Path(args.judge_key_file).read_text().strip()
    out = Path(args.out_dir)
    for sub in ("merged", "logs", "out", "files"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    ckpts = discover(args.ckpt_root)
    if args.only:
        ckpts = [c for c in ckpts if args.only in tag_for(c["exp"], c["step"])]
    if args.reverse:
        ckpts = ckpts[::-1]
    sn, sm = (int(x) for x in args.shard.split("/"))
    if sm > 1:
        ckpts = [c for i, c in enumerate(ckpts) if i % sm == sn]
    log(f"discovered {len(ckpts)} evaluable checkpoints{' (filtered)' if args.only else ''}"
        f"{f' [shard {sn}/{sm}]' if sm > 1 else ''}")

    done, failed = [], []
    for c in ckpts:
        tag = tag_for(c["exp"], c["step"])
        summary_json = out / "out" / f"{tag}.summary.json"
        if summary_json.exists():
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
            log(f"  norm-fill failed (continuing): {type(e).__name__}: {e}")

        ok = False
        for attempt in range(1, args.serve_retries + 1):
            if not wait_gpu_free(args):
                log(f"  our GPUs ({args.gpus}) not free before {tag} (attempt {attempt}) — scoped teardown")
                kill_our_gpu_procs(args)
                wait_gpu_free(args)
            proc = serve(merged, tag, str(out / "logs" / f"serve_{tag}.log"), args)
            try:
                if not wait_health(args.port, proc, args.serve_timeout):
                    log(f"  serve unhealthy for {tag} (attempt {attempt}/{args.serve_retries})")
                    continue
                log(f"server healthy; running HealthBench eval for {tag} (attempt {attempt})")
                ok = run_eval(tag, tag, args)
            finally:
                stop(proc, args)
            if ok:
                break
            log(f"  eval did not complete for {tag} (attempt {attempt}/{args.serve_retries})")
        (done if ok else failed).append(tag)
        if not args.keep_merged:
            shutil.rmtree(merged, ignore_errors=True)
        log(f"finished {tag}: {'OK' if ok else 'FAILED'} ({len(done)} ok / {len(failed)} failed so far)")

    log(f"DONE. ok={len(done)} failed={len(failed)}")
    if failed:
        log(f"failed: {failed}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
