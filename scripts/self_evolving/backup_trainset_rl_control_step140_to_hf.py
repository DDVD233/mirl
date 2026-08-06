"""Back up trainset-RL control checkpoint (step 140) to HuggingFace before
freeing server1/server5 (2026-07-31 capacity clear-out).

Merges FSDP shards -> HF safetensors (verl.model_merger), fills missing 1-D
norm tensors from the Qwen3.6-27B base, uploads to ddvd233/<run>, verifies.
Also uploads the small dataloader state (data.pt) so an exact-ish resume point
survives even if the raw 306 GB FSDP dir is lost. Resumable: skips if the repo
already has safetensors.

Run on server 1:
    /usr/local/bin/python scripts/self_evolving/backup_trainset_rl_control_step140_to_hf.py
"""

import glob
import json
import os
import shutil
import subprocess
import sys
import time

REPO = "/scratch/sheng/self_evolving/verl"
CKPT = "/scratch/sheng/self_evolving/checkpoints"
HF_CACHE = "/scratch/sheng/self_evolving/hf_cache/hub"
WORK = "/scratch/sheng/self_evolving/hf_backup"
PY = "/usr/local/bin/python"
HF_ORG = "ddvd233"

RUN = "mimiciv_rare_qwen36_27b_trainset_rl_control"
STEP = 140
CKDIR = f"{CKPT}/self_evolving_medical/{RUN}"
NOTE = ("mid-run backup: 27B train-set RL control (2x2 scaling ablation), "
        "step 140 of 500, saved before freeing the node 2026-07-31")


def log(m):
    print(f"[hfbackup-ctrl {time.strftime('%H:%M:%S')}] {m}", flush=True)


def base_safetensors():
    snaps = sorted(glob.glob(f"{HF_CACHE}/models--Qwen--Qwen3.6-27B/snapshots/*"))
    return sorted(glob.glob(f"{snaps[-1]}/*.safetensors")) if snaps else []


def fill_norms(merged, base_files):
    if not base_files:
        return
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    mk = set()
    for f in sorted(glob.glob(f"{merged}/*.safetensors")):
        with safe_open(f, framework="pt") as h:
            mk |= set(h.keys())
    add = {}
    for bf in base_files:
        with safe_open(bf, framework="pt") as h:
            for k in h.keys():
                if k not in mk and k not in add:
                    t = h.get_tensor(k)
                    if t.dim() == 1:
                        add[k] = t.to(torch.bfloat16)
    if not add:
        return
    log(f"  norm-fill: +{len(add)} 1-D tensors from base")
    idx = f"{merged}/model.safetensors.index.json"
    save_file(add, f"{merged}/model-fill.safetensors", metadata={"format": "pt"})
    j = json.load(open(idx))
    for k in add:
        j.setdefault("weight_map", {})[k] = "model-fill.safetensors"
    j.setdefault("metadata", {})["total_size"] = j["metadata"].get("total_size", 0) + sum(
        t.numel() * t.element_size() for t in add.values())
    json.dump(j, open(idx, "w"))


def repo_done(repo_id):
    try:
        from huggingface_hub import HfApi
        return any(f.endswith(".safetensors") for f in HfApi().list_repo_files(repo_id))
    except Exception:
        return False


def main():
    from huggingface_hub import HfApi
    os.makedirs(WORK, exist_ok=True)
    api = HfApi()
    repo_id = f"{HF_ORG}/{RUN}"
    if repo_done(repo_id):
        log(f"SKIP merge: repo {repo_id} already has weights")
        merged, cleanup = None, False
    else:
        actor = f"{CKDIR}/global_step_{STEP}/actor"
        if not glob.glob(f"{actor}/model_world_size_*_rank_0.pt"):
            log(f"FATAL: no actor weights at {actor}")
            return 1
        merged, cleanup = f"{WORK}/{RUN}", True
        shutil.rmtree(merged, ignore_errors=True)
        log(f"merging {RUN} step{STEP} -> {merged}")
        rc = subprocess.call([PY, "-m", "verl.model_merger", "merge", "--backend", "fsdp",
                              "--local_dir", actor, "--target_dir", merged],
                             cwd=REPO, env={**os.environ, "PYTHONPATH": REPO})
        if rc != 0 or not glob.glob(f"{merged}/*.safetensors"):
            log("MERGE FAILED")
            return 1
        try:
            fill_norms(merged, base_safetensors())
        except Exception as e:
            log(f"  norm-fill warn: {e}")
    log(f"uploading -> {repo_id}")
    try:
        api.create_repo(repo_id, repo_type="model", private=False, exist_ok=True)
        if merged:
            api.upload_folder(folder_path=merged, repo_id=repo_id, repo_type="model",
                              commit_message=f"{NOTE} of {RUN}")
        data_pt = f"{CKDIR}/global_step_{STEP}/data.pt"
        if os.path.isfile(data_pt) and os.path.getsize(data_pt) < 5 * 2**30:
            api.upload_file(path_or_fileobj=data_pt, path_in_repo=f"verl_state/global_step_{STEP}/data.pt",
                            repo_id=repo_id, repo_type="model",
                            commit_message="dataloader state for exact resume")
        ok = repo_done(repo_id)
    except Exception as e:
        log(f"UPLOAD FAILED: {type(e).__name__}: {e}")
        return 1
    finally:
        if cleanup and merged:
            shutil.rmtree(merged, ignore_errors=True)
    log(f"DONE: {'OK' if ok else 'VERIFY-FAILED'} -> https://huggingface.co/{repo_id}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
