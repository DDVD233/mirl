"""Back up each MIMIC-rare run's best checkpoint to HuggingFace (ddvd233/<repo>)
as a merged HF model, then free the local merged copy. Sequential + resumable
(skips repos that already contain a model.safetensors / index).

For each run: merge FSDP actor -> HF safetensors (verl.model_merger), backfill
any base-model 1-D norms missing from the merge (Gemma-3n k_norm), upload the
merged folder, then delete the local merged dir. The raw FSDP checkpoints are
NOT deleted here (do that separately after verifying uploads).

Run on server 1 (has HF auth for ddvd233 + the checkpoints):
    /usr/local/bin/python scripts/self_evolving/backup_checkpoints_to_hf.py
"""

import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time

REPO = "/scratch/sheng/self_evolving/verl"
ROOT = f"{REPO}/checkpoints/self_evolving_medical"
HF_CACHE = "/scratch/sheng/self_evolving/hf_cache/hub"
WORK = "/scratch/sheng/self_evolving/hf_backup"
PY = "/usr/local/bin/python"
HF_ORG = "ddvd233"

# run -> best checkpoint step (from the gpt-5.3 eval results.json)
BEST = {
    "qwen36_27b_full_opd_qwen397b_v2": 15, "qwen36_27b_selfimprove": 20,
    "qwen36_27b_full_gpt55": 65, "gemma4_31b_selfimprove_split": 340,
    "qwen36_27b_deepseekv4pro": 20, "gemma4_31b_rl_gpt51judge": 240,
    "qwen36_27b_selfimprove_sft": 220, "qwen36_27b_sft_long": 300,
    "qwen36_27b_full_kimi": 90, "qwen35_9b_selfimprove": 200,
    "qwen35_9b_sft_long": 700, "qwen35_9b_full_gpt55": 75,
    "gemma4_e4b_rl_gpt51judge": 350, "gemma4_e4b_sft.gpt51_archived": 5,
    "gemma4_e4b_sft": 350,
}
BASE_REPO = [("gemma4_31b", "models--google--gemma-4-31B-it"),
             ("gemma4_e4b", "models--google--gemma-4-E4B-it"),
             ("qwen35_9b", "models--Qwen--Qwen3.5-9B"),
             ("qwen36_27b", "models--Qwen--Qwen3.6-27B")]


def log(m):
    print(f"[hfbackup {time.strftime('%H:%M:%S')}] {m}", flush=True)


def base_safetensors(run):
    repo = next((r for t, r in BASE_REPO if t in run), None)
    if not repo:
        return []
    snaps = sorted(glob.glob(f"{HF_CACHE}/{repo}/snapshots/*"))
    return sorted(glob.glob(f"{snaps[-1]}/*.safetensors")) if snaps else []


def fill_norms(merged, base_files):
    if not base_files:
        return
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
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
    if os.path.exists(idx):
        save_file(add, f"{merged}/model-fill.safetensors", metadata={"format": "pt"})
        j = json.load(open(idx))
        for k in add:
            j.setdefault("weight_map", {})[k] = "model-fill.safetensors"
        j.setdefault("metadata", {})["total_size"] = j["metadata"].get("total_size", 0) + sum(
            t.numel() * t.element_size() for t in add.values())
        json.dump(j, open(idx, "w"))
    else:
        sd = load_file(f"{merged}/model.safetensors")
        sd.update(add)
        save_file(sd, f"{merged}/model.safetensors", metadata={"format": "pt"})


def repo_done(repo_id):
    """True if the HF repo already has model weights uploaded."""
    try:
        from huggingface_hub import HfApi
        files = HfApi().list_repo_files(repo_id)
        return any(f.endswith(".safetensors") for f in files)
    except Exception:
        return False


def main():
    os.makedirs(WORK, exist_ok=True)
    done, failed = [], []
    for run, step in BEST.items():
        repo_id = f"{HF_ORG}/mimiciv_rare_{run}"
        if repo_done(repo_id):
            log(f"SKIP {run} (repo {repo_id} already has weights)")
            done.append(run)
            continue
        exp = sorted(glob.glob(f"{ROOT}/mimiciv_rare_{run}*"))
        if not exp:
            log(f"MISS {run}: no exp dir"); failed.append(run); continue
        actor = f"{exp[0]}/global_step_{step}/actor"
        if not glob.glob(f"{actor}/model_world_size_*_rank_0.pt"):
            log(f"MISS {run}: no actor at {actor}"); failed.append(run); continue
        merged = f"{WORK}/{run}"
        shutil.rmtree(merged, ignore_errors=True)
        log(f"merging {run} step{step} -> {merged}")
        rc = subprocess.call([PY, "-m", "verl.model_merger", "merge", "--backend", "fsdp",
                              "--local_dir", actor, "--target_dir", merged],
                             cwd=REPO, env={**os.environ, "PYTHONPATH": REPO})
        if rc != 0 or not glob.glob(f"{merged}/*.safetensors"):
            log(f"MERGE FAILED {run}"); failed.append(run); continue
        try:
            fill_norms(merged, base_safetensors(run))
        except Exception as e:
            log(f"  norm-fill warn: {e}")
        log(f"uploading {run} -> {repo_id}")
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            api.create_repo(repo_id, repo_type="model", private=False, exist_ok=True)
            try:
                api.update_repo_settings(repo_id=repo_id, private=False)  # ensure public (free private quota)
            except Exception:
                pass
            api.upload_folder(folder_path=merged, repo_id=repo_id, repo_type="model",
                              commit_message=f"best checkpoint (step {step}) of {run}")
            ok = repo_done(repo_id)
        except Exception as e:
            log(f"UPLOAD FAILED {run}: {type(e).__name__}: {e}"); failed.append(run)
            shutil.rmtree(merged, ignore_errors=True); continue
        shutil.rmtree(merged, ignore_errors=True)
        (done if ok else failed).append(run)
        log(f"finished {run}: {'OK' if ok else 'VERIFY-FAILED'}")
    log(f"DONE backups ok={len(done)} failed={len(failed)}")
    if failed:
        log(f"failed: {failed}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
