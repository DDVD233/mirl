"""Back up the 2026-07 checkpoint sweep's best surviving checkpoints to
HuggingFace (ddvd233/<run>) as merged HF models, then free the merged copy.
Sequential + resumable (skips repos that already contain safetensors).

Selection (wandb project self_evolving_medical, val-core/overall/acc/mean,
best step *with surviving actor weights* — checkpoint rotation dropped most):
  - mimiciv_rare_qwen36_27b_evolve_from_sft   step 960  val 0.3397 (baseline 0.285)
  - mimiciv_rare_qwen36_27b_trainset_rl_from_sft step 180 val 0.3328 (baseline 0.277)
  - healthbench_rubric_qwen36_27b_v6          step 60   val 0.5519 (baseline 0.548)
  - mimiciv_rare_qwen36_27b_sft_distill       step 90   (SFT base of the RL runs;
    actor/huggingface already holds merged safetensors -> direct upload)

Raw FSDP checkpoints are NOT deleted here (done separately after verifying).

Run on server 1:
    /usr/local/bin/python scripts/self_evolving/backup_checkpoints_to_hf_v2.py
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

# (run_name, checkpoint dir, step, already-merged hf dir or None, extra folder, note)
TARGETS = [
    ("mimiciv_rare_qwen36_27b_evolve_from_sft",
     f"{CKPT}/self_evolving_medical/mimiciv_rare_qwen36_27b_evolve_from_sft", 960, None,
     f"{CKPT}/self_evolving_medical/mimiciv_rare_qwen36_27b_evolve_from_sft/reward_evolution",
     "best surviving checkpoint (step 960, val acc 0.3397 vs 0.285 baseline)"),
    ("mimiciv_rare_qwen36_27b_trainset_rl_from_sft",
     f"{CKPT}/self_evolving_medical/mimiciv_rare_qwen36_27b_trainset_rl_from_sft", 180, None, None,
     "best surviving checkpoint (step 180, val acc 0.3328 vs 0.277 baseline)"),
    ("healthbench_rubric_qwen36_27b_v6",
     f"{CKPT}/healthbench_rubric/healthbench_rubric_qwen36_27b_v6", 60, None, None,
     "best surviving checkpoint (step 60, val acc 0.5519 vs 0.548 baseline)"),
    ("mimiciv_rare_qwen36_27b_sft_distill",
     f"{CKPT}/self_evolving_medical/mimiciv_rare_qwen36_27b_sft_distill", 90,
     f"{CKPT}/self_evolving_medical/mimiciv_rare_qwen36_27b_sft_distill/global_step_90/actor/huggingface",
     None, "SFT distill warm-start (step 90), base model of the mimic-rare RL runs"),
]


def log(m):
    print(f"[hfbackup2 {time.strftime('%H:%M:%S')}] {m}", flush=True)


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
    done, failed = [], []
    for run, ckdir, step, premerged, extra, note in TARGETS:
        repo_id = f"{HF_ORG}/{run}"
        if repo_done(repo_id):
            log(f"SKIP {run} (repo {repo_id} already has weights)")
            done.append(run)
            continue
        if premerged and glob.glob(f"{premerged}/*.safetensors"):
            merged, cleanup = premerged, False
        else:
            actor = f"{ckdir}/global_step_{step}/actor"
            if not glob.glob(f"{actor}/model_world_size_*_rank_0.pt"):
                log(f"MISS {run}: no actor at {actor}"); failed.append(run); continue
            merged, cleanup = f"{WORK}/{run}", True
            shutil.rmtree(merged, ignore_errors=True)
            log(f"merging {run} step{step} -> {merged}")
            rc = subprocess.call([PY, "-m", "verl.model_merger", "merge", "--backend", "fsdp",
                                  "--local_dir", actor, "--target_dir", merged],
                                 cwd=REPO, env={**os.environ, "PYTHONPATH": REPO})
            if rc != 0 or not glob.glob(f"{merged}/*.safetensors"):
                log(f"MERGE FAILED {run}"); failed.append(run); continue
            try:
                fill_norms(merged, base_safetensors())
            except Exception as e:
                log(f"  norm-fill warn: {e}")
        log(f"uploading {run} -> {repo_id}")
        try:
            api.create_repo(repo_id, repo_type="model", private=False, exist_ok=True)
            try:
                api.update_repo_settings(repo_id=repo_id, private=False)
            except Exception:
                pass
            api.upload_folder(folder_path=merged, repo_id=repo_id, repo_type="model",
                              commit_message=f"{note} of {run}")
            if extra and os.path.isdir(extra):
                api.upload_folder(folder_path=extra, repo_id=repo_id, repo_type="model",
                                  path_in_repo="reward_evolution",
                                  commit_message="evolved reward artifacts")
            ok = repo_done(repo_id)
        except Exception as e:
            log(f"UPLOAD FAILED {run}: {type(e).__name__}: {e}"); failed.append(run)
            if cleanup:
                shutil.rmtree(merged, ignore_errors=True)
            continue
        if cleanup:
            shutil.rmtree(merged, ignore_errors=True)
        (done if ok else failed).append(run)
        log(f"finished {run}: {'OK' if ok else 'VERIFY-FAILED'}")
    log(f"DONE backups ok={len(done)} failed={len(failed)}")
    if failed:
        log(f"failed: {failed}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
