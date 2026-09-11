"""2026-09-11 MSR shutdown sweep: push the one best surviving checkpoint of every
experiment that is either a paper result or a comparison baseline to HuggingFace
(public, ddvd233/<run>_global_step_<N>) as merged HF safetensors.

Selection rule (David, 2026-09-11): one best checkpoint per experiment; skip runs whose
results are not good unless they are a baseline. "Best" = best validation score among
the steps whose actor weights still exist (checkpoint rotation dropped most steps).

Two independent queues so a merge never blocks an upload:
    /usr/local/bin/python backup_checkpoints_to_hf_v3.py --queue premerged
    /usr/local/bin/python backup_checkpoints_to_hf_v3.py --queue merge
Resumable: a repo that already holds safetensors is skipped.

Run on MSR pod 2333 (the only pod still reachable); everything lives on the shared NFS.
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

S = "/scratch/sheng/self_evolving"
CK = f"{S}/checkpoints"
REPO = f"{S}/verl_specgap"  # the checkout that produced checkpoints/hf_merged on 2026-09-07
HF_CACHE = f"{S}/hf_cache/hub"
WORK = f"{S}/hf_backup_v3"
PY = "/usr/local/bin/python"
HF_ORG = "ddvd233"

BASE = {
    "Qwen3.5-9B": f"{HF_CACHE}/models--Qwen--Qwen3.5-9B/snapshots",
    "Qwen3.6-27B": f"{HF_CACHE}/models--Qwen--Qwen3.6-27B/snapshots",
}

# repo suffix, base model, premerged dir or None, raw actor dir or None, card fields
TARGETS = [
    # ---------------- MSR-only (lost at end of day) ----------------
    dict(repo="medxpert9b_specgap_ship_retrieval_websearch_global_step_180", base="Qwen3.5-9B",
         raw=f"{CK}/hb9b/medxpert9b_specgap_ship_retrieval_websearch/global_step_180/actor",
         queue="merge",
         card=dict(paper="RRIMed (self-evolving rewards), held-out arm on MedXpertQA (text + multimodal), ARM 21",
                   score="MedXpertQA validation accuracy 0.407 (step 170 eval; run best 0.427 at step 70, whose weights were rotated away; untrained 0.357)",
                   note="RL-only run from the base model with retrieval + web search tools; the reward is a one-criterion rubric so the grader yields exact-match accuracy.")),
    dict(repo="prbench9b_specgap_ship_websearch_global_step_420", base="Qwen3.5-9B",
         raw=f"{CK}/hb9b/prbench9b_specgap_ship_websearch/global_step_420/actor",
         queue="merge",
         card=dict(paper="RRIMed (self-evolving rewards), held-out arm on PRBench Hard (finance + legal), ARM 19",
                   score="PRBench Hard length-adjusted rubric accuracy 0.216 (run best 0.225 at step 350, rotated away; untrained 0.175)",
                   note="Web search only (no domain knowledge base).")),
    dict(repo="profbench9b_specgap_ship_websearch_global_step_120", base="Qwen3.5-9B",
         raw=f"{CK}/hb9b/profbench9b_specgap_ship_websearch/global_step_120/actor",
         queue="merge",
         card=dict(paper="RRIMed (self-evolving rewards), held-out arm on ProfBench, ARM 20",
                   score="ProfBench rubric accuracy 0.392 on the 40 tasks (run best 0.405 at step 170, rotated away; untrained 0.356)",
                   note="Web search only (no domain knowledge base).")),
    dict(repo="mimiciv_rare_qwen35_9b_trainset_selfjudge_global_step_100", base="Qwen3.5-9B",
         raw=f"{CK}/self_evolving_medical/mimiciv_rare_qwen35_9b_trainset_selfjudge/global_step_100/actor",
         queue="merge",
         card=dict(paper="RSIMed (self-evolving data) baseline: GRPO on the real MIMIC-IV rare-diagnosis training set with the frozen 9B as judge (9B train-set RL control)",
                   score="in-loop validation accuracy 0.284 (step 0) -> 0.296 (step 40); later evaluations were not retained in the surviving log; step 100 is the last checkpoint",
                   note="Comparison baseline for the 9B self-evolving runs.")),
    dict(repo="hb27b_specgap_ship_retrieval_websearch_global_step_200", base="Qwen3.6-27B",
         pre=f"{CK}/hf_merged/hb27b_ser_step200",
         queue="premerged",
         card=dict(paper="RRIMed-27B (self-evolving rewards), main HealthBench Professional run, ARM 16",
                   score="HealthBench Professional length-adjusted accuracy 0.485 at this step (run best 0.516 at step 120, whose weights were rotated away; untrained 0.374; fixed-prompt baseline 0.392)",
                   note="Used for the HealthBench Hard transfer and MedXpertQA transfer rows.")),
    dict(repo="mimiciv_rare_qwen36_27b_trainset_rl_control_resume140_global_step_280", base="Qwen3.6-27B",
         raw=f"{CK}/self_evolving_medical/mimiciv_rare_qwen36_27b_trainset_rl_control_resume140/global_step_280/actor",
         queue="merge",
         card=dict(paper="RSIMed (self-evolving data) baseline: GRPO on the real MIMIC-IV rare-diagnosis training set (27B train-set RL control), resumed from ddvd233/mimiciv_rare_qwen36_27b_trainset_rl_control (step 140), so this is control step 420 overall",
                   score="in-loop validation accuracy (lenient judge) 0.400 at this step; run best 0.407 at resumed step 350 (rotated away); step-140 init 0.395",
                   note="Comparison baseline for the self-evolving 27B runs.")),
    # ---------------- also on AICR (safe there), cheap because already merged ----------------
    dict(repo="hb27b_specgap_simple_retrieval_aicr_global_step_60", base="Qwen3.6-27B",
         pre=f"{CK}/hf_merged/hb27b_fixed_step60",
         queue="premerged",
         card=dict(paper="RRIMed paper baseline: identical RL recipe with the reward held fixed (fixed prompt), 27B, ARM 18",
                   score="HealthBench Professional length-adjusted accuracy 0.370 at this step (run best 0.392 at step 75, rotated away; untrained 0.374)",
                   note="Fixed-prompt control for RRIMed-27B.")),
    dict(repo="hb9b_specgap_ship_noretrieval_sj_aicr_global_step_480", base="Qwen3.5-9B",
         pre=f"{CK}/hf_merged/hb9b_noretr_ser_step480",
         queue="premerged",
         card=dict(paper="RRIMed-9B, no-retrieval setting with the frozen 9B as grader, ARM 11",
                   score="HealthBench Professional length-adjusted accuracy 0.391 at this step (run best 0.421 at step 460, rotated away; untrained 0.244; fixed-prompt baseline 0.381)",
                   note="Self-graded setting; every auxiliary role served by the frozen 9B.")),
    dict(repo="gpt56_sft_qwen36_27b_global_step_28", base="Qwen3.6-27B",
         pre=f"{CK}/hb9b_aicr/self_evolving_medical/gpt56_sft_qwen36_27b/global_step_28/hf_merged",
         queue="premerged",
         card=dict(paper="Reference baseline: Qwen3.6-27B supervised fine-tuned on GPT-5.6 traces for HealthBench Professional (distillation from a stronger teacher)",
                   score="HealthBench Professional length-adjusted accuracy 0.419 under the official no-tools protocol (untrained 0.381)",
                   note="The 9B counterpart is ddvd233/gpt56_sft_qwen35_9b_global_step_28.")),
]


def log(m):
    print(f"[hfbackup3 {time.strftime('%H:%M:%S')}] {m}", flush=True)


def base_files(base):
    snaps = sorted(glob.glob(f"{BASE[base]}/*"))
    return sorted(glob.glob(f"{snaps[-1]}/*.safetensors")) if snaps else []


def base_keys(base):
    snaps = sorted(glob.glob(f"{BASE[base]}/*"))
    idx = glob.glob(f"{snaps[-1]}/*.index.json") if snaps else []
    if idx:
        return set(json.load(open(idx[0]))["weight_map"].keys())
    return set()


def merged_keys(merged):
    from safetensors import safe_open
    mk = set()
    for f in sorted(glob.glob(f"{merged}/*.safetensors")):
        with safe_open(f, framework="pt") as h:
            mk |= set(h.keys())
    return mk


def fill_norms(merged, base):
    """Safety net from the July sweep: older mergers dropped 1-D (norm) tensors."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    mk = merged_keys(merged)
    bk = base_keys(base)
    # The base checkpoints ship a multi-token-prediction head (mtp.*) that verl never
    # trains or saves; every earlier merged upload lacks it and serves fine in vLLM.
    ignored = {k for k in bk - mk if k.startswith("mtp.")}
    if ignored:
        log(f"  key check: ignoring {len(ignored)} untrained mtp.* head tensors")
    missing = bk - mk - ignored
    if not missing:
        log(f"  key check: merged has all {len(bk)-len(ignored)} trained base keys")
        return True
    add, bad = {}, []
    for bf in base_files(base):
        with safe_open(bf, framework="pt") as h:
            for k in h.keys():
                if k in missing:
                    t = h.get_tensor(k)
                    if t.dim() == 1:
                        add[k] = t.to(torch.bfloat16)
                    else:
                        bad.append(k)
    if bad:
        log(f"  MISSING NON-1D TENSORS ({len(bad)}), e.g. {bad[:3]} -- refusing to upload")
        return False
    log(f"  norm-fill: +{len(add)} 1-D tensors from base")
    idx = f"{merged}/model.safetensors.index.json"
    save_file(add, f"{merged}/model-fill.safetensors", metadata={"format": "pt"})
    if os.path.exists(idx):
        j = json.load(open(idx))
    else:
        j = {"metadata": {"total_size": 0}, "weight_map": {}}
        single = glob.glob(f"{merged}/model.safetensors")
        if single:
            with safe_open(single[0], framework="pt") as h:
                for k in h.keys():
                    j["weight_map"][k] = "model.safetensors"
    for k in add:
        j.setdefault("weight_map", {})[k] = "model-fill.safetensors"
    j.setdefault("metadata", {})["total_size"] = j["metadata"].get("total_size", 0) + sum(
        t.numel() * t.element_size() for t in add.values())
    json.dump(j, open(idx, "w"))
    return True


def repo_done(api, repo_id):
    try:
        return any(f.endswith(".safetensors") for f in api.list_repo_files(repo_id))
    except Exception:
        return False


def write_card(merged, t):
    c = t["card"]
    step = t["repo"].rsplit("_global_step_", 1)[1]
    run = t["repo"].rsplit("_global_step_", 1)[0]
    txt = f"""---
base_model: Qwen/{t['base']}
license: apache-2.0
tags:
- medical
- reinforcement-learning
- verl
---

# {run} (global step {step})

Merged HF weights (bf16 safetensors) of the verl FSDP checkpoint `{run}/global_step_{step}`.

- **Base model:** Qwen/{t['base']}
- **Experiment:** {c['paper']}
- **Score:** {c['score']}
- **Note:** {c['note']}

This is the best checkpoint of the run whose weights survived checkpoint rotation; the
run's best-validation step is stated above when it differs.

Research artifact trained on model-written tasks and evaluated on one benchmark family.
**Not for clinical use.**
"""
    with open(f"{merged}/README.md", "w") as f:
        f.write(txt)


def process(api, t):
    repo_id = f"{HF_ORG}/{t['repo']}"
    if repo_done(api, repo_id):
        log(f"SKIP {t['repo']} (repo already has weights)")
        return True
    if t.get("pre"):
        merged, cleanup = t["pre"], False
        if not glob.glob(f"{merged}/*.safetensors"):
            log(f"MISS premerged {merged}")
            return False
    else:
        actor = t["raw"]
        if not glob.glob(f"{actor}/model_world_size_*_rank_0.pt"):
            log(f"MISS raw actor {actor}")
            return False
        merged, cleanup = f"{WORK}/{t['repo']}", True
        if not (os.path.exists(f"{merged}/MERGE_DONE") and glob.glob(f"{merged}/*.safetensors")):
            shutil.rmtree(merged, ignore_errors=True)
            log(f"merging {actor} -> {merged}")
            t0 = time.time()
            with open(f"{WORK}/merge_{t['repo']}.log", "w") as lf:
                rc = subprocess.call([PY, "-m", "verl.model_merger", "merge", "--backend", "fsdp",
                                      "--local_dir", actor, "--target_dir", merged],
                                     cwd=REPO, env={**os.environ, "PYTHONPATH": REPO, "HF_HOME": f"{S}/hf_cache"},
                                     stdout=lf, stderr=subprocess.STDOUT)
            if not glob.glob(f"{merged}/*.safetensors"):
                log(f"MERGE FAILED rc={rc} (see {WORK}/merge_{t['repo']}.log)")
                return False
            log(f"  merged in {time.time()-t0:.0f}s (rc={rc})")
            open(f"{merged}/MERGE_DONE", "w").close()
    try:
        if not fill_norms(merged, t["base"]):
            return False
    except Exception as e:
        log(f"  key-check warn: {type(e).__name__}: {e}")
    write_card(merged, t)
    log(f"uploading {merged} -> {repo_id}")
    t0 = time.time()
    try:
        api.create_repo(repo_id, repo_type="model", private=False, exist_ok=True)
        try:
            api.update_repo_settings(repo_id=repo_id, private=False)
        except Exception:
            pass
        api.upload_folder(folder_path=merged, repo_id=repo_id, repo_type="model",
                          ignore_patterns=["MERGE_DONE", "*.log", ".cache/**"],
                          commit_message=f"{t['repo']}: best surviving checkpoint (MSR shutdown sweep 2026-09-11)")
    except Exception as e:
        log(f"UPLOAD FAILED {repo_id}: {type(e).__name__}: {e}")
        return False
    ok = repo_done(api, repo_id)
    log(f"  upload {'OK' if ok else 'VERIFY-FAILED'} in {time.time()-t0:.0f}s")
    if ok and cleanup:
        shutil.rmtree(merged, ignore_errors=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", choices=["premerged", "merge", "all"], default="all")
    ap.add_argument("--only", default=None, help="substring filter on repo name")
    a = ap.parse_args()
    from huggingface_hub import HfApi
    os.makedirs(WORK, exist_ok=True)
    api = HfApi()
    log(f"HF user: {api.whoami()['name']}")
    done, failed = [], []
    for t in TARGETS:
        if a.queue != "all" and t["queue"] != a.queue:
            continue
        if a.only and a.only not in t["repo"]:
            continue
        try:
            ok = process(api, t)
        except Exception as e:
            log(f"ERROR {t['repo']}: {type(e).__name__}: {e}")
            ok = False
        (done if ok else failed).append(t["repo"])
    log(f"QUEUE {a.queue} DONE ok={len(done)} failed={len(failed)}")
    for r in done:
        log(f"  ok  https://huggingface.co/{HF_ORG}/{r}")
    for r in failed:
        log(f"  FAILED {r}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
