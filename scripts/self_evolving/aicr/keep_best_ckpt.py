#!/usr/bin/env python3
"""Pin the best-validation checkpoint of each general arm on AICR.

verl keeps one checkpoint on the NFS and the hourly backup keeps the last two on AICR,
so a run whose validation peaks and then declines would lose its best weights. This
loop reads each arm's HealthBench-Pro validation curve from wandb, finds the best
checkpointed step among the copies present on AICR, and hard-links that directory to
best_global_step_<N> (same filesystem, instant, no extra quota beyond the link tree).
Older best_* pins are removed when a better one appears. Runs anywhere with wandb and
ssh to AICR; started on mib in tmux `main:keeper`.

    /home/dvd/miniconda3/envs/new2/bin/python scripts/self_evolving/aicr/keep_best_ckpt.py
"""
import subprocess
import time

import wandb

AICR = "dvdai_mit@login.aicr.ai"
B = "/scratch/dvdai_mit/msr_backup_2026-09-11/checkpoints/hb9b"
ARMS = ["hb9b_general_specgap_ship_retrieval_websearch", "hb27b_general_specgap_ship_retrieval_websearch",
        "hb9b_general_simple_retrieval_websearch", "hb27b_general_simple_retrieval_websearch",
        "hb9b_general_specgap_ship_retrieval_websearch2", "hb9b_general_simple_retrieval_websearch2"]
VAL = "val-core/healthbench_professional/acc/mean@1"


def ssh(cmd: str) -> str:
    return subprocess.run(["ssh", "-o", "ConnectTimeout=30", "-o", "BatchMode=yes", AICR, cmd],
                          capture_output=True, text=True, timeout=300).stdout


def main() -> None:
    api = wandb.Api(timeout=120)
    while True:
        for exp in ARMS:
            try:
                runs = sorted(api.runs("ddavid233/self_evolving_medical", filters={"display_name": exp}),
                              key=lambda r: r._attrs.get("createdAt", ""))
                if not runs:
                    continue
                vals = {}
                for row in runs[-1].scan_history(keys=["_step", VAL], page_size=3000):
                    if row.get(VAL) is not None:
                        vals[int(row["_step"])] = float(row[VAL])
                present = [int(d.split("_")[-1]) for d in
                           ssh(f"ls -d {B}/{exp}/global_step_* 2>/dev/null").split() if d.split("_")[-1].isdigit()]
                complete = [s for s in present if ssh(f"[ -f {B}/{exp}/global_step_{s}/data.pt ] && echo ok").strip() == "ok"]
                scored = [(vals[s], s) for s in complete if s in vals]
                if not scored:
                    print(time.strftime("%FT%TZ", time.gmtime()), exp, "no scored checkpoint on AICR yet", flush=True)
                    continue
                best_val, best = max(scored)
                pinned = ssh(f"ls -d {B}/{exp}/best_global_step_* 2>/dev/null").split()
                pinned_steps = [int(p.split("_")[-1]) for p in pinned]
                if best in pinned_steps and len(pinned_steps) == 1:
                    print(time.strftime("%FT%TZ", time.gmtime()), exp, f"best stays step {best} ({best_val:.3f})", flush=True)
                    continue
                if best not in pinned_steps:
                    ssh(f"cp -al {B}/{exp}/global_step_{best} {B}/{exp}/best_global_step_{best}")
                for p in pinned:
                    if int(p.split("_")[-1]) != best:
                        ssh(f"rm -rf {p}")
                print(time.strftime("%FT%TZ", time.gmtime()), exp, f"PINNED step {best} ({best_val:.3f}); candidates {sorted(scored)}", flush=True)
            except Exception as e:  # keep the loop alive across transient ssh/wandb failures
                print(time.strftime("%FT%TZ", time.gmtime()), exp, "error:", str(e)[:200], flush=True)
        time.sleep(1800)


if __name__ == "__main__":
    main()
