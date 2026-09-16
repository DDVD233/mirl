#!/usr/bin/env python3
"""Pin the best-validation checkpoint of each general arm where the checkpoints live.

verl keeps one checkpoint per run (max_actor_ckpt_to_keep=1), so a run whose validation
peaks and then declines loses its best weights. This loop reads each arm's HealthBench-Pro
validation curve from wandb, finds the best step among the COMPLETE checkpoints present,
and hard-links that directory to best_global_step_<N> on the same filesystem (instant; the
link tree keeps the blocks alive when verl rotates the original). Older best_* pins are
removed when a better one appears.

Default target is the MSR NFS through pod 2333 (2026-09-15: David asked for no continuous
checkpoint transfer to AICR; the one-time transfer at the end of an arm is
scripts/self_evolving/aicr/msr_backup_once.sh). CKPT_SSH / CKPT_ROOT switch the target.
Started on mib in tmux `main:keeper`:

    /home/dvd/miniconda3/envs/new2/bin/python scripts/self_evolving/aicr/keep_best_ckpt.py
"""
import os
import shlex
import subprocess
import time

import wandb

SSH_CMD = shlex.split(os.environ.get(
    "CKPT_SSH", "ssh -o ConnectTimeout=30 -o BatchMode=yes -o StrictHostKeyChecking=no "
                "-o UserKnownHostsFile=/dev/null -p 2333 root@point.dd.works"))
B = os.environ.get("CKPT_ROOT", "/scratch/sheng/self_evolving/checkpoints/hb9b")
ARMS = ["hb27b_general_specgap_ship_retrieval_websearch", "hb27b_general_simple_retrieval_websearch",
        "hb9b_general_specgap_ship_retrieval_websearch2", "hb9b_general_simple_retrieval_websearch2",
        "hb9b_general_evolveonly_retrieval_websearch", "hb27b_general_evolveonly_retrieval_websearch"]
VAL = "val-core/healthbench_professional/acc/mean@1"


def ssh(cmd: str) -> str:
    return subprocess.run(SSH_CMD + [cmd], capture_output=True, text=True, timeout=300).stdout


def main() -> None:
    while True:
        # A fresh Api per pass: wandb.Api caches the result of runs() per query, so a
        # long-lived object keeps returning the run list from its first call and never
        # sees an arm's later attempts (2026-09-15: pinned step 10 with attempt 1's score).
        api = wandb.Api(timeout=120)
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
                # verl writes latest_checkpointed_iteration.txt after a save completes; a
                # directory above that mark is still being written.
                latest_txt = ssh(f"cat {B}/{exp}/latest_checkpointed_iteration.txt 2>/dev/null").strip()
                latest = int(latest_txt) if latest_txt.isdigit() else -1
                complete = [s for s in present if s <= latest
                            and ssh(f"ls {B}/{exp}/global_step_{s}/actor/*.pt >/dev/null 2>&1 && echo ok").strip() == "ok"]
                # An existing pin is a candidate too: on the NFS verl keeps only the latest
                # checkpoint, so once the true best is rotated the pin is its only copy
                # (2026-09-16: the pin of the 27B's step 70 was dropped for a worse step 80).
                pinned = ssh(f"ls -d {B}/{exp}/best_global_step_* 2>/dev/null").split()
                pinned_steps = [int(p.split("_")[-1]) for p in pinned if p.split("_")[-1].isdigit()]
                scored = [(vals[s], s) for s in set(complete) | set(pinned_steps) if s in vals]
                if not scored:
                    print(time.strftime("%FT%TZ", time.gmtime()), exp, "no scored checkpoint yet", flush=True)
                    continue
                best_val, best = max(scored)
                if best in pinned_steps:
                    print(time.strftime("%FT%TZ", time.gmtime()), exp, f"best stays step {best} ({best_val:.3f})", flush=True)
                    continue
                # Only a strictly better complete checkpoint replaces the pin.
                ssh(f"cp -al {B}/{exp}/global_step_{best} {B}/{exp}/best_global_step_{best}")
                if ssh(f"ls {B}/{exp}/best_global_step_{best}/actor/*.pt >/dev/null 2>&1 && echo ok").strip() != "ok":
                    print(time.strftime("%FT%TZ", time.gmtime()), exp, f"pin of step {best} incomplete; keeping old pins", flush=True)
                    continue
                for p in pinned:
                    if int(p.split("_")[-1]) != best:
                        ssh(f"rm -rf {p}")
                print(time.strftime("%FT%TZ", time.gmtime()), exp, f"PINNED step {best} ({best_val:.3f}); candidates {sorted(scored)}", flush=True)
            except Exception as e:  # keep the loop alive across transient ssh/wandb failures
                print(time.strftime("%FT%TZ", time.gmtime()), exp, "error:", str(e)[:200], flush=True)
        time.sleep(1800)


if __name__ == "__main__":
    main()
