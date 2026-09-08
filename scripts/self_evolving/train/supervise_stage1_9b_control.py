"""Restart only dead, owned tmux panes after their assigned GPUs are released."""

import fcntl
import subprocess
import time
from pathlib import Path

S = Path("/scratch/sheng/self_evolving")
LOG = S / "logs_stage1_9b_control"
TRAIN = S / "verl_stage1_9b_control/scripts/self_evolving/train"
CHECKPOINT = S / "checkpoints/self_evolving_medical/mimiciv_rare_qwen35_9b_trainset_selfjudge"
PANES = [
    ("stage1-selfjudge", [3], f"bash {TRAIN}/serve_stage1_9b_selfjudge.sh >> {LOG}/judge.log 2>&1"),
    (
        "stage1-control",
        [0, 1],
        f"VAL_BEFORE_TRAIN=False bash {TRAIN}/run_stage1_9b_control.sh >> {LOG}/train.log 2>&1",
    ),
    (
        "stage1-regrade",
        [],
        f"bash {S}/stage1_component_audit/code/watch_stage1_9b_control.sh >> {LOG}/fixed_judge.log 2>&1",
    ),
]


def command(args):
    return subprocess.check_output(args, text=True, timeout=30).strip()


def main():
    with (LOG / "supervisor.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            try:
                checkpoint = CHECKPOINT / "latest_checkpointed_iteration.txt"
                complete = checkpoint.exists() and int(checkpoint.read_text()) >= 1000
                memory = [
                    int(value)
                    for value in command(
                        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
                    ).splitlines()
                ]
                for name, gpus, launch in PANES:
                    if complete and name == "stage1-control":
                        continue
                    target = "main:" + name
                    dead = command(["tmux", "list-panes", "-t", target, "-F", "#{pane_dead}"])
                    if dead == "1" and all(memory[gpu] < 4096 for gpu in gpus):
                        subprocess.run(["tmux", "respawn-pane", "-t", target, launch], check=True, timeout=30)
                        print(f"Restarted {target} after pane exit and GPU release", flush=True)
                print(f"Heartbeat: GPU memory={memory}, training_complete={complete}", flush=True)
            except (ValueError, OSError, subprocess.SubprocessError) as exc:
                print(f"Probe failed; no destructive action: {exc}", flush=True)
            time.sleep(300)


if __name__ == "__main__":
    main()
