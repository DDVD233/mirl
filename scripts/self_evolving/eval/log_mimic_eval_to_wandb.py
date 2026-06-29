#!/usr/bin/env python3
"""Log mimic-rare checkpoint-eval summaries (run_checkpoint_eval_sweep.py output) to W&B.

The mimic sweep writes {out_dir}/out/<tag>.summary.json per checkpoint (overall + by_category
metrics, judged here by gpt-5.5). This logs them to one W&B run with the checkpoint step on the
x-axis, so the two checkpoints (step 20 vs 40) are directly comparable. Eval-only.
"""
import argparse, glob, json, os, re

import wandb

ap = argparse.ArgumentParser()
ap.add_argument("--out_dir", required=True, help="eval out_dir containing out/*.summary.json")
ap.add_argument("--project", default="mimic-rare-eval-gpt55")
ap.add_argument("--run_name", default="mimic-eval-gpt55")
args = ap.parse_args()

files = sorted(glob.glob(os.path.join(args.out_dir, "out", "*.summary.json")))
if not files:
    raise SystemExit(f"no summary.json under {args.out_dir}/out")

run = wandb.init(project=args.project, name=args.run_name)
print(f"logging {len(files)} checkpoint summaries to wandb {args.project}/{args.run_name}")
for f in files:
    d = json.load(open(f))
    tag = os.path.basename(f).replace(".summary.json", "")
    m = re.search(r"step[_]?(\d+)", tag)
    step = int(m.group(1)) if m else 0
    payload = {"mimic_eval/n_total": d.get("n_total", 0), "mimic_eval/judge": 0}
    for k, v in (d.get("overall") or {}).items():
        if isinstance(v, (int, float)):
            payload[f"mimic_eval/overall/{k}"] = v
    for cat, cm in (d.get("by_category") or {}).items():
        for k, v in (cm or {}).items():
            if isinstance(v, (int, float)):
                payload[f"mimic_eval/{cat}/{k}"] = v
    wandb.log(payload, step=step)
    ov = d.get("overall", {})
    print(f"  {tag} (step {step}): acc={ov.get('acc')} lenient={ov.get('judge_acc_lenient')} "
          f"answer_q={ov.get('answer_quality')} score={ov.get('score')} judge={d.get('judge_model')}")
run.finish()
print("done")
