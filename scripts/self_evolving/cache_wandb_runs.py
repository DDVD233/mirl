#!/usr/bin/env python
"""Cache every run from the `ddavid233/self_evolving_medical` W&B project locally.

Run with the env that has wandb installed, from a dir WITHOUT a local `wandb/`
folder (which shadows the package):

    cd /tmp && /home/dvd/miniconda3/envs/new2/bin/python \
        /home/dvd/mirl/scripts/self_evolving/cache_wandb_runs.py

Caches per run:
  - <cache>/runs/<id>.json   : id, name, state, timestamps, runtime, group,
                               tags, full config, full summary, history-keys
  - <cache>/history/<id>.parquet : full scan_history() metric table

Idempotent / resumable: skips runs already cached unless --refresh is passed.
Writes <cache>/index.json with the per-run metadata (no history) for fast load.
"""
import argparse
import json
import os
import sys
import time

ENTITY = "ddavid233"
PROJECT = "self_evolving_medical"
DEFAULT_CACHE = "/home/dvd/mirl/scripts/self_evolving/wandb_cache"


def jsonable(o):
    try:
        json.dumps(o)
        return o
    except (TypeError, ValueError):
        return str(o)


def clean_dict(d):
    return {k: jsonable(v) for k, v in dict(d).items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=DEFAULT_CACHE)
    ap.add_argument("--refresh", action="store_true", help="re-pull even if cached")
    ap.add_argument("--no-history", action="store_true", help="skip metric history")
    args = ap.parse_args()

    import wandb
    import pandas as pd

    runs_dir = os.path.join(args.cache, "runs")
    hist_dir = os.path.join(args.cache, "history")
    os.makedirs(runs_dir, exist_ok=True)
    os.makedirs(hist_dir, exist_ok=True)

    api = wandb.Api(timeout=120)
    runs = api.runs(f"{ENTITY}/{PROJECT}")
    total = len(runs)
    print(f"Found {total} runs in {ENTITY}/{PROJECT}")

    index = []
    for i, run in enumerate(runs):
        meta_path = os.path.join(runs_dir, f"{run.id}.json")
        hist_path = os.path.join(hist_dir, f"{run.id}.parquet")
        cached = os.path.exists(meta_path) and (
            args.no_history or os.path.exists(hist_path)
        )
        tag = f"[{i + 1}/{total}] {run.state:9s} {run.id} {run.name}"
        if cached and not args.refresh:
            with open(meta_path) as f:
                meta = json.load(f)
            index.append({k: meta[k] for k in meta if k != "config_full"})
            print(tag + "  (cached)")
            continue

        summary = clean_dict(run.summary._json_dict)
        config = clean_dict(run.config)
        hist_keys = sorted(summary.keys())
        meta = {
            "id": run.id,
            "name": run.name,
            "state": run.state,
            "group": run.group,
            "tags": list(run.tags) if run.tags else [],
            "created_at": str(run.created_at),
            "runtime_s": summary.get("_runtime"),
            "last_step": summary.get("_step"),
            "summary": summary,
            "history_keys": hist_keys,
            "config_full": config,
            # a few config highlights pulled to the top for quick scanning
            "model": config.get("actor_rollout_ref", {}).get("model", {}).get("path")
            if isinstance(config.get("actor_rollout_ref"), dict)
            else None,
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)

        rows = 0
        if not args.no_history:
            try:
                hist = list(run.scan_history())
                if hist:
                    df = pd.DataFrame(hist)
                    df.to_parquet(hist_path)
                    rows = len(df)
                else:
                    # crashed before logging anything; write empty marker
                    pd.DataFrame().to_parquet(hist_path)
            except Exception as e:
                print(f"    history failed for {run.id}: {type(e).__name__}: {e}")

        index.append({k: meta[k] for k in meta if k != "config_full"})
        print(tag + f"  cached ({rows} history rows)")
        time.sleep(0.05)

    with open(os.path.join(args.cache, "index.json"), "w") as f:
        json.dump(index, f, indent=2, default=str)
    print(f"\nWrote index for {len(index)} runs -> {args.cache}/index.json")


if __name__ == "__main__":
    sys.exit(main())
