#!/usr/bin/env python
"""Summarize cached W&B runs for self_evolving_medical.

Main metric = lenient accuracy:
  prefer  val-aux/<ds>/judge_acc_lenient/mean@1   (true LLM-judge lenient match)
  else    val-core/<ds>/acc/mean@1                (headline; == lenient for runs
                                                    after the 2026-06-10 reweight,
                                                    EM for older/pubmedqa/pmcvqa)
Secondary = embedding similarity  val-aux/<ds>/embed_sim/mean@1
  (biobert_sim is the OUTDATED encoder -> ignored.)

Usage:
  /home/dvd/miniconda3/envs/new2/bin/python analyze_wandb_runs.py
Writes summary.md and summary.csv into the cache dir.
"""
import json
import os
import re
import glob
import math
import collections

import pandas as pd

CACHE = "/home/dvd/mirl/scripts/self_evolving/wandb_cache"


def family(name):
    """Collapse a run name to an experiment family (strip timestamps/run salt)."""
    n = re.sub(r"_\d{8}_\d{6}$", "", name)
    n = re.sub(r"_\d{8}$", "", n)
    n = re.sub(r"_step\d+$", "", n)
    return n


def col_max_last(df, col):
    if col not in df.columns:
        return None, None
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if s.empty:
        return None, None
    return float(s.max()), float(s.iloc[-1])


def data_sources_in(keys):
    ds = set()
    for k in keys:
        m = re.match(r"val-(?:core|aux)/([^/]+)/", k)
        if m:
            ds.add(m.group(1))
    return ds


def lenient_for_ds(df, keys, ds):
    """Return (max, last) lenient acc for a data source."""
    lk = f"val-aux/{ds}/judge_acc_lenient/mean@1"
    if lk in keys:
        return col_max_last(df, lk) + ("judge_lenient",)
    ak = f"val-core/{ds}/acc/mean@1"
    if ak in keys:
        return col_max_last(df, ak) + ("acc(EM)",)
    return None, None, None


def embed_for_ds(df, keys, ds):
    ek = f"val-aux/{ds}/embed_sim/mean@1"
    if ek in keys:
        return col_max_last(df, ek)
    return None, None


def main():
    idx = json.load(open(f"{CACHE}/index.json"))
    rows = []
    for m in idx:
        rid = m["id"]
        keys = set(m.get("history_keys", []))
        hp = f"{CACHE}/history/{rid}.parquet"
        df = pd.DataFrame()
        if os.path.exists(hp):
            try:
                df = pd.read_parquet(hp)
            except Exception:
                pass

        dss = data_sources_in(keys)
        # primary data source = the one we care about; mimiciv/self_evolving/pubmedqa
        # a run may validate on several; emit one row per (run, ds) that has lenient.
        emitted = False
        for ds in sorted(dss):
            lmax, llast, lsrc = lenient_for_ds(df, keys, ds)
            emax, elast = embed_for_ds(df, keys, ds)
            if lmax is None and emax is None:
                continue
            rows.append({
                "family": family(m["name"]),
                "name": m["name"],
                "id": rid,
                "state": m["state"],
                "ds": ds,
                "lenient_max": lmax,
                "lenient_last": llast,
                "lenient_src": lsrc,
                "embed_max": emax,
                "embed_last": elast,
                "steps": int(m.get("last_step") or 0),
                "runtime_min": round((m.get("runtime_s") or 0) / 60, 1),
                "created": m.get("created_at", "")[:19],
            })
            emitted = True
        if not emitted:
            rows.append({
                "family": family(m["name"]),
                "name": m["name"],
                "id": rid,
                "state": m["state"],
                "ds": "-",
                "lenient_max": None, "lenient_last": None, "lenient_src": None,
                "embed_max": None, "embed_last": None,
                "steps": int(m.get("last_step") or 0),
                "runtime_min": round((m.get("runtime_s") or 0) / 60, 1),
                "created": m.get("created_at", "")[:19],
            })

    df = pd.DataFrame(rows)
    df.to_csv(f"{CACHE}/summary.csv", index=False)

    has_val = df[df["lenient_max"].notna()].copy()
    no_val = df[df["lenient_max"].isna()].copy()

    print(f"Total runs: {len(idx)}")
    print(f"Run-rows with a lenient metric: {len(has_val)}")
    print(f"Run-rows with NO val metric (crashed/no-eval): {len(no_val)}")
    print(f"Data sources seen: {sorted(df['ds'].unique())}\n")

    # ---- best run per family (by lenient_max), per data source ----
    print("=" * 100)
    print("BEST lenient accuracy per experiment family (data source in brackets)")
    print("=" * 100)
    best = (has_val.sort_values("lenient_max", ascending=False)
                  .groupby(["family", "ds"], as_index=False).first())
    best = best.sort_values("lenient_max", ascending=False)
    for _, r in best.iterrows():
        emb = f"{r.embed_max:.3f}" if r.embed_max is not None and not math.isnan(r.embed_max) else "  -  "
        print(f"  {r.lenient_max:6.3f} (last {r.lenient_last:6.3f}) [{r.lenient_src:12s}] "
              f"emb {emb}  {r.steps:4d}st  {r.ds:18s} {r.family}")

    # markdown
    with open(f"{CACHE}/summary.md", "w") as f:
        f.write("# self_evolving_medical — W&B run summary\n\n")
        f.write(f"- Total runs cached: **{len(idx)}**\n")
        f.write(f"- Run-rows with a lenient-accuracy val metric: **{len(has_val)}**\n")
        f.write(f"- Run-rows with no val metric (crashed/no-eval): **{len(no_val)}**\n")
        f.write("- Main metric = lenient accuracy "
                "(`judge_acc_lenient/mean@1`, fallback headline `acc/mean@1`).\n")
        f.write("- Secondary = embedding similarity (`embed_sim/mean@1`); "
                "biobert_sim ignored (outdated).\n\n")

        f.write("## Best lenient accuracy per experiment family\n\n")
        f.write("| lenient max | lenient last | metric src | embed max | steps | data source | family |\n")
        f.write("|---:|---:|:--|---:|---:|:--|:--|\n")
        for _, r in best.iterrows():
            emb = f"{r.embed_max:.3f}" if r.embed_max is not None and not math.isnan(r.embed_max) else "-"
            f.write(f"| {r.lenient_max:.3f} | {r.lenient_last:.3f} | {r.lenient_src} | {emb} "
                    f"| {r.steps} | {r.ds} | {r.family} |\n")

        f.write("\n## All run-rows with val metrics (sorted by lenient max)\n\n")
        f.write("| lenient max | last | src | embed | steps | state | ds | name | id |\n")
        f.write("|---:|---:|:--|---:|---:|:--|:--|:--|:--|\n")
        for _, r in has_val.sort_values("lenient_max", ascending=False).iterrows():
            emb = f"{r.embed_max:.3f}" if r.embed_max is not None and not math.isnan(r.embed_max) else "-"
            f.write(f"| {r.lenient_max:.3f} | {r.lenient_last:.3f} | {r.lenient_src} | {emb} | "
                    f"{r.steps} | {r.state} | {r.ds} | {r.name} | {r.id} |\n")

        f.write("\n## Runs with NO val metric (crashed / never evaluated)\n\n")
        fam_counts = collections.Counter(no_val["family"])
        f.write("| family | n runs | states |\n|:--|---:|:--|\n")
        for fam, c in fam_counts.most_common():
            states = ",".join(sorted(set(no_val[no_val.family == fam].state)))
            f.write(f"| {fam} | {c} | {states} |\n")

    print(f"\nWrote {CACHE}/summary.md and summary.csv")


if __name__ == "__main__":
    main()
