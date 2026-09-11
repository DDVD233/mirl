#!/usr/bin/env python3
"""Summarize the stage-2 transfer evals into one JSON (run on the MSR pod, then copy to
paper_data/stage2/transfer.json for the table renderer).

  HealthBench Hard  in-loop val-only dumps logs_hb9b/val_generations/<exp>/0.jsonl; the
                    official score is the mean per-example clipped rubric fraction
                    (`acc_raw`; no length term), plus the per-theme means.
  MedXpertQA text   eval_out/medxpertqa_text_<tag>.json from medxpertqa_eval.py (official
                    exact-match accuracy, truncation count).

Rows are declared below so the same names feed the renderer; a missing file leaves the
cell empty rather than failing.
"""
import glob
import json
import os
import sys

S = "/scratch/sheng/self_evolving"
ROWS = [
    # (setting, label, hbhard exp, medxpertqa tag)
    ("27B", "Untrained", "base27b_hbhard", "base27b"),
    ("27B", "Fixed prompt", "hb27b_fixed60_hbhard", "hb27b_fixed_step60"),
    ("27B", "RRIMed", "hb27b_ser200_hbhard", "hb27b_ser_step200"),
    ("9B no-retrieval", "Untrained", "base9b_noretr_hbhard", "base9b"),
    ("9B no-retrieval", "Fixed prompt", "hb9b_noretr_fixed860_hbhard", "hb9b_noretr_fixed_step860"),
    ("9B no-retrieval", "RRIMed", "hb9b_noretr_ser480_hbhard", "hb9b_noretr_ser_step480"),
]


def hbhard(exp):
    p = f"{S}/logs_hb9b/val_generations/{exp}/0.jsonl"
    if not os.path.exists(p):
        return None
    rows = [json.loads(l) for l in open(p) if l.strip()]
    if not rows:
        return None
    import pandas as pd
    val = pd.read_parquet(f"{S}/healthbench_hard_val.parquet")
    themes = [ei["use_case"] for ei in val["extra_info"]]
    if len(themes) != len(rows):
        themes = [None] * len(rows)
    by = {}
    for r, t in zip(rows, themes):
        by.setdefault(t, []).append(float(r.get("acc_raw") or 0))
    return {"n": len(rows), "acc_raw": sum(float(r.get("acc_raw") or 0) for r in rows) / len(rows),
            "acc_raw_signed": sum(float(r.get("acc_raw_signed") or 0) for r in rows) / len(rows),
            "think_unclosed": sum(1 for r in rows if not r.get("think_closed", 1)),
            "mean_chars": sum(len(r.get("extracted_answer") or "") for r in rows) / len(rows),
            "by_theme": {k: {"n": len(v), "acc_raw": sum(v) / len(v)} for k, v in by.items()}}


def medx(tag):
    p = f"{S}/eval_out/medxpertqa_text_{tag}.json"
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    summ = d.get("summary", d)
    keep = {k: v for k, v in summ.items() if not isinstance(v, (list, dict))}
    for k in ("by_task", "by_body_system", "by_question_type"):
        if k in summ:
            keep[k] = summ[k]
    return keep


out = {"rows": []}
for setting, label, exp, tag in ROWS:
    out["rows"].append({"setting": setting, "label": label, "hbhard_exp": exp, "medx_tag": tag,
                        "hbhard": hbhard(exp), "medxpertqa_text": medx(tag)})
json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else f"{S}/paper_refresh/transfer.json", "w"), indent=1)
for r in out["rows"]:
    h, m = r["hbhard"], r["medxpertqa_text"]
    print(f"{r['setting']:16s} {r['label']:14s} hbhard={h['acc_raw'] if h else None} "
          f"medx={m.get('accuracy') if m else None} trunc={m.get('truncated') if m else None}")
