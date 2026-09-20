#!/usr/bin/env python3
"""Collect the HealthBench Professional reference rows (standalone official pipeline, no
tools, greedy, gpt-chat-latest grader) into one JSON for the paper's reference table.

Sources (all on the shared NFS):
  $S/logs/hbpro_eval_stage2/result__hbpro-<tag>.json   open models + our merged RRI-27B
  $S/logs/hbpro_eval/result__hbpro-<model>.json         TRAPI-served frontier models

Run on any MSR pod; copy the output to paper_data/stage2/reference_rows.json.
"""
import glob
import json
import os
import sys

S = "/scratch/sheng/self_evolving"
NAMES = {
    "hb27b_ser_step200": ("RRI, Qwen3.6-27B", "ours"),
    "Qwen_Qwen3.6-27B": ("Qwen3.6-27B", "open"),
    "Qwen_Qwen3.5-9B": ("Qwen3.5-9B", "open"),
    "google_gemma-4-31B-it": ("Gemma 4 31B IT", "open"),
    "lingshu-medical-mllm_Lingshu-32B": ("Lingshu-32B", "open"),
    "google_medgemma-27b-text-it": ("MedGemma 27B", "open"),
    "gpt56_sft_qwen36_27b": ("Qwen3.6-27B, SFT on GPT-5.6 traces", "sft"),
    "gpt-5.3-chat_2026-03-03": ("GPT-5.3 chat", "frontier"),
    "gpt-5.4_2026-03-05": ("GPT-5.4", "frontier"),
    "gpt-chat-latest_2026-05-28": ("GPT chat latest", "frontier"),
}
rows = []
for p in sorted(glob.glob(f"{S}/logs/hbpro_eval_stage2/result__hbpro-*.json") +
                glob.glob(f"{S}/logs/hbpro_eval/result__hbpro-*.json")):
    d = json.load(open(p))
    tag = d["model_under_test"]
    if d.get("n_examples") != 525 or "gpt-chat-latest" not in str(d.get("grader_model")):
        continue
    name, kind = NAMES.get(tag, (tag, "other"))
    rows.append({"tag": tag, "name": name, "kind": kind, "raw": d["overall_score"],
                 "len_adj": d["overall_score_length_adjusted"], "stamp": d.get("stamp"), "file": p})
json.dump({"rows": rows}, open(sys.argv[1] if len(sys.argv) > 1 else f"{S}/paper_refresh/reference_rows.json", "w"), indent=1)
for r in rows:
    print(f"{r['kind']:8s} {r['name']:28s} raw={r['raw']:.3f} len_adj={r['len_adj']:.3f}")
