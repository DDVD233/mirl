"""
Post-hoc reasoning metric computation from reasoning_eval.py output JSONL.

Metrics computed:
  Consistency:
    self_consistency_rate    — fraction of samples where all stochastic answers agree
    para_consistency_rate    — fraction where reasoning_answer == para_reasoning_answer

  Faithfulness:
    direct_accuracy          — accuracy of greedy direct predictions vs ground truth
    reasoning_accuracy       — accuracy of greedy reasoning predictions
    accuracy_delta           — reasoning_accuracy - direct_accuracy
    flip_to_correct          — direct wrong, reasoning correct
    flip_to_wrong            — direct correct, reasoning wrong
    kl_direct_vs_reasoning_mean — mean KL(P_reasoning || P_direct) over samples with captured probs

  Concision:
    mean_reasoning_tokens    — average length of <think>...</think> trace
    median_reasoning_tokens
    std_reasoning_tokens
    efficiency_score         — reasoning_accuracy / mean_reasoning_tokens * 1000

Usage:
    python compute_reasoning_metrics.py \
        --input_jsonl  /results/reasoning_eval/dreaddit_reasoning.jsonl \
        --model_name   "Qwen/Qwen2.5-Omni-7B" \
        --wandb_project reasoning-evaluation \
        --wandb_run_id  my-run-id
"""

import argparse
import json
import math
import os
import re

import numpy as np


# ── Helpers ────────────────────────────────────────────────────────────────────

def _norm(s: str | None) -> str:
    return (s or "").strip().lower()


def _kl_divergence(p: dict, q: dict) -> float:
    """KL(P || Q) in nats. Adds epsilon for numerical stability."""
    eps = 1e-10
    labels = sorted(set(p.keys()) | set(q.keys()))
    result = 0.0
    for lbl in labels:
        pi = p.get(lbl, 0.0) + eps
        qi = q.get(lbl, 0.0) + eps
        result += pi * math.log(pi / qi)
    return result


# ── Metric computation ─────────────────────────────────────────────────────────

def compute_consistency_metrics(results: list[dict]) -> dict:
    self_agree = 0
    para_agree = 0
    n_self = 0
    n_para = 0

    for r in results:
        # Self-consistency: all stochastic answers identical
        stoc = [_norm(a) for a in r.get("stochastic_answers", []) if a is not None]
        if stoc:
            n_self += 1
            if len(set(stoc)) == 1:
                self_agree += 1

        # Para-consistency: reasoning == para-reasoning
        ra  = _norm(r.get("reasoning_answer"))
        pra = _norm(r.get("para_reasoning_answer"))
        if ra and pra:
            n_para += 1
            if ra == pra:
                para_agree += 1

    return {
        "self_consistency_rate": self_agree / n_self if n_self else None,
        "para_consistency_rate": para_agree / n_para if n_para else None,
        "n_self_consistency":    n_self,
        "n_para_consistency":    n_para,
    }


def compute_faithfulness_metrics(results: list[dict]) -> dict:
    gts, directs, reasonings = [], [], []
    for r in results:
        gt = _norm(r.get("answer"))
        da = _norm(r.get("direct_answer"))
        ra = _norm(r.get("reasoning_answer"))
        if gt and da and ra:
            gts.append(gt)
            directs.append(da)
            reasonings.append(ra)

    n = len(gts)
    if n == 0:
        return {k: None for k in [
            "direct_accuracy", "reasoning_accuracy", "accuracy_delta",
            "flip_to_correct", "flip_to_wrong", "kl_direct_vs_reasoning_mean",
            "n_faithfulness"
        ]}

    direct_correct    = [d == g for d, g in zip(directs, gts)]
    reasoning_correct = [r == g for r, g in zip(reasonings, gts)]

    direct_acc    = sum(direct_correct) / n
    reasoning_acc = sum(reasoning_correct) / n

    flip_to_correct = sum(
        1 for dc, rc in zip(direct_correct, reasoning_correct) if not dc and rc
    ) / n
    flip_to_wrong = sum(
        1 for dc, rc in zip(direct_correct, reasoning_correct) if dc and not rc
    ) / n

    # KL divergence over label distributions
    kl_vals = []
    for r in results:
        p = r.get("reasoning_label_probs")
        q = r.get("direct_label_probs")
        if p and q:
            try:
                kl_vals.append(_kl_divergence(p, q))
            except Exception:
                pass

    return {
        "direct_accuracy":               direct_acc,
        "reasoning_accuracy":            reasoning_acc,
        "accuracy_delta":                reasoning_acc - direct_acc,
        "flip_to_correct":               flip_to_correct,
        "flip_to_wrong":                 flip_to_wrong,
        "kl_direct_vs_reasoning_mean":   float(np.mean(kl_vals)) if kl_vals else None,
        "kl_n_valid":                    len(kl_vals),
        "n_faithfulness":                n,
    }


def compute_concision_metrics(results: list[dict], reasoning_accuracy: float | None) -> dict:
    token_counts = [
        r["reasoning_trace_tokens"]
        for r in results
        if r.get("reasoning_trace_tokens", 0) > 0
    ]
    if not token_counts:
        return {
            "mean_reasoning_tokens":   None,
            "median_reasoning_tokens": None,
            "std_reasoning_tokens":    None,
            "efficiency_score":        None,
            "n_traces":                0,
        }

    arr = np.array(token_counts, dtype=float)
    mean_tok   = float(np.mean(arr))
    median_tok = float(np.median(arr))
    std_tok    = float(np.std(arr))

    efficiency = (
        reasoning_accuracy / mean_tok * 1000
        if reasoning_accuracy is not None and mean_tok > 0
        else None
    )

    return {
        "mean_reasoning_tokens":   mean_tok,
        "median_reasoning_tokens": median_tok,
        "std_reasoning_tokens":    std_tok,
        "efficiency_score":        efficiency,
        "n_traces":                len(token_counts),
    }


# ── W&B logging (mirrors inference.py _log_wandb) ─────────────────────────────

def _log_wandb(args, dataset_name: str, metrics: dict) -> None:
    if not getattr(args, "wandb_project", None):
        return
    try:
        import wandb
        run_id   = getattr(args, "wandb_run_id",   None) or None
        run_name = getattr(args, "wandb_run_name", None) or run_id
        entity   = getattr(args, "wandb_entity",   None) or None
        model_name = getattr(args, "model_name", "unknown")
        wandb.init(
            project=args.wandb_project,
            entity=entity,
            id=run_id,
            resume="allow",
            name=run_name,
            config={"model": model_name},
        )
        # Only log numeric values; skip None and metadata counters
        loggable = {
            f"{dataset_name}/{k}": v
            for k, v in metrics.items()
            if v is not None and isinstance(v, (int, float))
        }
        wandb.log(loggable)
        wandb.finish()
        print(f"W&B: logged {len(loggable)} metrics to project '{args.wandb_project}'")
    except Exception as exc:
        print(f"[WARN] W&B logging failed: {exc}")


# ── Summary printer ────────────────────────────────────────────────────────────

def _fmt(v) -> str:
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def print_summary(dataset_name: str, model_name: str, n: int, all_metrics: dict) -> None:
    print(f"\n{'='*60}")
    print(f"  Dataset  : {dataset_name}")
    print(f"  Model    : {model_name}")
    print(f"  N        : {n}")
    print(f"  {'─'*54}")
    print(f"  CONSISTENCY")
    print(f"    self_consistency_rate    : {_fmt(all_metrics.get('self_consistency_rate'))}")
    print(f"    para_consistency_rate    : {_fmt(all_metrics.get('para_consistency_rate'))}")
    print(f"  FAITHFULNESS")
    print(f"    direct_accuracy          : {_fmt(all_metrics.get('direct_accuracy'))}")
    print(f"    reasoning_accuracy       : {_fmt(all_metrics.get('reasoning_accuracy'))}")
    print(f"    accuracy_delta           : {_fmt(all_metrics.get('accuracy_delta'))}")
    print(f"    flip_to_correct          : {_fmt(all_metrics.get('flip_to_correct'))}")
    print(f"    flip_to_wrong            : {_fmt(all_metrics.get('flip_to_wrong'))}")
    print(f"    kl_direct_vs_reasoning   : {_fmt(all_metrics.get('kl_direct_vs_reasoning_mean'))}")
    print(f"  CONCISION")
    print(f"    mean_reasoning_tokens    : {_fmt(all_metrics.get('mean_reasoning_tokens'))}")
    print(f"    median_reasoning_tokens  : {_fmt(all_metrics.get('median_reasoning_tokens'))}")
    print(f"    std_reasoning_tokens     : {_fmt(all_metrics.get('std_reasoning_tokens'))}")
    print(f"    efficiency_score         : {_fmt(all_metrics.get('efficiency_score'))}")
    print(f"{'='*60}\n")


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        results = [json.loads(ln) for ln in f if ln.strip()]

    if not results:
        print("Empty input JSONL — nothing to compute.")
        return

    dataset_name = results[0].get("dataset", "unknown")

    consistency  = compute_consistency_metrics(results)
    faithfulness = compute_faithfulness_metrics(results)
    concision    = compute_concision_metrics(
        results, faithfulness.get("reasoning_accuracy")
    )
    all_metrics = {**consistency, **faithfulness, **concision}

    print_summary(dataset_name, args.model_name, len(results), all_metrics)

    # Save JSON metrics file alongside the input JSONL
    metrics_path = re.sub(r"\.jsonl$", "_reasoning_metrics.json", args.input_jsonl)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {"model": args.model_name, "dataset": dataset_name, "n_samples": len(results),
             **all_metrics},
            f, indent=2,
        )
    print(f"Metrics → {metrics_path}")

    _log_wandb(args, dataset_name, all_metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute reasoning metrics from reasoning_eval.py output")
    parser.add_argument("--input_jsonl",    required=True, help="Output JSONL from reasoning_eval.py")
    parser.add_argument("--model_name",     default="unknown", help="Model label for output JSON")
    parser.add_argument("--wandb_project",  default=None)
    parser.add_argument("--wandb_run_id",   default=None)
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--wandb_entity",   default=None)
    args = parser.parse_args()
    main(args)
