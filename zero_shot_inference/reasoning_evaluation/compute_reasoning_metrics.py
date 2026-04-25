"""
Post-hoc reasoning metric computation from reasoning_eval.py output JSONL.

Metrics computed:
  Consistency:
    self_consistency_rate    — fraction of samples where all stochastic answers agree
    self_consistency_correct — fraction where all stochastic answers agree AND match ground truth
    self_consistency_incorrect — fraction where all stochastic answers agree AND don't match ground truth
    para_consistency_rate    — fraction where reasoning_answer == para_reasoning_answer
    para_consistency_correct — fraction where reasoning/para agree AND match ground truth
    para_consistency_incorrect — fraction where reasoning/para agree AND don't match ground truth
    direct/reasoning/para accuracy and weighted_f1 — inference-style task metrics

  Faithfulness:
    direct_accuracy          — accuracy of greedy direct predictions vs ground truth
    reasoning_accuracy       — accuracy of greedy reasoning predictions
    accuracy_delta           — reasoning_accuracy - direct_accuracy
    flip_to_correct          — direct wrong, reasoning correct
    flip_to_wrong            — direct correct, reasoning wrong
    kl_direct_vs_reasoning_mean — mean KL(P_reasoning || P_direct) over samples with captured probs

  Concision (overall):
    mean_reasoning_tokens    — average length of <think>...</think> trace
    median_reasoning_tokens
    std_reasoning_tokens
    efficiency_score         — reasoning_accuracy / mean_reasoning_tokens * 1000
    n_traces

  Concision (stratified by correctness):
    mean_reasoning_tokens_correct / _incorrect
    median_reasoning_tokens_correct / _incorrect
    std_reasoning_tokens_correct / _incorrect
    n_traces_correct / _incorrect

Usage:
    # Single dataset
    python compute_reasoning_metrics.py \
        --input_jsonl  /results/reasoning_eval/dreaddit_reasoning.jsonl \
        --model_name   "Qwen/Qwen2.5-Omni-7B"

    # Multiple datasets → per-dataset + overall summary
    python compute_reasoning_metrics.py \
        --input_jsonl  /results/dreaddit_reasoning.jsonl /results/eatd_reasoning.jsonl \
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
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer


# ── Helpers ────────────────────────────────────────────────────────────────────

def _norm(s: str | None) -> str:
    return (s or "").strip().lower()


def _parse_multilabel(text: str | None) -> list[str]:
    return sorted(lbl.strip().lower() for lbl in (text or "").split(",") if lbl.strip())


def _auto_multilabel(results: list[dict]) -> bool:
    return any("," in str(r.get("answer", "")) for r in results)


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


def _token_stats(token_counts: list[int | float], suffix: str) -> dict:
    """Compute mean/median/std/n for a list of token counts, keyed with suffix."""
    if not token_counts:
        return {
            f"mean_reasoning_tokens{suffix}":   None,
            f"median_reasoning_tokens{suffix}": None,
            f"std_reasoning_tokens{suffix}":    None,
            f"n_traces{suffix}":                0,
        }
    arr = np.array(token_counts, dtype=float)
    return {
        f"mean_reasoning_tokens{suffix}":   float(np.mean(arr)),
        f"median_reasoning_tokens{suffix}": float(np.median(arr)),
        f"std_reasoning_tokens{suffix}":    float(np.std(arr)),
        f"n_traces{suffix}":                len(token_counts),
    }


# ── Metric computation ─────────────────────────────────────────────────────────

def compute_consistency_metrics(results: list[dict]) -> dict:
    self_agree = 0
    self_agree_correct = 0
    self_agree_incorrect = 0
    para_agree = 0
    para_agree_correct = 0
    para_agree_incorrect = 0
    para_correct = 0
    n_self = 0
    n_para = 0
    n_para_accuracy = 0

    for r in results:
        gt = _norm(r.get("answer"))

        # Self-consistency: all stochastic answers identical
        stoc = [_norm(a) for a in r.get("stochastic_answers", []) if a is not None]
        if stoc:
            n_self += 1
            if len(set(stoc)) == 1:
                self_agree += 1
                if gt:
                    if stoc[0] == gt:
                        self_agree_correct += 1
                    else:
                        self_agree_incorrect += 1

        # Para-consistency: reasoning == para-reasoning
        ra  = _norm(r.get("reasoning_answer"))
        pra = _norm(r.get("para_reasoning_answer"))
        if gt and pra:
            n_para_accuracy += 1
            if pra == gt:
                para_correct += 1

        if ra and pra:
            n_para += 1
            if ra == pra:
                para_agree += 1
                if gt:
                    if pra == gt:
                        para_agree_correct += 1
                    else:
                        para_agree_incorrect += 1

    return {
        "self_consistency_rate":      self_agree          / n_self if n_self else None,
        "self_consistency_correct":   self_agree_correct  / n_self if n_self else None,
        "self_consistency_incorrect": self_agree_incorrect / n_self if n_self else None,
        "para_consistency_rate":      para_agree          / n_para if n_para else None,
        "para_consistency_correct":   para_agree_correct  / n_para if n_para else None,
        "para_consistency_incorrect": para_agree_incorrect / n_para if n_para else None,
        "para_accuracy":              para_correct / n_para_accuracy if n_para_accuracy else None,
        "n_self_consistency":             n_self,
        "n_self_consistency_correct":     self_agree_correct,
        "n_self_consistency_incorrect":   self_agree_incorrect,
        "n_para_consistency":             n_para,
        "n_para_consistency_correct":     para_agree_correct,
        "n_para_consistency_incorrect":   para_agree_incorrect,
        "n_para_accuracy":                n_para_accuracy,
        "n_para_correct":                 para_correct,
        "n_para_incorrect":               n_para_accuracy - para_correct,
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


def compute_prediction_metrics(
    results: list[dict],
    prediction_key: str,
    prefix: str,
    multilabel: bool = False,
) -> dict:
    """Compute inference-style accuracy and weighted F1 for one answer field."""
    valid = [
        r for r in results
        if _norm(r.get("answer")) and _norm(r.get(prediction_key))
    ]
    n_key = f"n_{prefix}_samples"
    if not valid:
        return {
            f"{prefix}_accuracy": None,
            f"{prefix}_weighted_f1": None,
            n_key: 0,
        }

    if multilabel:
        gt_parsed = [_parse_multilabel(r.get("answer")) for r in valid]
        pred_parsed = [_parse_multilabel(r.get(prediction_key)) for r in valid]
        mlb = MultiLabelBinarizer()
        mlb.fit(gt_parsed + pred_parsed)
        gt_bin = mlb.transform(gt_parsed)
        pred_bin = mlb.transform(pred_parsed)
        exact_match = float(accuracy_score(gt_bin, pred_bin))
        weighted_f1 = float(f1_score(gt_bin, pred_bin, average="weighted", zero_division=0))
        return {
            f"{prefix}_accuracy": exact_match,
            f"{prefix}_exact_match_accuracy": exact_match,
            f"{prefix}_weighted_f1": weighted_f1,
            n_key: len(valid),
        }

    gts = [_norm(r.get("answer")) for r in valid]
    preds = [_norm(r.get(prediction_key)) for r in valid]
    return {
        f"{prefix}_accuracy": float(accuracy_score(gts, preds)),
        f"{prefix}_weighted_f1": float(f1_score(gts, preds, average="weighted", zero_division=0)),
        n_key: len(valid),
    }


def compute_inference_style_metrics(results: list[dict], multilabel: bool = False) -> dict:
    """Compute no-extra-inference task metrics from saved reasoning-eval answers."""
    metrics = {}
    for prediction_key, prefix in [
        ("direct_answer", "direct"),
        ("reasoning_answer", "reasoning"),
        ("para_reasoning_answer", "para"),
    ]:
        metrics.update(compute_prediction_metrics(results, prediction_key, prefix, multilabel))
    return metrics


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


def compute_concision_metrics_by_correctness(results: list[dict]) -> dict:
    """Token-length statistics split by whether the reasoning prediction was correct."""
    correct_tokens: list[int] = []
    incorrect_tokens: list[int] = []

    for r in results:
        gt     = _norm(r.get("answer"))
        ra     = _norm(r.get("reasoning_answer"))
        tokens = r.get("reasoning_trace_tokens", 0)
        if not (gt and ra and tokens > 0):
            continue
        if ra == gt:
            correct_tokens.append(tokens)
        else:
            incorrect_tokens.append(tokens)

    return {
        **_token_stats(correct_tokens,   "_correct"),
        **_token_stats(incorrect_tokens, "_incorrect"),
    }


# ── W&B logging (mirrors inference.py _log_wandb) ─────────────────────────────

def _wandb_loggable(dataset_name: str, metrics: dict) -> dict:
    """Return a flat dict of numeric metrics prefixed by dataset_name/, for wandb.log."""
    return {
        f"{dataset_name}/{k}": v
        for k, v in metrics.items()
        if v is not None and isinstance(v, (int, float))
    }


def _log_wandb_all(args, named_metrics: list[tuple[str, dict]]) -> None:
    """Init one W&B run, log all (dataset_name, metrics) pairs, then finish."""
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
        loggable = {}
        for dataset_name, metrics in named_metrics:
            loggable.update(_wandb_loggable(dataset_name, metrics))
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
    print(f"    self_consistency_correct : {_fmt(all_metrics.get('self_consistency_correct'))}")
    print(f"    self_consistency_incorrect:{_fmt(all_metrics.get('self_consistency_incorrect'))}")
    print(f"    para_consistency_rate    : {_fmt(all_metrics.get('para_consistency_rate'))}")
    print(f"    para_consistency_correct : {_fmt(all_metrics.get('para_consistency_correct'))}")
    print(f"    para_consistency_incorrect:{_fmt(all_metrics.get('para_consistency_incorrect'))}")
    print(f"  FAITHFULNESS")
    print(f"    direct_accuracy          : {_fmt(all_metrics.get('direct_accuracy'))}")
    print(f"    direct_weighted_f1       : {_fmt(all_metrics.get('direct_weighted_f1'))}")
    print(f"    reasoning_accuracy       : {_fmt(all_metrics.get('reasoning_accuracy'))}")
    print(f"    reasoning_weighted_f1    : {_fmt(all_metrics.get('reasoning_weighted_f1'))}")
    print(f"    para_accuracy            : {_fmt(all_metrics.get('para_accuracy'))}")
    print(f"    para_weighted_f1         : {_fmt(all_metrics.get('para_weighted_f1'))}")
    print(f"    accuracy_delta           : {_fmt(all_metrics.get('accuracy_delta'))}")
    print(f"    flip_to_correct          : {_fmt(all_metrics.get('flip_to_correct'))}")
    print(f"    flip_to_wrong            : {_fmt(all_metrics.get('flip_to_wrong'))}")
    print(f"    kl_direct_vs_reasoning   : {_fmt(all_metrics.get('kl_direct_vs_reasoning_mean'))}")
    print(f"  CONCISION (overall)")
    print(f"    mean_reasoning_tokens    : {_fmt(all_metrics.get('mean_reasoning_tokens'))}")
    print(f"    median_reasoning_tokens  : {_fmt(all_metrics.get('median_reasoning_tokens'))}")
    print(f"    std_reasoning_tokens     : {_fmt(all_metrics.get('std_reasoning_tokens'))}")
    print(f"    efficiency_score         : {_fmt(all_metrics.get('efficiency_score'))}")
    print(f"  CONCISION (correct samples, n={all_metrics.get('n_traces_correct', 0)})")
    print(f"    mean_reasoning_tokens    : {_fmt(all_metrics.get('mean_reasoning_tokens_correct'))}")
    print(f"    median_reasoning_tokens  : {_fmt(all_metrics.get('median_reasoning_tokens_correct'))}")
    print(f"    std_reasoning_tokens     : {_fmt(all_metrics.get('std_reasoning_tokens_correct'))}")
    print(f"  CONCISION (incorrect samples, n={all_metrics.get('n_traces_incorrect', 0)})")
    print(f"    mean_reasoning_tokens    : {_fmt(all_metrics.get('mean_reasoning_tokens_incorrect'))}")
    print(f"    median_reasoning_tokens  : {_fmt(all_metrics.get('median_reasoning_tokens_incorrect'))}")
    print(f"    std_reasoning_tokens     : {_fmt(all_metrics.get('std_reasoning_tokens_incorrect'))}")
    print(f"{'='*60}\n")


# ── Per-file processing ────────────────────────────────────────────────────────

def process_file(jsonl_path: str, args: argparse.Namespace) -> tuple[str, list[dict], dict]:
    """Load one JSONL, compute all metrics, print summary, save JSON. Returns (dataset_name, results, metrics)."""
    with open(jsonl_path, "r", encoding="utf-8") as f:
        results = [json.loads(ln) for ln in f if ln.strip()]

    if not results:
        print(f"Empty input JSONL — nothing to compute: {jsonl_path}")
        return "unknown", [], {}

    dataset_name = results[0].get("dataset", "unknown")
    multilabel = args.multilabel or _auto_multilabel(results)

    consistency  = compute_consistency_metrics(results)
    faithfulness = compute_faithfulness_metrics(results)
    inference_style = compute_inference_style_metrics(results, multilabel=multilabel)
    concision    = compute_concision_metrics(
        results,
        inference_style.get("reasoning_accuracy", faithfulness.get("reasoning_accuracy")),
    )
    concision_by_correct = compute_concision_metrics_by_correctness(results)
    all_metrics  = {**consistency, **faithfulness, **inference_style, **concision, **concision_by_correct}

    print_summary(dataset_name, args.model_name, len(results), all_metrics)

    metrics_path = re.sub(r"\.jsonl$", "_reasoning_metrics.json", jsonl_path)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {"model": args.model_name, "dataset": dataset_name, "n_samples": len(results),
             **all_metrics},
            f, indent=2,
        )
    print(f"Metrics → {metrics_path}")

    return dataset_name, results, all_metrics


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    all_results: list[dict] = []
    wandb_pairs: list[tuple[str, dict]] = []

    for jsonl_path in args.input_jsonl:
        dataset_name, results, metrics = process_file(jsonl_path, args)
        all_results.extend(results)
        if results:
            wandb_pairs.append((dataset_name, metrics))

    # Overall summary across all datasets when more than one file is provided
    if len(args.input_jsonl) > 1 and all_results:
        print("\n" + "#" * 60)
        print("  OVERALL (all datasets combined)")
        print("#" * 60)

        consistency  = compute_consistency_metrics(all_results)
        faithfulness = compute_faithfulness_metrics(all_results)
        multilabel = args.multilabel or _auto_multilabel(all_results)
        inference_style = compute_inference_style_metrics(all_results, multilabel=multilabel)
        concision    = compute_concision_metrics(
            all_results,
            inference_style.get("reasoning_accuracy", faithfulness.get("reasoning_accuracy")),
        )
        concision_by_correct = compute_concision_metrics_by_correctness(all_results)
        overall_metrics = {**consistency, **faithfulness, **inference_style, **concision, **concision_by_correct}

        print_summary("overall", args.model_name, len(all_results), overall_metrics)

        # Save overall JSON alongside the first input file
        first_dir = os.path.dirname(os.path.abspath(args.input_jsonl[0]))
        model_slug = args.model_name.replace("/", "_")
        overall_path = os.path.join(first_dir, f"{model_slug}_overall_reasoning_metrics.json")
        with open(overall_path, "w", encoding="utf-8") as f:
            json.dump(
                {"model": args.model_name, "dataset": "overall",
                 "n_samples": len(all_results), "n_datasets": len(args.input_jsonl),
                 **overall_metrics},
                f, indent=2,
            )
        print(f"Overall metrics → {overall_path}")

        wandb_pairs.append(("overall", overall_metrics))

    # Single wandb.init / wandb.finish for the entire run so all datasets
    # (including "overall") land on the same run with no finish/resume races.
    _log_wandb_all(args, wandb_pairs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute reasoning metrics from reasoning_eval.py output")
    parser.add_argument("--input_jsonl",    required=True, nargs="+",
                        help="Output JSONL file(s) from reasoning_eval.py. Pass multiple to also emit overall metrics.")
    parser.add_argument("--model_name",     default="unknown", help="Model label for output JSON")
    parser.add_argument("--wandb_project",  default=None)
    parser.add_argument("--wandb_run_id",   default=None)
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--wandb_entity",   default=None)
    parser.add_argument("--multilabel", action="store_true",
                        help="Force comma-separated multilabel metrics. Defaults to auto-detecting comma-separated gold labels.")
    args = parser.parse_args()
    main(args)
