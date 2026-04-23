"""
Compute per-class weighted accuracy and mean WA for IEMOCAP zero-shot predictions.

WA_c = 0.5 * (TP/P + TN/N)
Mean WA = mean of WA_c over all classes

Usage:
    python compute_iemocap_emotion_acc.py \
        --predictions_jsonl /path/to/model_iemocap_merged.jsonl \
        --model keentomato/harpo_hier_step400 \
        [--wandb_project zero-shot-inference --wandb_run_id <run_id>]
"""

import argparse
import json
import re
from typing import Dict, List, Optional

import numpy as np

LABEL_TO_INDEX: Dict[str, int] = {
    "anger": 0,
    "happiness": 1,
    "neutral state": 2,
    "sadness": 3,
}
INDEX_TO_LABEL: Dict[int, str] = {v: k for k, v in LABEL_TO_INDEX.items()}


def _safe_div(a: float, b: float) -> float:
    return a / b if b > 0 else 0.0


def compute_emotion_weighted_accuracies(
    y_pred: List[int],
    y_true: List[int],
    index_to_label: Optional[Dict[int, str]] = None,
) -> Dict[str, object]:
    """
    For each class c:
      WA_c = 0.5 * (TP/P + TN/N)
    Returns:
      {
        "mean_weighted_accuracy": float,
        "weighted_accuracy_per_class": [
           {"label_index": int, "class_name": str, "weighted_accuracy": float}
           ...
        ]
      }
    """
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)

    classes = sorted(set(y_true.tolist()))

    if len(y_true) == 0 or len(classes) == 0:
        return {"mean_weighted_accuracy": 0.0, "weighted_accuracy_per_class": []}

    wa_per_class = []
    accs = []
    for c in classes:
        P = int((y_true == c).sum())
        N = int((y_true != c).sum())
        TP = int(((y_true == c) & (y_pred == c)).sum())
        TN = int(((y_true != c) & (y_pred != c)).sum())
        wa_c = 0.5 * (_safe_div(TP, P) + _safe_div(TN, N))
        class_name = index_to_label.get(int(c), str(c)) if index_to_label else str(c)
        wa_per_class.append({
            "label_index": int(c),
            "class_name": class_name,
            "weighted_accuracy": float(wa_c),
        })
        accs.append(wa_c)

    mean_wa = float(np.mean(accs)) if accs else 0.0
    return {"mean_weighted_accuracy": mean_wa, "weighted_accuracy_per_class": wa_per_class}


def _log_wandb(args, metrics: dict):
    if not getattr(args, "wandb_project", None):
        return
    try:
        import wandb
        run_id   = getattr(args, "wandb_run_id",   None) or None
        run_name = getattr(args, "wandb_run_name", None) or run_id
        entity   = getattr(args, "wandb_entity",   None) or None
        wandb.init(
            project=args.wandb_project,
            entity=entity,
            id=run_id,
            resume="allow",
            name=run_name,
            config={"model": args.model},
        )
        wandb.log(metrics)
        wandb.finish()
    except Exception as exc:
        print(f"[WARN] W&B logging failed: {exc}")


def main():
    parser = argparse.ArgumentParser(description="Compute IEMOCAP emotion mean accuracy")
    parser.add_argument("--predictions_jsonl", required=True,
                        help="Path to merged predictions JSONL")
    parser.add_argument("--model", required=True,
                        help="Model name (slashes OK; converted to underscores for key lookup)")
    parser.add_argument("--dataset_filter", default="iemocap",
                        help="Only process rows where dataset == this value. Set to '' to skip.")
    parser.add_argument("--wandb_project", default="")
    parser.add_argument("--wandb_run_id", default="")
    parser.add_argument("--wandb_run_name", default="")
    parser.add_argument("--wandb_entity", default="")
    args = parser.parse_args()

    model_key = f"predicted_answer_{args.model.replace('/', '_')}"

    with open(args.predictions_jsonl, encoding="utf-8") as f:
        rows = [json.loads(ln) for ln in f if ln.strip()]

    if args.dataset_filter:
        rows = [r for r in rows if r.get("dataset") == args.dataset_filter]

    y_true, y_pred = [], []
    n_skipped = 0
    for row in rows:
        gt_raw  = row.get("answer", "").strip().lower()
        pred_raw = row.get(model_key, "").strip().lower()

        gt_idx = LABEL_TO_INDEX.get(gt_raw)
        if gt_idx is None:
            n_skipped += 1
            continue

        pred_idx = LABEL_TO_INDEX.get(pred_raw, -1)  # -1 = unknown → counts as wrong

        y_true.append(gt_idx)
        y_pred.append(pred_idx)

    if not y_true:
        print("No valid samples found — check --predictions_jsonl and --model.")
        return

    results = compute_emotion_weighted_accuracies(y_pred, y_true, INDEX_TO_LABEL)

    print(f"\n{'='*60}")
    print(f"  Dataset  : iemocap")
    print(f"  Model    : {args.model}")
    print(f"  N samples: {len(y_true)}  (skipped: {n_skipped})")
    print(f"  Mean WA  : {results['mean_weighted_accuracy']:.4f}")
    print(f"{'─'*60}")
    for entry in results["weighted_accuracy_per_class"]:
        print(f"  {entry['class_name']:>16s}  WA = {entry['weighted_accuracy']:.4f}")
    print(f"{'='*60}\n")

    output = {
        "model": args.model,
        "dataset": "iemocap",
        "n_samples": len(y_true),
        "n_skipped": n_skipped,
        **results,
    }

    out_path = re.sub(r"\.jsonl$", "", args.predictions_jsonl) + "_iemocap_emotion_acc.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"Saved → {out_path}")

    wandb_metrics = {
        "iemocap/mean_weighted_accuracy": results["mean_weighted_accuracy"],
    }
    for entry in results["weighted_accuracy_per_class"]:
        key = entry["class_name"].replace(" ", "_")
        wandb_metrics[f"iemocap/wa_{key}"] = entry["weighted_accuracy"]

    _log_wandb(args, wandb_metrics)


if __name__ == "__main__":
    main()
