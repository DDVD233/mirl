import datetime
import json
import os
from collections import defaultdict
from typing import Dict, List, Set

def extract_boxed_content(text: str) -> str:
    import re
    boxed_match = re.search(r"\\boxed{([^}]*)}", text)
    if boxed_match:
        return boxed_match.group(1)
    markdown_match = re.search(r"\[(.*?)\]", text)
    if markdown_match:
        return markdown_match.group(1)
    return text

def _safe_div(num: float, den: float) -> float:
    return num / den if den > 0 else 0.0

# -----------------------
# combine per-class counts + metrics (single-label multiclass)
# -----------------------
def compute_class_counts_and_metrics(
    predictions: List[str],
    ground_truths: List[str],
) -> Dict[str, object]:
    """
    Single-label multiclass:
      1) Build per-class one-vs-rest TP/FP/FN and support (count).
      2) Compute per-class metrics.
         - Per-class 'accuracy' is defined as TP/support (i.e., recall).

    Returns dict with:
      - class_metrics: {label: {precision, recall, f1, accuracy, count, confusion_matrix}}
      - pooled_counts: {'tp','fp','fn'}   # no TN
      - active_classes: int
      - total_support: int
    """
    assert len(predictions) == len(ground_truths)

    y_pred = [extract_boxed_content(p).strip() for p in predictions]
    y_true = [extract_boxed_content(t).strip() for t in ground_truths]
    labels = sorted(set(y_true) | set(y_pred))

    N = len(y_true)
    # Per-class counts
    matrices = {c: {"tp": 0, "fp": 0, "fn": 0, "count": 0} for c in labels}

    for p, t in zip(y_pred, y_true):
        for c in labels:
            if t == c:
                matrices[c]["count"] += 1
            if p == c and t == c:
                matrices[c]["tp"] += 1
            elif p == c and t != c:
                matrices[c]["fp"] += 1
            elif p != c and t == c:
                matrices[c]["fn"] += 1

    # pooled counts for micro
    pooled_tp = sum(m["tp"] for m in matrices.values())
    pooled_fp = sum(m["fp"] for m in matrices.values())
    pooled_fn = sum(m["fn"] for m in matrices.values())

    # per-class metrics
    class_metrics: Dict[str, Dict[str, float]] = {}
    active_classes = 0
    total_support = 0

    for c, m in matrices.items():
        tp, fp, fn = m["tp"], m["fp"], m["fn"]
        support = m["count"]

        precision = _safe_div(tp, (tp + fp))
        recall    = _safe_div(tp, (tp + fn))
        f1        = _safe_div(2 * precision * recall, (precision + recall))
        accuracy  = recall  # per-class accuracy = TP / support (over the total amount of true instances of the class)

        class_metrics[c] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "count": support,
            "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn},
        }

        if support > 0:
            active_classes += 1
            total_support += support

    return {
        "class_metrics": class_metrics,
        "pooled_counts": {"tp": pooled_tp, "fp": pooled_fp, "fn": pooled_fn},
        "active_classes": active_classes,
        "total_support": total_support,
    }

def compute_dataset_metrics(predictions: List[str], ground_truths: List[str]) -> Dict[str, Dict]:
    """
    Compute per-class metrics and dataset-level macro/micro/weighted metrics with clear prefixes.
    Accuracy is corrected for single-label multiclass:
      - micro_accuracy = #correct / N
      - per-class 'accuracy' = TP/support (== recall)
    """
    summary = compute_class_counts_and_metrics(predictions, ground_truths)
    class_metrics = summary["class_metrics"]
    pooled = summary["pooled_counts"]
    active_classes = summary["active_classes"]
    total_support = summary["total_support"]  # == N

    # Accumulate macro & weighted
    keys = ["precision", "recall", "f1", "accuracy"]

    macro_sum = {k: 0.0 for k in keys}
    weighted_sum = {k: 0.0 for k in keys}

    for c, cm in class_metrics.items():
        support = cm["count"]
        if support > 0:
            for k in keys:
                macro_sum[k] += cm[k]
                weighted_sum[k] += cm[k] * support

    # Macro
    macro = {
        f"macro_{k}": _safe_div(macro_sum[k], active_classes) if active_classes > 0 else 0.0
        for k in keys
    }

    # Weighted
    weighted = {
        f"weighted_{k}": _safe_div(weighted_sum[k], total_support) if total_support > 0 else 0.0
        for k in keys
    }

    # Micro (pooled). In single-label multiclass:
    # micro_precision = micro_recall = micro_f1 = accuracy = pooled_tp / N
    PTP, PFP, PFN = pooled["tp"], pooled["fp"], pooled["fn"]
    micro_precision = _safe_div(PTP, (PTP + PFP))
    micro_recall    = _safe_div(PTP, (PTP + PFN))
    micro_f1        = _safe_div(2 * micro_precision * micro_recall, (micro_precision + micro_recall))
    micro_accuracy  = _safe_div(PTP, total_support)

    micro = {
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "micro_accuracy": micro_accuracy,
    }

    dataset_metrics = {}
    dataset_metrics.update(macro)
    dataset_metrics.update(weighted)
    dataset_metrics.update(micro)

    return {
        "class_metrics": class_metrics,
        "dataset_metrics": dataset_metrics,
        "active_classes": active_classes,
    }

def compute_metrics_by_data_source(
    predictions: List[str],
    ground_truths: List[str],
    datasets: List[str],
    save_path: str = None,
    global_steps: int = None,
) -> Dict[str, float]:
    """
    Compute metrics at the dataset level and a global mean across datasets (no data sources).
    Aggregates the prefixed dataset-level metrics produced by compute_dataset_metrics.
    """

    with open(
        os.path.join(save_path, f"val_generations_{global_steps}.json"), "w"
    ) as f:
        json.dump(
            {
                "predictions": predictions,
                "ground_truths": ground_truths,
                "datasets": datasets,
            },
            f,
            indent=4,
        )

    # Group by dataset
    grouped = defaultdict(lambda: {"preds": [], "gts": []})
    for p, t, d in zip(predictions, ground_truths, datasets):
        grouped[d]["preds"].append(p)
        grouped[d]["gts"].append(t)

    result: Dict[str, float] = {}
    discovered_metric_keys: List[str] = []

    global_accum = None
    n_datasets = 0

    for dataset_name, data in grouped.items():
        ds_res = compute_dataset_metrics(data["preds"], data["gts"])
        ds_metrics = ds_res["dataset_metrics"]

        if not discovered_metric_keys:
            discovered_metric_keys = sorted(ds_metrics.keys())

        for k in discovered_metric_keys:
            result[f"{dataset_name}/{k}"] = ds_metrics.get(k, 0.0)

        if ds_res["active_classes"] == 0:
            continue

        if global_accum is None:
            global_accum = {k: 0.0 for k in discovered_metric_keys}
        for k in discovered_metric_keys:
            global_accum[k] += ds_metrics.get(k, 0.0)
        n_datasets += 1

    if n_datasets > 0:
        for k in discovered_metric_keys:
            result[f"val/averaged_{k}"] = global_accum[k] / n_datasets

    return result

if __name__ == "__main__":
    predictions   = [
        # DatasetA (6 samples)
        "<think>Well, looking at the image, the person's eyes are looking down and their mouth is in a sort of frown. There's no big smile or anything that would suggest happiness. It doesn't look like they're angry or fearful either. So, I'd say the primary facial expression is sad.</think> \\boxed{sad}", "A", "B", "B", "A", "B",
        # DatasetB (5 samples)
        "A", "C", "C", "B", "B",
    ]
    ground_truths = [
        # DatasetA GT
        "sad", "B", "B", "B", "A", "A",
        # DatasetB GT
        "A", "C", "B", "B", "C",
    ]
    datasets = [
        "DatasetA","DatasetA","DatasetA","DatasetA","DatasetA","DatasetA",
        "DatasetB","DatasetB","DatasetB","DatasetB","DatasetB",
    ]

    metrics = compute_metrics_by_data_source(predictions, ground_truths, datasets)
    print("=== Synthetic two-dataset sanity check ===")
    print(json.dumps(metrics, indent=4))