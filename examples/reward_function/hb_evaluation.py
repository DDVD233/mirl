import datetime
import json
import os
from collections import defaultdict
from typing import Dict, List, Set

def parse_conditions(text: str) -> Set[str]:
    # Kept for compatibility; not used by the single-label pipeline.
    text = text.replace("\\boxed{", "").replace("}", "")
    for sep in [", ", " and ", " & ", ",", "&"]:
        if sep in text:
            return set(cond.strip() for cond in text.split(sep))
    return {text.strip()}

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
# NEW: combine confusion + per-class metrics
# -----------------------
def compute_class_counts_and_metrics(
    predictions: List[str],
    ground_truths: List[str],
) -> Dict[str, object]:
    """
    Single-label multiclass:
      1) Build per-class one-vs-rest TP/FP/FN/TN and support (count).
      2) Compute per-class metrics in the same function.

    Returns dict with:
      - class_metrics: {label: {precision, recall, sensitivity, specificity, f1, accuracy, count, confusion_matrix}}
      - pooled_counts: {'tp','fp','fn','tn'}
      - active_classes: int
      - total_support: int
    """
    assert len(predictions) == len(ground_truths)

    y_pred = [extract_boxed_content(p).strip() for p in predictions]
    y_true = [extract_boxed_content(t).strip() for t in ground_truths]
    labels = sorted(set(y_true) | set(y_pred))

    # 1) build counts
    matrices = {c: {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "count": 0} for c in labels}

    for p, t in zip(y_pred, y_true):
        for c in labels:
            if t == c:
                matrices[c]["count"] += 1  # support for class c
            if p == c and t == c:
                matrices[c]["tp"] += 1
            elif p == c and t != c:
                matrices[c]["fp"] += 1
            elif p != c and t == c:
                matrices[c]["fn"] += 1
            else:
                matrices[c]["tn"] += 1

    # pooled counts for micro
    pooled_tp = sum(m["tp"] for m in matrices.values())
    pooled_fp = sum(m["fp"] for m in matrices.values())
    pooled_fn = sum(m["fn"] for m in matrices.values())
    pooled_tn = sum(m["tn"] for m in matrices.values())

    # 2) per-class metrics
    class_metrics: Dict[str, Dict[str, float]] = {}
    active_classes = 0
    total_support = 0

    for c, m in matrices.items():
        
        tp, fp, fn, tn = m["tp"], m["fp"], m["fn"], m["tn"]
        
        # support is the number of samples
        support = m["count"]

        precision   = _safe_div(tp, (tp + fp))
        recall      = _safe_div(tp, (tp + fn))
        sensitivity = recall
        specificity = _safe_div(tn, (tn + fp))
        f1          = _safe_div(2 * precision * recall, (precision + recall))
        accuracy    = _safe_div((tp + tn), (tp + tn + fp + fn))

        class_metrics[c] = {
            "precision": precision,
            "recall": recall,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "f1": f1,
            "accuracy": accuracy,
            "count": support,
            "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        }

        if support > 0:
            # if there is more than 1 sample for the class,
            # the class is considered active and present
            active_classes += 1
            total_support += support

    return {
        "class_metrics": class_metrics,
        "pooled_counts": {"tp": pooled_tp, "fp": pooled_fp, "fn": pooled_fn, "tn": pooled_tn},
        "active_classes": active_classes,
        "total_support": total_support,
    }

def compute_dataset_metrics(predictions: List[str], ground_truths: List[str]) -> Dict[str, Dict]:
    """
    Compute per-class metrics and dataset-level macro/micro/weighted metrics with clear prefixes.
    """
    summary = compute_class_counts_and_metrics(predictions, ground_truths)
    class_metrics = summary["class_metrics"]
    pooled = summary["pooled_counts"]
    active_classes = summary["active_classes"]
    total_support = summary["total_support"]

    # Accumulate macro & weighted from per-class metrics
    keys = ["precision", "recall", "sensitivity", "specificity", "f1", "accuracy"]

    macro_sum = {k: 0.0 for k in keys}
    weighted_sum = {k: 0.0 for k in keys}
    
    # iterate over the metrics of each class
    for c, cm in class_metrics.items():
        support = cm["count"]

        # if the class has more than 1 sample, then we can use it
        # essentially, we are totalling up the different precision/ recall/ f1 etc. for all classes
        # which is later than divided in the next section
        if support > 0:
            for k in keys:
                macro_sum[k] += cm[k]
                weighted_sum[k] += cm[k] * support

    # Macro (equal class weighting over active classes)
    macro = {
        f"macro_{k}": _safe_div(macro_sum[k], active_classes) if active_classes > 0 else 0.0
        for k in keys
    }

    # Weighted (by support)
    weighted = {
        f"weighted_{k}": _safe_div(weighted_sum[k], total_support) if total_support > 0 else 0.0
        for k in keys
    }

    # Micro (pooled)
    # Essentially taking all the TP/FP/FN/TN and calculating the metrics
    PTP, PFP, PFN, PTN = pooled["tp"], pooled["fp"], pooled["fn"], pooled["tn"]
    micro_precision   = _safe_div(PTP, (PTP + PFP))
    micro_recall      = _safe_div(PTP, (PTP + PFN))
    micro_f1          = _safe_div(2 * micro_precision * micro_recall, (micro_precision + micro_recall))
    micro_specificity = _safe_div(PTN, (PTN + PFP))
    micro_accuracy    = _safe_div((PTP + PTN), (PTP + PTN + PFP + PFN))

    micro = {
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_sensitivity": micro_recall,
        "micro_specificity": micro_specificity,
        "micro_f1": micro_f1,
        "micro_accuracy": micro_accuracy,
    }

    dataset_metrics = {}
    dataset_metrics.update(macro)
    dataset_metrics.update(weighted)
    dataset_metrics.update(micro)

    return {
        "class_metrics": class_metrics,     # unchanged shape, useful for diagnostics
        "dataset_metrics": dataset_metrics, # only prefixed keys
        "active_classes": active_classes, # number of active classes, which we then use later to identify whether the dataset is active
    }


def compute_metrics_by_data_source(
    predictions: List[str],
    ground_truths: List[str],
    datasets: List[str],
) -> Dict[str, float]:
    """
    Compute metrics at the dataset level and a global mean across datasets (no data sources).
    Aggregates the prefixed dataset-level metrics produced by compute_dataset_metrics.
    """
    # Save inputs (no data sources) to outputs/
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    with open(
        os.path.join(output_dir, f"input_data_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"), "w"
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

    # Per-dataset results + accumulate for global mean (equal dataset weighting)
    global_accum = None
    n_datasets = 0

    for dataset_name, data in grouped.items():
        ds_res = compute_dataset_metrics(data["preds"], data["gts"])
        ds_metrics = ds_res["dataset_metrics"]

        # Discover metric keys once (e.g., macro_*, micro_*, weighted_*)
        if not discovered_metric_keys:
            discovered_metric_keys = sorted(ds_metrics.keys())

        # Store dataset-level metrics
        for k in discovered_metric_keys:
            result[f"{dataset_name}/{k}"] = ds_metrics.get(k, 0.0)

        # Skip empty datasets (no active classes)
        if ds_res["active_classes"] == 0:
            continue

        # Accumulate for global mean
        if global_accum is None:
            global_accum = {k: 0.0 for k in discovered_metric_keys}
        for k in discovered_metric_keys:
            global_accum[k] += ds_metrics.get(k, 0.0)
        n_datasets += 1

    # Global (equal dataset weighting)
    if n_datasets > 0:
        for k in discovered_metric_keys:
            result[f"val/{k}"] = global_accum[k] / n_datasets

    return result

if __name__ == "__main__":
    outputs_dir = "../../outputs"
    output_files = [f for f in os.listdir(outputs_dir) if f.startswith("input_data_") and f.endswith(".json")]
    if not output_files:
        print("No output files found in the outputs directory.")
    else:
        latest_file = max(output_files, key=lambda f: os.path.getmtime(os.path.join(outputs_dir, f)))
        with open(os.path.join(outputs_dir, latest_file), "r") as f:
            input_data = json.load(f)

        predictions = input_data["predictions"]
        ground_truths = input_data["ground_truths"]
        data_sources = input_data["data_sources"]
        datasets = input_data["datasets"]
        demographics = input_data.get("demographics", [])  # unused

        metrics = compute_metrics_by_data_source(predictions, ground_truths, data_sources, datasets, demographics)
        print(json.dumps(metrics, indent=4))