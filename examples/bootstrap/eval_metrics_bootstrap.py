import argparse
import json
import numpy as np
from sklearn.metrics import f1_score

RNG = np.random.RandomState(42)

TASK_GROUPS = {
    "EMO": ["cremad", "meld_emotion", "mosei_emotion", "tess"],
    "HUM": ["urfunny"],
    "PTSD": ["ptsd_in_the_wild"],
    "ANX": ["mmpsy_anxiety"],
    "DEP": ["mmpsy_depression", "daicwoz"],
    "SEN": ["meld_senti", "chsimsv2", "mosei_senti"],
    "SAR": ["mmsd"],
    "INT": ["intentqa"],
    "SOC": ["social_iq_2"],
    "NVC": ["mimeqa"],
}

# ── metric functions ──────────────────────────────────────────────────────────

def bootstrap_std(y_true, y_pred, metric_fn, n_boot=1000):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    scores = []
    for _ in range(n_boot):
        idx = RNG.randint(0, len(y_true), len(y_true))
        scores.append(metric_fn(y_true[idx], y_pred[idx]))
    return float(np.std(scores))


def emotion_wa_metric(y_true, y_pred):
    """Mean weighted accuracy: mean of 0.5*(TP/P + TN/N) per class."""
    classes = np.unique(y_true)
    accs = []
    for c in classes:
        P = int((y_true == c).sum())
        N = int((y_true != c).sum())
        TP = int(((y_true == c) & (y_pred == c)).sum())
        TN = int(((y_true != c) & (y_pred != c)).sum())
        accs.append(0.5 * ((TP / P if P else 0.0) + (TN / N if N else 0.0)))
    return float(np.mean(accs)) if accs else 0.0


def make_sentiment_f1w2_metric(meta):
    """
    Returns a closure that computes binary weighted F1 (F1w2) for sentiment.
    Collapses 7-way sentiment to binary (NEG=0, POS=1), excluding neutral GT.
    """
    canonical = [
        "highly negative",   # pos 0
        "negative",          # pos 1
        "weakly negative",   # pos 2
        "neutral",           # pos 3
        "weakly positive",   # pos 4
        "positive",          # pos 5
        "highly positive",   # pos 6
    ]
    gc = meta["global_classes"]["sentiment_intensity"]
    label_to_global = {e["label"]: e["index"] for e in gc}
    global_to_pos = {label_to_global[lab]: i for i, lab in enumerate(canonical)}

    NEG_POS = {0, 1, 2}   # canonical positions for negative
    POS_POS = {4, 5, 6}   # canonical positions for positive
    NEUTRAL_POS = 3

    def _translate(seq):
        return [global_to_pos.get(int(v), -1) for v in seq]

    def fn(y_true, y_pred):
        y_true_pos = _translate(y_true)
        y_pred_pos = _translate(y_pred)

        # exclude neutral GT
        keep = [i for i, t in enumerate(y_true_pos) if t != NEUTRAL_POS and t != -1]
        if not keep:
            return 0.0

        def collapse(seq_pos, indices):
            out = []
            for i in indices:
                p = seq_pos[i]
                if p in NEG_POS:
                    out.append(0)
                elif p in POS_POS:
                    out.append(1)
                else:
                    out.append(-1)
            return out

        yt2 = collapse(y_true_pos, keep)
        yp2 = collapse(y_pred_pos, keep)

        # filter out any -1 that slipped through in pred
        valid = [(t, p) for t, p in zip(yt2, yp2) if t != -1]
        if not valid:
            return 0.0
        yt2, yp2 = zip(*valid)
        return float(f1_score(list(yt2), list(yp2), average="weighted", zero_division=0))

    return fn


def weighted_f1_metric(y_true, y_pred):
    return float(f1_score(y_true, y_pred, average="weighted", zero_division=0))


def accuracy_metric(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


# ── data loading ─────────────────────────────────────────────────────────────

def load_cls_json(path):
    with open(path) as f:
        d = json.load(f)
    datasets = {}
    for pred, gt, ds in zip(d["predictions"], d["ground_truths"], d["datasets"]):
        datasets.setdefault(ds, {"preds": [], "gts": [], "source": "cls"})
        datasets[ds]["preds"].append(int(pred))
        datasets[ds]["gts"].append(int(gt))
    return datasets


def load_llm_json(path):
    with open(path) as f:
        records = json.load(f)
    datasets = {}
    for r in records:
        ds = r["dataset"]
        datasets.setdefault(ds, {"preds": [], "gts": [], "source": "llm"})
        datasets[ds]["preds"].append(int(bool(r["graded_result"])))
        datasets[ds]["gts"].append(1)
    return datasets


# ── per-dataset computation ───────────────────────────────────────────────────

def compute_dataset_result(ds_name, data, dataset_to_domain, senti_fn, n_boot=1000):
    preds = np.asarray(data["preds"])
    gts = np.asarray(data["gts"])
    source = data["source"]
    N = len(gts)

    if source == "llm":
        metric_key = "accuracy"
        fn = accuracy_metric
    else:
        domain = dataset_to_domain.get(ds_name)
        if domain == "emotion":
            metric_key = "mean_weighted_accuracy"
            fn = emotion_wa_metric
        elif domain == "sentiment_intensity":
            metric_key = "F1w2"
            fn = senti_fn
        else:
            metric_key = "weighted_f1"
            fn = weighted_f1_metric

    mean = fn(gts, preds)
    std = bootstrap_std(gts, preds, fn, n_boot=n_boot)
    return {
        "N": N,
        metric_key: {
            "mean": round(mean, 6),
            "std": round(std, 6),
            "fmt": f"{mean:.4f}±{std:.4f}",
        },
    }


# ── task-level avg ────────────────────────────────────────────────────────────

def compute_task_avg(ds_results):
    """Macro-average of dataset means; std across dataset means."""
    # Collect metric keys (excluding "N")
    metric_keys = set()
    for r in ds_results.values():
        metric_keys.update(k for k in r if k != "N")

    avg = {}
    for mk in metric_keys:
        vals = [r[mk]["mean"] for r in ds_results.values() if mk in r]
        if not vals:
            continue
        m = float(np.mean(vals))
        s = float(np.std(vals))
        avg[mk] = {"mean": round(m, 6), "std": round(s, 6), "fmt": f"{m:.4f}±{s:.4f}"}
    return avg


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cls_json", required=True)
    parser.add_argument("--llm_json", required=True)
    parser.add_argument("--label_map", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n_boot", type=int, default=1000,
                        help="Number of bootstrap resamples (default: 1000)")
    args = parser.parse_args()

    # Load label map
    with open(args.label_map) as f:
        lm = json.load(f)
    meta = lm["meta"]
    dataset_to_domain = meta.get("dataset_domain", {})
    senti_fn = make_sentiment_f1w2_metric(meta)

    # Load both data sources
    all_datasets = {}
    all_datasets.update(load_cls_json(args.cls_json))
    all_datasets.update(load_llm_json(args.llm_json))

    # Compute per-dataset results
    ds_results = {}
    for ds_name, data in all_datasets.items():
        print(f"  Computing {ds_name} (N={len(data['gts'])}, source={data['source']})...")
        ds_results[ds_name] = compute_dataset_result(
            ds_name, data, dataset_to_domain, senti_fn, n_boot=args.n_boot
        )

    # Assemble output by task
    output = {}
    for task, members in TASK_GROUPS.items():
        task_ds = {ds: ds_results[ds] for ds in members if ds in ds_results}
        if not task_ds:
            continue
        output[task] = dict(task_ds)
        output[task]["avg"] = compute_task_avg(task_ds)

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nSaved: {args.output}")

    # Print summary table
    print(f"\n{'Task':<6} {'Dataset':<22} {'Metric':<26} {'Value'}")
    print("-" * 70)
    for task, task_data in output.items():
        for ds, result in task_data.items():
            for mk, mv in result.items():
                if mk == "N":
                    continue
                n_str = f"(N={result.get('N', '?')})" if ds != "avg" else ""
                print(f"{task:<6} {ds:<22} {mk:<26} {mv['fmt']}  {n_str}")


if __name__ == "__main__":
    main()
