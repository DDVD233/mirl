import argparse
import json
import sys
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.stats import chi2 as chi2_dist
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
    rng = np.random.RandomState(42)
    scores = []
    for _ in range(n_boot):
        idx = rng.randint(0, len(y_true), len(y_true))
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
        raw = json.load(f)

    datasets = {}

    if isinstance(raw, list):
        # Per-sample list of dicts: [{dataset, graded_result, ...}, ...]
        for r in raw:
            ds = r["dataset"]
            datasets.setdefault(ds, {"preds": [], "gts": [], "source": "llm"})
            datasets[ds]["preds"].append(int(bool(r["graded_result"])))
            datasets[ds]["gts"].append(1)
    elif isinstance(raw, dict) and "datasets" in raw:
        # Parallel arrays format (mirrors CLS JSON): {datasets: [...], predictions: [...], ...}
        grades = raw.get("graded_results") or raw.get("predictions")
        for ds, grade in zip(raw["datasets"], grades):
            datasets.setdefault(ds, {"preds": [], "gts": [], "source": "llm"})
            datasets[ds]["preds"].append(int(bool(grade)))
            datasets[ds]["gts"].append(1)
    else:
        raise ValueError(
            f"Unrecognised LLM JSON format. Top-level type: {type(raw)}. "
            f"Keys (if dict): {list(raw.keys()) if isinstance(raw, dict) else 'N/A'}"
        )

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


# ── single-model bootstrap (runs in subprocess) ───────────────────────────────

def _run_model_bootstrap(args_tuple):
    """
    Top-level function (picklable) for ProcessPoolExecutor.
    Returns (model_name, task_output, all_datasets_raw).
    all_datasets_raw: {ds_name: {"preds": [...], "gts": [...], "source": ...}}
    """
    model_cfg, label_map_path, n_boot = args_tuple
    model_name = model_cfg["name"]

    print(f"[{model_name}] Loading data...", flush=True)

    with open(label_map_path) as f:
        lm = json.load(f)
    meta = lm["meta"]
    dataset_to_domain = meta.get("dataset_domain", {})
    senti_fn = make_sentiment_f1w2_metric(meta)

    all_datasets = {}
    all_datasets.update(load_cls_json(model_cfg["cls_json"]))
    all_datasets.update(load_llm_json(model_cfg["llm_json"]))

    ds_results = {}
    for ds_name, data in all_datasets.items():
        print(f"  [{model_name}] Computing {ds_name} (N={len(data['gts'])}, source={data['source']})...", flush=True)
        ds_results[ds_name] = compute_dataset_result(
            ds_name, data, dataset_to_domain, senti_fn, n_boot=n_boot
        )

    output = {}
    for task, members in TASK_GROUPS.items():
        task_ds = {ds: ds_results[ds] for ds in members if ds in ds_results}
        if not task_ds:
            continue
        output[task] = dict(task_ds)
        output[task]["avg"] = compute_task_avg(task_ds)

    print(f"[{model_name}] Done.", flush=True)
    return model_name, output, all_datasets


# ── McNemar test ──────────────────────────────────────────────────────────────

def compute_mcnemar_tests(method_name, all_model_samples, alpha=0.05):
    """
    For each baseline (every model that is not the method), for each dataset,
    compute the McNemar test comparing method vs baseline correctness.

    Returns:
        {baseline_name: {dataset: {N, b, c, chi2, p_value, significant, ...}}}
    """
    method_samples = all_model_samples.get(method_name)
    if method_samples is None:
        print(f"WARNING: method '{method_name}' not found in model samples. Skipping McNemar.", flush=True)
        return {}

    results = {}
    for baseline_name, baseline_samples in all_model_samples.items():
        if baseline_name == method_name:
            continue
        results[baseline_name] = {}
        for ds_name, m_data in method_samples.items():
            if ds_name not in baseline_samples:
                print(f"  WARNING: dataset '{ds_name}' missing from baseline '{baseline_name}', skipping.", flush=True)
                continue
            b_data = baseline_samples[ds_name]

            m_preds = np.asarray(m_data["preds"])
            m_gts   = np.asarray(m_data["gts"])
            b_preds = np.asarray(b_data["preds"])
            b_gts   = np.asarray(b_data["gts"])

            if len(m_preds) != len(b_preds):
                print(
                    f"  WARNING: sample count mismatch for '{ds_name}': "
                    f"{method_name}={len(m_preds)}, {baseline_name}={len(b_preds)}. Skipping.",
                    flush=True,
                )
                continue

            m_correct = (m_preds == m_gts).astype(int)
            b_correct = (b_preds == b_gts).astype(int)

            # b: method correct, baseline wrong
            # c: method wrong, baseline correct
            b_count = int(np.sum((m_correct == 1) & (b_correct == 0)))
            c_count = int(np.sum((m_correct == 0) & (b_correct == 1)))
            N = len(m_preds)

            if (b_count + c_count) == 0:
                chi2_stat = 0.0
                p_value = 1.0
            else:
                # Edwards' continuity correction
                chi2_stat = (abs(b_count - c_count) - 1) ** 2 / (b_count + c_count)
                p_value = float(1.0 - chi2_dist.cdf(chi2_stat, df=1))

            if b_count > c_count:
                direction = "method"
            elif c_count > b_count:
                direction = "baseline"
            else:
                direction = "tie"

            significant = p_value < alpha

            results[baseline_name][ds_name] = {
                "N": N,
                "b": b_count,
                "c": c_count,
                "delta_b_minus_c": b_count - c_count,
                "chi2": round(chi2_stat, 4),
                "p_value": round(p_value, 4),
                "alpha": alpha,
                "direction": direction,
                "significant": significant,
                "method_significant_win": significant and direction == "method",
                "baseline_significant_win": significant and direction == "baseline",
            }

    return results


# ── markdown generation ───────────────────────────────────────────────────────

def _get_metric_key(task_output, ds_name):
    """Return the metric key for a dataset result (first non-N key)."""
    ds_result = task_output.get(ds_name, {})
    for k in ds_result:
        if k != "N":
            return k
    return None


def _format_rank(rank):
    if rank is None:
        return "–"
    if float(rank).is_integer():
        return str(int(rank))
    return f"{rank:.1f}"


def _compute_ranks(all_model_results, model_names, task, ds_name, metric_key):
    """
    Rank models by metric mean for one row. Lower rank is better.
    Tied means receive the average rank across the tied positions.
    """
    values = []
    for mn in model_names:
        cell = all_model_results.get(mn, {}).get(task, {}).get(ds_name, {}).get(metric_key)
        if cell is not None and "mean" in cell:
            values.append((mn, cell["mean"]))

    values.sort(key=lambda item: item[1], reverse=True)
    ranks = {}
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[j][1] == values[i][1]:
            j += 1
        rank = ((i + 1) + j) / 2.0
        for mn, _ in values[i:j]:
            ranks[mn] = rank
        i = j

    return ranks


def _format_metric_cell(cell, rank):
    if cell is None:
        return "–"
    return f"{cell['fmt']} (r={_format_rank(rank)})"


def _append_rank_summary(lines, title, rank_values, model_names):
    if not any(rank_values.values()):
        return

    lines.append(f"### {title}\n")
    lines.append("| Model | Average rank | Ranked rows |")
    lines.append("|-------|--------------|-------------|")

    ordered = sorted(
        model_names,
        key=lambda mn: (
            float(np.mean(rank_values[mn])) if rank_values[mn] else float("inf"),
            model_names.index(mn),
        ),
    )
    for mn in ordered:
        vals = rank_values[mn]
        if not vals:
            lines.append(f"| {mn} | – | 0 |")
            continue
        lines.append(f"| {mn} | {float(np.mean(vals)):.2f} | {len(vals)} |")
    lines.append("")


def generate_markdown(all_model_results, mcnemar_results, method_name, model_names, mcnemar_alpha=0.05):
    lines = []

    # ── Section 1: Bootstrap Results ─────────────────────────────────────────
    lines.append("## Bootstrap Evaluation Results: Individual Datasets\n")

    # Build header
    header = "| Task | Dataset | N | Metric |"
    sep    = "|------|---------|---|--------|"
    for mn in model_names:
        header += f" {mn} |"
        sep    += "----------|"
    lines.append(header)
    lines.append(sep)

    # Use the first model's result structure to drive row ordering
    first_model = model_names[0]
    first_results = all_model_results[first_model]
    dataset_rank_values = {mn: [] for mn in model_names}
    avg_rank_values = {mn: [] for mn in model_names}

    for task in TASK_GROUPS:
        if task not in first_results:
            continue
        task_data = first_results[task]
        datasets_in_task = [ds for ds in TASK_GROUPS[task] if ds in task_data]

        for ds in datasets_in_task:
            # Determine metric key from first model
            mk = _get_metric_key(task_data, ds)
            if mk is None:
                continue
            N = task_data[ds].get("N", "–")
            ranks = _compute_ranks(all_model_results, model_names, task, ds, mk)
            row = f"| {task} | {ds} | {N} | {mk} |"
            for mn in model_names:
                cell = all_model_results.get(mn, {}).get(task, {}).get(ds, {}).get(mk)
                row += f" {_format_metric_cell(cell, ranks.get(mn))} |"
                if mn in ranks:
                    dataset_rank_values[mn].append(ranks[mn])
            lines.append(row)

    lines.append("")
    _append_rank_summary(lines, "Average Rank Across Individual Datasets", dataset_rank_values, model_names)

    # ── Section 2: Task Average Results ──────────────────────────────────────
    lines.append("## Bootstrap Evaluation Results: Task Averages\n")

    lines.append(header)
    lines.append(sep)

    for task in TASK_GROUPS:
        if task not in first_results:
            continue
        task_data = first_results[task]
        mk = _get_metric_key(task_data, "avg") if "avg" in task_data else None
        if mk:
            ranks = _compute_ranks(all_model_results, model_names, task, "avg", mk)
            row = f"| {task} | **avg** | – | {mk} |"
            for mn in model_names:
                cell = all_model_results.get(mn, {}).get(task, {}).get("avg", {}).get(mk)
                row += f" {_format_metric_cell(cell, ranks.get(mn))} |"
                if mn in ranks:
                    avg_rank_values[mn].append(ranks[mn])
            lines.append(row)

    lines.append("")
    _append_rank_summary(lines, "Average Rank Across Task Averages", avg_rank_values, model_names)

    # ── Section 3: McNemar Tests ──────────────────────────────────────────────
    if mcnemar_results:
        alpha_label = f"{mcnemar_alpha:.2f}"
        lines.append(f"## McNemar Test: {method_name} vs Baselines\n")
        lines.append(
            "> Per-sample correctness test. "
            "b = method correct & baseline wrong; c = method wrong & baseline correct. "
            "χ² uses Edwards' continuity correction (df=1). "
            f"Sig Method Win = b > c and p < {alpha_label}.\n"
        )

        lines.append("### Summary\n")
        lines.append("| Baseline | Method-favored datasets | Significant method wins | Significant baseline wins | Not significant |")
        lines.append("|----------|-------------------------|--------------------------|----------------------------|-----------------|")
        for baseline_name, ds_map in mcnemar_results.items():
            method_favored = sum(1 for r in ds_map.values() if r["direction"] == "method")
            method_sig = sum(1 for r in ds_map.values() if r["method_significant_win"])
            baseline_sig = sum(1 for r in ds_map.values() if r["baseline_significant_win"])
            not_sig = sum(1 for r in ds_map.values() if not r["significant"])
            total = len(ds_map)
            lines.append(
                f"| {baseline_name} | {method_favored}/{total} | {method_sig}/{total} "
                f"| {baseline_sig}/{total} | {not_sig}/{total} |"
            )
        lines.append("")

        for baseline_name, ds_map in mcnemar_results.items():
            lines.append(f"### vs {baseline_name}\n")
            lines.append("| Task | Dataset | N | b (M+/B−) | c (M−/B+) | Δ=b−c | χ² | p-value | Favored | Sig Method Win |")
            lines.append("|------|---------|---|-----------|-----------|-------|-----|---------|---------|----------------|")

            # Walk datasets in TASK_GROUPS order
            for task, members in TASK_GROUPS.items():
                for ds in members:
                    if ds not in ds_map:
                        continue
                    r = ds_map[ds]
                    if r["direction"] == "method":
                        favored = method_name
                    elif r["direction"] == "baseline":
                        favored = baseline_name
                    else:
                        favored = "tie"
                    sig = "✓" if r["method_significant_win"] else "✗"
                    lines.append(
                        f"| {task} | {ds} | {r['N']} | {r['b']} | {r['c']} "
                        f"| {r['delta_b_minus_c']} | {r['chi2']:.4f} | {r['p_value']:.4f} "
                        f"| {favored} | {sig} |"
                    )
            lines.append("")

    return "\n".join(lines)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()

    # Multi-model mode
    parser.add_argument(
        "--config", default=None,
        help="Path to JSON config file with 'method' (string) and 'models' (list of {name, cls_json, llm_json}).",
    )

    # Single-model mode (legacy / convenience)
    parser.add_argument("--cls_json", default=None)
    parser.add_argument("--llm_json", default=None)
    parser.add_argument("--name", default="model", help="Model name when using single-model mode")

    parser.add_argument("--label_map", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--output_md", default=None,
        help="Path for markdown output. Defaults to <output>.md",
    )
    parser.add_argument("--n_boot", type=int, default=1000,
                        help="Number of bootstrap resamples (default: 1000)")
    parser.add_argument(
        "--mcnemar_alpha",
        type=float,
        default=0.05,
        help="Alpha threshold for McNemar significance labels (default: 0.05).",
    )
    parser.add_argument("--wandb_project", default=None,
                        help="W&B project name. If set, uploads outputs as an artifact.")
    parser.add_argument("--wandb_entity", default=None, help="W&B entity (team/user)")
    parser.add_argument("--wandb_run_name", default=None, help="W&B run name")
    parser.add_argument("--wandb_artifact_name", default="bootstrap_eval",
                        help="W&B artifact name (default: bootstrap_eval)")
    args = parser.parse_args()

    if not 0.0 < args.mcnemar_alpha < 1.0:
        parser.error("--mcnemar_alpha must be between 0 and 1.")

    # Resolve output_md path
    md_path = args.output_md or (
        os.path.splitext(args.output)[0] + ".md"
    )

    # ── Build model configs ───────────────────────────────────────────────────
    if args.config:
        with open(args.config) as f:
            cfg = json.load(f)
        model_configs = cfg["models"]
        method_name = cfg.get("method", model_configs[0]["name"])
    elif args.cls_json and args.llm_json:
        model_configs = [{"name": args.name, "cls_json": args.cls_json, "llm_json": args.llm_json}]
        method_name = args.name
    else:
        parser.error("Provide either --config or both --cls_json and --llm_json.")

    model_names = [m["name"] for m in model_configs]

    # ── Run bootstrap in parallel ─────────────────────────────────────────────
    all_model_results = {}   # {model_name: task_output}
    all_model_samples = {}   # {model_name: {dataset: {preds, gts, source}}}

    worker_args = [(mc, args.label_map, args.n_boot) for mc in model_configs]

    n_workers = min(len(model_configs), os.cpu_count() or 1)
    print(f"Running bootstrap for {len(model_configs)} model(s) with {n_workers} worker(s)...\n")

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_run_model_bootstrap, wa): wa[0]["name"] for wa in worker_args}
        for future in as_completed(futures):
            model_name = futures[future]
            try:
                mn, task_output, raw_datasets = future.result()
                all_model_results[mn] = task_output
                all_model_samples[mn] = raw_datasets
            except Exception as exc:
                print(f"ERROR: model '{model_name}' raised: {exc}", file=sys.stderr)
                raise

    # Preserve original ordering
    all_model_results = {mn: all_model_results[mn] for mn in model_names if mn in all_model_results}
    all_model_samples = {mn: all_model_samples[mn] for mn in model_names if mn in all_model_samples}

    # ── McNemar test ──────────────────────────────────────────────────────────
    mcnemar_results = {}
    if len(model_configs) > 1:
        print(f"\nComputing McNemar tests at alpha={args.mcnemar_alpha:.2f}...")
        mcnemar_results = compute_mcnemar_tests(method_name, all_model_samples, alpha=args.mcnemar_alpha)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    json_output = {
        "models": all_model_results,
        "mcnemar_alpha": args.mcnemar_alpha,
        "mcnemar": mcnemar_results,
    }
    with open(args.output, "w") as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved JSON: {args.output}")

    # ── Save Markdown ─────────────────────────────────────────────────────────
    md_content = generate_markdown(
        all_model_results,
        mcnemar_results,
        method_name,
        model_names,
        mcnemar_alpha=args.mcnemar_alpha,
    )
    with open(md_path, "w") as f:
        f.write(md_content)
    print(f"Saved Markdown: {md_path}")

    # ── Upload to W&B ─────────────────────────────────────────────────────────
    if args.wandb_project:
        import wandb
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            job_type="bootstrap_eval",
        )
        artifact = wandb.Artifact(
            name=args.wandb_artifact_name,
            type="eval_metrics",
            description="Bootstrap evaluation metrics (domain-aware) by task and dataset",
        )
        artifact.add_file(args.output)
        artifact.add_file(md_path)
        run.log_artifact(artifact)
        run.finish()
        print(f"Uploaded to W&B: {args.wandb_project}/{args.wandb_artifact_name}")

    # ── Print summary table ───────────────────────────────────────────────────
    print(f"\n{'Task':<6} {'Dataset':<22} {'Metric':<26}", end="")
    for mn in model_names:
        print(f"  {mn:<20}", end="")
    print()
    print("-" * (56 + 22 * len(model_names)))

    first_results = all_model_results[model_names[0]]
    for task in TASK_GROUPS:
        if task not in first_results:
            continue
        task_data = first_results[task]
        for ds in list(TASK_GROUPS[task]) + ["avg"]:
            if ds not in task_data:
                continue
            for mk, mv in task_data[ds].items():
                if mk == "N":
                    continue
                n_str = f"(N={task_data[ds].get('N', '?')})" if ds != "avg" else ""
                print(f"{task:<6} {ds:<22} {mk:<26}", end="")
                for mn in model_names:
                    cell = all_model_results.get(mn, {}).get(task, {}).get(ds, {}).get(mk)
                    val = cell["fmt"] if cell else "–"
                    print(f"  {val:<20}", end="")
                print(f"  {n_str}")


if __name__ == "__main__":
    main()
