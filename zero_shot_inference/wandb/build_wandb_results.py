#!/usr/bin/env python3
"""
Pull zero-shot / reasoning metrics from W&B and render a LaTeX table.

The inference scripts log one W&B run per model and use summary keys shaped like:
    {dataset}/accuracy
    {dataset}/weighted_f1
    {dataset}/reasoning_accuracy

This utility normalizes those run summaries into JSON and writes a table from the
same JSON, so the table can be regenerated later without another W&B request.

Examples:
    python build_wandb_results.py \\
        --entity my-team \\
        --project zero-shot-inference \\
        --models Qwen/Qwen2.5-Omni-7B PhilipC/HumanOmniV2 \\
        --task inference \\
        --output_json outputs/zero_shot_results.json \\
        --output_tex outputs/zero_shot_results.tex

    python build_wandb_results.py \\
        --entity my-team \\
        --project reasoning-evaluation \\
        --models Qwen/Qwen2.5-Omni-7B ddvd233/OmniSapiens-7B-RL \\
        --task reasoning \\
        --table_metrics reasoning_accuracy direct_accuracy self_consistency_rate para_consistency_rate \\
        --output_json outputs/reasoning_results.json \\
        --output_tex outputs/reasoning_results.tex

    python build_wandb_results.py \\
        --from_json outputs/reasoning_results.json \\
        --output_tex outputs/reasoning_results.tex
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_DATASETS = ["eatd", "mvsa", "av-asd", "iemocap", "dreaddit", "sarcnet", "overall"]
INFERENCE_METRICS = ["accuracy", "exact_match_accuracy", "weighted_f1", "n_samples"]
REASONING_METRICS = [
    "reasoning_accuracy",
    "direct_accuracy",
    "accuracy_delta",
    "self_consistency_rate",
    "para_consistency_rate",
    "mean_reasoning_tokens",
    "efficiency_score",
    "n_faithfulness",
]
TASK_DEFAULT_TABLE_METRICS = {
    "inference": ["accuracy", "weighted_f1"],
    "reasoning": [
        "reasoning_accuracy",
        "direct_accuracy",
        "self_consistency_rate",
        "para_consistency_rate",
        "mean_reasoning_tokens",
    ],
}
METRIC_LABELS = {
    "accuracy": "Acc.",
    "exact_match_accuracy": "Exact Acc.",
    "weighted_f1": "W-F1",
    "n_samples": "N",
    "reasoning_accuracy": "Reason Acc.",
    "direct_accuracy": "Direct Acc.",
    "accuracy_delta": "$\\Delta$ Acc.",
    "self_consistency_rate": "Self Cons.",
    "para_consistency_rate": "Para Cons.",
    "mean_reasoning_tokens": "Mean Tokens",
    "efficiency_score": "Efficiency",
    "n_faithfulness": "N",
}


def _clean_summary_value(value: Any) -> Any:
    """Convert W&B summary values to plain JSON scalars when possible."""
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def _summary_dict(run: Any) -> dict[str, Any]:
    """Return W&B run.summary as a plain dict, excluding private W&B keys."""
    raw = dict(run.summary)
    return {
        str(k): _clean_summary_value(v)
        for k, v in raw.items()
        if not str(k).startswith("_")
    }


def _parse_metric_key(key: str) -> tuple[str, str] | None:
    if "/" not in key:
        return None
    dataset, metric = key.split("/", 1)
    dataset = dataset.strip()
    metric = metric.strip()
    if not dataset or not metric:
        return None
    return dataset, metric


def _model_from_run(run: Any) -> str:
    config = dict(getattr(run, "config", {}) or {})
    return str(config.get("model") or getattr(run, "name", None) or getattr(run, "id", "unknown"))


def _run_matches(run: Any, models: list[str] | None, contains: list[str] | None) -> bool:
    if not models and not contains:
        return True

    config = dict(getattr(run, "config", {}) or {})
    candidates = {
        str(getattr(run, "id", "")),
        str(getattr(run, "name", "")),
        str(getattr(run, "display_name", "")),
        str(config.get("model", "")),
    }
    if models and any(model in candidates for model in models):
        return True

    haystack = " ".join(candidates)
    if contains and any(piece in haystack for piece in contains):
        return True
    return False


def _run_timestamp(run: Any) -> str | None:
    created_at = getattr(run, "created_at", None)
    if created_at is None:
        return None
    if isinstance(created_at, datetime):
        return created_at.astimezone(timezone.utc).isoformat()
    return str(created_at)


def _select_latest_by_model(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for run in sorted(runs, key=lambda r: r.get("created_at") or ""):
        latest[run["model"]] = run
    return list(latest.values())


def fetch_wandb_results(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import wandb
    except ImportError as exc:
        raise SystemExit("wandb is not installed. Install it with `pip install wandb`.") from exc

    if not args.project:
        raise SystemExit("--project is required unless --from_json is used")

    api = wandb.Api(timeout=args.timeout)
    project_path = f"{args.entity}/{args.project}" if args.entity else args.project

    runs = []
    if args.run_ids:
        for run_id in args.run_ids:
            run = api.run(f"{project_path}/{run_id}")
            runs.append(run)
    else:
        runs = list(api.runs(project_path))

    normalized_runs: list[dict[str, Any]] = []
    wanted_metrics = set(args.metrics or [])
    wanted_datasets = set(args.datasets or [])

    for run in runs:
        if not _run_matches(run, args.models, args.run_name_contains):
            continue

        summary = _summary_dict(run)
        datasets: OrderedDict[str, dict[str, Any]] = OrderedDict()
        for key, value in sorted(summary.items()):
            parsed = _parse_metric_key(key)
            if not parsed:
                continue
            dataset, metric = parsed
            if wanted_datasets and dataset not in wanted_datasets:
                continue
            if wanted_metrics and metric not in wanted_metrics:
                continue
            datasets.setdefault(dataset, OrderedDict())[metric] = value

        if not datasets:
            continue

        normalized_runs.append({
            "model": _model_from_run(run),
            "run_id": str(run.id),
            "run_name": str(run.name),
            "run_path": "/".join(run.path),
            "url": getattr(run, "url", None),
            "state": getattr(run, "state", None),
            "created_at": _run_timestamp(run),
            "datasets": datasets,
        })

    if args.latest_per_model:
        normalized_runs = _select_latest_by_model(normalized_runs)

    if args.models:
        order = {model: i for i, model in enumerate(args.models)}
        normalized_runs.sort(key=lambda r: (order.get(r["model"], len(order)), r["model"], r["run_name"]))
    else:
        normalized_runs.sort(key=lambda r: (r["model"], r["run_name"]))

    task = args.task
    if task == "auto":
        seen_metrics = {
            metric
            for run in normalized_runs
            for metrics in run["datasets"].values()
            for metric in metrics
        }
        task = "reasoning" if "reasoning_accuracy" in seen_metrics else "inference"

    return {
        "source": {
            "entity": args.entity,
            "project": args.project,
            "task": task,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        },
        "filters": {
            "models": args.models or [],
            "run_ids": args.run_ids or [],
            "run_name_contains": args.run_name_contains or [],
            "datasets": args.datasets or [],
            "metrics": args.metrics or [],
            "latest_per_model": args.latest_per_model,
        },
        "runs": normalized_runs,
    }


def _load_json(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str | os.PathLike[str], payload: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _latex_escape(text: Any) -> str:
    s = str(text)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in s)


def _format_value(value: Any, metric: str, percent: bool) -> str:
    if value is None or value == "":
        return "--"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if metric.startswith("n_") or metric == "n_samples":
            return str(int(round(value)))
        if percent and -1.0 <= value <= 1.0:
            return f"{value * 100:.1f}"
        if abs(value) >= 100:
            return f"{value:.1f}"
        return f"{value:.3f}"
    return _latex_escape(value)


def _metric_value(metrics: dict[str, Any], metric: str) -> Any:
    """Return metric value, with table-only fallbacks for equivalent columns."""
    if metric in metrics:
        return metrics[metric]
    if metric == "accuracy":
        return metrics.get("exact_match_accuracy")
    if metric == "exact_match_accuracy":
        return metrics.get("accuracy")
    return None


def _ordered_datasets(payload: dict[str, Any], requested: list[str] | None) -> list[str]:
    seen = {
        dataset
        for run in payload.get("runs", [])
        for dataset in run.get("datasets", {})
    }
    if requested:
        return [dataset for dataset in requested if dataset in seen]
    ordered = [dataset for dataset in DEFAULT_DATASETS if dataset in seen]
    ordered.extend(sorted(seen - set(ordered)))
    return ordered


def _table_metrics(args: argparse.Namespace, payload: dict[str, Any]) -> list[str]:
    if args.table_metrics:
        return args.table_metrics
    task = payload.get("source", {}).get("task", "auto")
    if task in TASK_DEFAULT_TABLE_METRICS:
        return TASK_DEFAULT_TABLE_METRICS[task]
    seen = []
    for run in payload.get("runs", []):
        for metrics in run.get("datasets", {}).values():
            for metric in metrics:
                if metric not in seen:
                    seen.append(metric)
    return seen[:4]


def _apply_alias(name: str, aliases: dict[str, str]) -> str:
    return aliases.get(name, name)


def _parse_aliases(raw_aliases: list[str] | None) -> dict[str, str]:
    aliases = {}
    for item in raw_aliases or []:
        if "=" not in item:
            raise SystemExit(f"Alias must be OLD=NEW, got: {item}")
        old, new = item.split("=", 1)
        aliases[old] = new
    return aliases


def render_latex_table(payload: dict[str, Any], args: argparse.Namespace) -> str:
    metrics = _table_metrics(args, payload)
    datasets = _ordered_datasets(payload, args.datasets)
    aliases = _parse_aliases(args.model_alias)
    caption = args.caption or (
        "Zero-shot inference results." if payload.get("source", {}).get("task") == "inference"
        else "Reasoning evaluation results."
    )
    label = args.label or (
        "tab:zero-shot-inference" if payload.get("source", {}).get("task") == "inference"
        else "tab:reasoning-evaluation"
    )

    n_metric_cols = len(metrics)
    colspec = "ll" + ("r" * n_metric_cols)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
    ]
    header = ["Model", "Dataset"] + [METRIC_LABELS.get(metric, metric.replace("_", " ").title()) for metric in metrics]
    lines.append(" & ".join(_latex_escape(h) if not str(h).startswith("$") else str(h) for h in header) + r" \\")
    lines.append(r"\midrule")

    runs = payload.get("runs", [])
    for run_idx, run in enumerate(runs):
        model = _latex_escape(_apply_alias(run.get("model", "unknown"), aliases))
        first_dataset_for_model = True
        for dataset in datasets:
            row_metrics = run.get("datasets", {}).get(dataset)
            if not row_metrics:
                continue
            model_cell = model if first_dataset_for_model else ""
            row = [
                model_cell,
                _latex_escape(dataset),
                *[_format_value(_metric_value(row_metrics, metric), metric, args.percent) for metric in metrics],
            ]
            lines.append(" & ".join(row) + r" \\")
            first_dataset_for_model = False
        if run_idx != len(runs) - 1:
            lines.append(r"\addlinespace")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption{{{_latex_escape(caption)}}}",
        rf"\label{{{_latex_escape(label)}}}",
        r"\end{table}",
        "",
    ])
    return "\n".join(lines)


def _write_text(path: str | os.PathLike[str], text: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    source = parser.add_argument_group("W&B source")
    source.add_argument("--entity", default=None, help="W&B entity/team. Optional when wandb has a default entity.")
    source.add_argument("--project", default=None, help="W&B project, e.g. zero-shot-inference or reasoning-evaluation.")
    source.add_argument("--models", nargs="+", default=None,
                        help="Model identifiers to keep. Matches config.model, run id, run name, or display name exactly.")
    source.add_argument("--run_ids", nargs="+", default=None, help="Fetch specific W&B run IDs.")
    source.add_argument("--run_name_contains", nargs="+", default=None,
                        help="Keep runs whose id/name/display_name/config.model contains any of these strings.")
    source.add_argument("--latest_per_model", action="store_true",
                        help="If multiple matching runs have the same config.model, keep the latest one.")
    source.add_argument("--timeout", type=int, default=60, help="W&B API timeout in seconds.")
    source.add_argument("--from_json", default=None,
                        help="Skip W&B and render from a previously generated JSON file.")

    output = parser.add_argument_group("Outputs")
    output.add_argument("--output_json", default="wandb_results.json", help="Normalized JSON output path.")
    output.add_argument("--output_tex", default="wandb_results.tex", help="LaTeX table output path.")

    selection = parser.add_argument_group("Metric selection")
    selection.add_argument("--task", choices=["auto", "inference", "reasoning"], default="auto")
    selection.add_argument("--datasets", nargs="+", default=None, help="Datasets to keep/order in JSON and table.")
    selection.add_argument("--metrics", nargs="+", default=None,
                           help="Metrics to keep in JSON. Defaults to all summary metrics with dataset/metric keys.")
    selection.add_argument("--table_metrics", nargs="+", default=None,
                           help="Metrics to show in the TeX table. Defaults depend on --task.")

    table = parser.add_argument_group("Table formatting")
    table.add_argument("--caption", default=None)
    table.add_argument("--label", default=None)
    table.add_argument("--model_alias", nargs="+", default=None,
                       help="Model display aliases, e.g. Qwen/Qwen2.5-Omni-7B=Qwen2.5-Omni.")
    table.add_argument("--percent", action=argparse.BooleanOptionalAction, default=True,
                       help="Render values in [-1, 1] as percentages. Default: true.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.from_json:
        payload = _load_json(args.from_json)
    else:
        payload = fetch_wandb_results(args)
        _write_json(args.output_json, payload)

    tex = render_latex_table(payload, args)
    _write_text(args.output_tex, tex)

    n_runs = len(payload.get("runs", []))
    print(f"Wrote {args.output_tex}")
    if not args.from_json:
        print(f"Wrote {args.output_json}")
    print(f"Included {n_runs} run(s).")


if __name__ == "__main__":
    main()
