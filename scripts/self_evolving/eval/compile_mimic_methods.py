"""Compile complete method runs into a paper table and a token-cost ledger.

Run after syncing the two backbone directories from the remote shared storage.
Partial results never become paper rows. --allow-partial is for monitoring only
and writes JSON with status fields, without generating LaTeX.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from mimic_method_baselines import METHODS, digest, file_digest, read_rows, write_json

CHAPTERS = (
    "neoplasms",
    "endocrine_metabolic",
    "nervous_system",
    "blood_immune",
    "circulatory",
    "infectious",
    "digestive",
    "other",
)

LABELS = {
    "direct": "Direct",
    "medrag": "Medical RAG",
    "rag_fusion": "RAG-Fusion",
    "imedrag": "i-MedRAG",
    "self_consistency": "Self-consistency (5 samples)",
    "self_refine": "Self-refinement",
    "cove": "Chain-of-verification",
}


def collect(directory, method):
    result = {"method": method, "status": "pending"}
    summary_path = directory / f"{method}.summary.json"
    if not summary_path.exists():
        return result
    # Only open immutable completed files, never a JSONL being appended by a worker.
    generations = read_rows(directory / f"{method}.jsonl")
    graded = read_rows(directory / f"{method}.graded.jsonl")
    summary = json.loads(summary_path.read_text())
    manifest = json.loads((directory / "manifest.json").read_text())
    config = manifest["config"]
    reference_path = Path(config["val_file"])
    if not reference_path.exists():
        reference_path = directory.parent / "test.jsonl"
    if digest(config) != manifest["config_sha256"] or file_digest(reference_path) != config["dataset_sha256"]:
        raise ValueError("Manifest or original evaluation data changed")
    reference = {str(row["extra_info"]["hadm_id"]): row for row in read_rows(reference_path)}
    generation_ids = [str(row["hadm_id"]) for row in generations]
    graded_ids = [str(row["hadm_id"]) for row in graded]
    if (
        summary["n"] != 2452
        or len(generations) != 2452
        or len(graded) != 2452
        or len(set(generation_ids)) != 2452
        or set(graded_ids) != set(generation_ids)
        or set(generation_ids) != set(reference)
    ):
        raise ValueError(f"Incomplete/duplicate rows in {directory}/{method}")
    if summary["config_sha256"] != manifest["config_sha256"]:
        raise ValueError("Summary/manifest mismatch")
    if any(row["config_sha256"] != manifest["config_sha256"] for row in generations + graded):
        raise ValueError("Row/manifest mismatch")
    allowed_implementations = {config["code_sha256"], *manifest.get("implementation_history", {})}
    for row in generations:
        if row.get("implementation_sha256", config["code_sha256"]) not in allowed_implementations:
            raise ValueError("Unregistered generation implementation")
    if summary["method"] != method or summary["model"] != config["model"]:
        raise ValueError("Summary method/model mismatch")
    if summary["judge_model"] != config["judge_model"]:
        raise ValueError("Summary judge mismatch")
    by_id = {str(row["hadm_id"]): row for row in generations}
    counts, scores = Counter(), Counter()
    for row in graded:
        original = reference[str(row["hadm_id"])]
        generated = by_id[str(row["hadm_id"])]
        if row["ground_truth"] != original["reward_model"]["ground_truth"]:
            raise ValueError("Ground-truth alignment mismatch")
        if row["data_source"] != original["data_source"]:
            raise ValueError("Chapter alignment mismatch")
        for key in ("ground_truth", "data_source", "extracted_answer", "response", "method", "model"):
            if row[key] != generated[key]:
                raise ValueError(f"Graded/generation mismatch: {key}")
        if row["method"] != method or row["model"] != config["model"] or row["judge_acc_lenient"] not in (0, 1):
            raise ValueError("Invalid per-case method, model, or grade")
        category = row["data_source"].split("/")[-1]
        counts[category] += 1
        scores[category] += row["judge_acc_lenient"]
    if set(counts) != set(CHAPTERS) or dict(counts) != summary["cat_n"]:
        raise ValueError("Chapter counts disagree with summary")
    if any(abs(scores[key] / counts[key] - summary["per_cat"][key]) > 1e-12 for key in counts):
        raise ValueError("Chapter grades disagree with summary")
    accuracy = sum(row["judge_acc_lenient"] for row in graded) / len(graded)
    if abs(accuracy - summary["overall"]) > 1e-12:
        raise ValueError("Per-case grades disagree with summary")
    calls = [call for row in generations for call in row["trace"] if call["kind"] == "chat"]
    retrievals = [call for row in generations for call in row["trace"] if call["kind"] == "retrieval"]
    result.update(
        summary,
        status="complete",
        verified=True,
        generated=len(generations),
        graded=len(graded),
        config=manifest["config"],
        mean_model_calls=len(calls) / 2452,
        mean_prompt_tokens=sum(call["usage"].get("prompt_tokens", 0) for call in calls) / 2452,
        mean_completion_tokens=sum(call["usage"].get("completion_tokens", 0) for call in calls) / 2452,
        calls_missing_usage=sum(not call["usage"] for call in calls),
        truncated_calls=sum(call["finish_reason"] == "length" for call in calls),
        retrieval_calls=len(retrievals),
        empty_retrieval_calls=sum(not call["documents"] for call in retrievals),
        deepened_retrieval_calls=sum(len(call.get("searches", [])) > 1 for call in retrievals),
        retrieval_calls_with_depth_audit=sum("searches" in call for call in retrievals),
    )
    return result


def validate_protocol(rows):
    completed = [row for values in rows.values() for row in values if row["status"] == "complete"]
    configs = [row["config"] for row in completed]
    for key in (
        "dataset_sha256",
        "reward_sha256",
        "judge_model",
        "max_tokens",
        "sources",
        "collection",
        "embed_model",
        "top_k",
        "passage_chars",
        "max_pixels",
        "max_text_chars",
        "code_sha256",
        "eval_module_sha256",
        "samples",
        "seed",
        "queries",
        "rounds",
        "followups",
        "verifications",
        "aux_tokens",
        "provider",
        "reasoning_effort",
    ):
        if len({digest(config[key]) for config in configs}) > 1:
            raise ValueError(f"Unmatched protocol across backbones: {key}")
    if len({row["judge_prompt_sha256"] for row in completed}) > 1:
        raise ValueError("Unmatched judge prompts")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True, help="JSON ledger path; complete runs also write sibling .tex")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    rows = {tag: [collect(Path(args.root) / tag, method) for method in METHODS] for tag in ("qwen35_9b", "qwen36_27b")}
    validate_protocol(rows)
    incomplete = [
        f"{tag}/{row['method']}" for tag, values in rows.items() for row in values if row["status"] != "complete"
    ]
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, {"rows": rows, "incomplete": incomplete})
    if incomplete:
        print("Pending: " + ", ".join(incomplete))
        if not args.allow_partial:
            raise SystemExit("Full matrix required for a paper table")
        return
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Frozen-backbone inference methods on MIMIC-IV rare-disease diagnosis ($n=2452$). "
        r"All methods use the same cases and fixed judge. RAG methods share a medical reference index.}",
        r"\label{tab:mimic-methods}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Method & Qwen3.5-9B & Qwen3.6-27B \\",
        r"\midrule",
    ]
    for index, method in enumerate(METHODS):
        nine = rows["qwen35_9b"][index]["overall"]
        large = rows["qwen36_27b"][index]["overall"]
        lines.append(f"{LABELS[method]} & {nine:.3f} & {large:.3f} " + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    path.with_suffix(".tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
