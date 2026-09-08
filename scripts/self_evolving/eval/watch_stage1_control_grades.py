"""Grade complete 9B control validation dumps with the fixed paper judge.

Training and in-run validation remain self-judged. This independent, resumable
watcher never converts an API failure into an incorrect grade.
"""

import argparse
import asyncio
import fcntl
import json
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import aiohttp
from mimic_method_baselines import file_digest, grade, load_module, read_rows, write_json


async def refresh(args):
    reference = read_rows(args.test)
    if len(reference) != 2452 or len({r["extra_info"]["hadm_id"] for r in reference}) != 2452:
        raise ValueError("Expected the full unique-admission reference set")
    reward = load_module("stage1_control_reward", str(args.reward))
    judge_args = SimpleNamespace(
        judge_model="gpt-chat-latest_2026-05-28",
        judge_base="http://point.dd.works:18890/v1",
        judge_key_env="TRAPI_API_KEY",
        retries=4,
    )
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as session:
        for path in sorted(args.traces.glob("*.jsonl"), key=lambda p: int(p.stem)):
            summary_path = args.output / (path.stem + ".summary.json")
            if summary_path.exists():
                summary = json.loads(summary_path.read_text())
                if summary["trace_sha256"] != file_digest(path) or summary["dataset_sha256"] != file_digest(args.test):
                    raise ValueError("Previously graded control inputs changed")
                continue
            try:
                traces = read_rows(path)
            except ValueError:
                continue
            if len(traces) != len(reference):
                continue
            if any(t["gts"] != r["reward_model"]["ground_truth"] for t, r in zip(traces, reference, strict=True)):
                raise ValueError(f"Validation/reference order mismatch: {path}")
            identity = {
                "trace_sha256": file_digest(path),
                "dataset_sha256": file_digest(args.test),
                "reward_sha256": file_digest(args.reward),
                "judge_model": judge_args.judge_model,
            }
            meta_path = args.output / (path.stem + ".manifest.json")
            if meta_path.exists() and json.loads(meta_path.read_text()) != identity:
                raise ValueError("A partially graded validation input changed")
            write_json(meta_path, identity)
            graded_path = args.output / (path.stem + ".graded.jsonl")
            existing = read_rows(graded_path)
            done = {r["hadm_id"] for r in existing}
            if len(done) != len(existing):
                raise ValueError("Duplicate resumed grades")
            semaphore = asyncio.Semaphore(32)
            errors = []

            async def one(trace, entry, *, done=done, semaphore=semaphore, graded_path=graded_path, errors=errors):
                hadm_id = entry["extra_info"]["hadm_id"]
                if hadm_id in done:
                    return
                async with semaphore:
                    try:
                        row = {
                            "hadm_id": hadm_id,
                            "ground_truth": trace["gts"],
                            "extracted_answer": trace.get("extracted_answer") or "",
                            "response": trace["output"],
                            "data_source": entry["data_source"],
                        }
                        row.update(await grade(row, entry, session, judge_args, reward))
                        with graded_path.open("a") as stream:
                            stream.write(json.dumps(row) + "\n")
                        done.add(hadm_id)
                    except Exception as exc:
                        errors.append({"hadm_id": hadm_id, "error": str(exc)[:500]})

            await asyncio.gather(*(one(t, r) for t, r in zip(traces, reference, strict=True)))
            write_json(args.output / (path.stem + ".errors.json"), errors)
            if errors:
                print(f"{path.stem}: {len(errors)} missing grades will retry", flush=True)
                continue
            graded = read_rows(graded_path)
            if len(graded) != 2452 or {r["hadm_id"] for r in graded} != {r["extra_info"]["hadm_id"] for r in reference}:
                raise ValueError("Incomplete grade coverage")
            counts, scores = Counter(), Counter()
            for row in graded:
                chapter = row["data_source"].split("/")[-1]
                counts[chapter] += 1
                scores[chapter] += row["judge_acc_lenient"]
            summary = {
                "model": "Qwen3.5-9B SFT + curated-data RL (self-judge)",
                "step": int(path.stem),
                "n": 2452,
                "overall": sum(scores.values()) / 2452,
                "per_cat": {k: scores[k] / counts[k] for k in counts},
                "cat_n": dict(counts),
                **identity,
            }
            write_json(summary_path, summary)
            print(json.dumps(summary), flush=True)
    summaries = [json.loads(p.read_text()) for p in args.output.glob("*.summary.json")]
    if summaries:
        write_json(
            args.output / "curve.json",
            {"points": sorted(summaries, key=lambda r: r["step"]), "best": max(summaries, key=lambda r: r["overall"])},
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test", type=Path, default=Path("/scratch/sheng/self_evolving/mimiciv_rare/test.jsonl"))
    parser.add_argument(
        "--reward",
        type=Path,
        default=Path("/scratch/sheng/self_evolving/verl/verl/utils/reward_score/self_evolving.py"),
    )
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "watch.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            try:
                asyncio.run(refresh(args))
            except Exception as exc:
                print(f"ERROR: {exc}", flush=True)
                if args.once:
                    raise
            if args.once:
                break
            time.sleep(120)


if __name__ == "__main__":
    main()
