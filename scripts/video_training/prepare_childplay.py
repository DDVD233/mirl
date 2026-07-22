"""One-click ChildPlay ADOS-2 dataset preparation for video SFT + GRPO training.

Copies the raw dataset (video chunks + per-chunk ADOS annotation JSONs + split
definition) from the source directory to a writable data directory, then builds:

  - RL JSONL (tactile schema): one row per (chunk, ADOS item) with an integer
    score in the item's rubric label space (score 8 / "Cannot assess from video"
    dropped, mirroring the upstream ChildPlay pipeline)
  - SFT parquet: same prompts with the annotation justification as the
    assistant target ("Reasoning: <justification>\nScore: [x]")
  - item-stratified mini val JSONL and tiny smoke train/val JSONL

Video paths in all outputs are relative ("chunks/<file>.mp4"); RLHFDataset and
MultiTurnSFTDataset resolve them against the JSONL/parquet's own directory, so
the same files work on the host and inside docker as long as the data dir is
mounted at the same path.

Usage:
  python scripts/video_training/prepare_childplay.py \
      --src /scratch/emorfin/childplay_dataset \
      --out /scratch/dvdai/childplay_dataset
"""

import argparse
import collections
import json
import random
import re
import subprocess
import sys
from pathlib import Path

DATA_SOURCE = "childplay_ados"
YTID_RE = re.compile(r"\[([^\]]+)\]")

SYSTEM_PROMPT = (
    "You are an expert clinician scoring items from the ADOS-2 (Autism Diagnostic "
    "Observation Schedule, Second Edition), Module 1. Watch the video of the child "
    "carefully and score the requested item based only on what is observable in the "
    "video. First give a brief justification grounded in the child's observed "
    "behavior, then give the score. You MUST answer in exactly this format:\n"
    "Reasoning: <your justification>\n"
    "Score: [x]\n"
    "where x is one of the allowed score labels for the item."
)

USER_TEMPLATE = (
    "<video>\n"
    "Score the following ADOS-2 Module 1 item for the child in this video.\n\n"
    "Module: {module}\n"
    "Item: {item_title}\n"
    "Description: {description}\n\n"
    "Score options:\n{options}\n\n"
    "Answer in exactly this format:\n"
    "Reasoning: <your justification>\n"
    "Score: [x]"
)


def load_rubric(path):
    """Return {code: {module, title, description, labels: [(label, text), ...]}}."""
    items = {}
    for it in json.loads(Path(path).read_text()):
        code = it["test type"].split(".")[0].strip()
        items[code] = {
            "module": it["module"],
            "title": it["test type"],
            "description": it["description"],
            # 8 = "not assessable in this context" — never a training target and
            # not a meaningful choice for a 1-minute clip, so it is not offered.
            "labels": [(l, t) for l, t in it["labels"] if str(l) != "8"],
        }
    return items


def rsync_raw(src, out):
    out.mkdir(parents=True, exist_ok=True)
    for rel in ["chunks", "annotations"]:
        print(f"[prep] rsync {src / rel}/ -> {out / rel}/")
        subprocess.run(
            ["rsync", "-a", "--info=stats1", f"{src / rel}/", f"{out / rel}/"],
            check=True,
        )
    (out / "splits").mkdir(exist_ok=True)
    subprocess.run(
        ["rsync", "-a", str(src / "splits" / "split_groups.json"), str(out / "splits/")],
        check=True,
    )


def build_rows(out, rubric):
    split_groups = json.loads((out / "splits" / "split_groups.json").read_text())
    group_of = {g: "train" for g in split_groups["train_groups"]}
    group_of.update({g: "val" for g in split_groups["val_groups"]})

    rows = {"train": [], "val": []}
    stats = {
        "chunks_seen": 0,
        "chunks_missing_video": 0,
        "chunks_unknown_group": 0,
        "dropped_scores": collections.Counter(),
        "kept_per_item": collections.Counter(),
        "score_hist": collections.Counter(),
    }

    for ann_path in sorted((out / "annotations").glob("*_ados.json")):
        ann = json.loads(ann_path.read_text())
        video_rel = Path("chunks") / (ann["video_file"] + ".mp4")
        stats["chunks_seen"] += 1
        if not (out / video_rel).exists():
            print(f"[prep] WARNING: missing video for {ann_path.name}, skipping")
            stats["chunks_missing_video"] += 1
            continue
        m = YTID_RE.search(ann["video_file"])
        split = group_of.get(m.group(1)) if m else None
        if split is None:
            print(f"[prep] WARNING: no split group for {ann_path.name}, skipping")
            stats["chunks_unknown_group"] += 1
            continue

        scores = ann.get("ados_analysis", {}).get("scores", {})
        for code, entry in scores.items():
            item = rubric.get(code)
            if item is None:
                stats["dropped_scores"][f"unknown_item:{code}"] += 1
                continue
            score = entry.get("score")
            allowed = {l for l, _ in item["labels"]}
            if not isinstance(score, int) or str(score) not in allowed:
                stats["dropped_scores"][str(score)] += 1
                continue

            options = "\n".join(f"{l} = {t}" for l, t in item["labels"])
            user = USER_TEMPLATE.format(
                module=item["module"],
                item_title=item["title"],
                description=item["description"],
                options=options,
            )
            justification = (entry.get("justification") or "").strip()
            rows[split].append(
                {
                    "data_source": DATA_SOURCE,
                    "prompt": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user},
                    ],
                    "videos": [{"video": str(video_rel), "min_frames": 4, "max_frames": None}],
                    "reward_model": {"style": "rule", "ground_truth": str(score)},
                    "extra_info": {
                        "split": split,
                        "item": code,
                        "ytid": m.group(1),
                        "chunk_file": str(video_rel),
                        "confidence": entry.get("confidence"),
                        "justification": justification,
                        "answer": str(score),
                    },
                }
            )
            stats["kept_per_item"][code] += 1
            stats["score_hist"][str(score)] += 1

    return rows, stats


def set_frames(rows, max_frames):
    for r in rows:
        for v in r["videos"]:
            v["max_frames"] = max_frames
    return rows


def to_sft_record(row):
    just = row["extra_info"]["justification"] or "Based on the observable behavior in the video."
    answer = row["extra_info"]["answer"]
    assistant = f"Reasoning: {just}\nScore: [{answer}]"
    return {
        "messages": list(row["prompt"]) + [{"role": "assistant", "content": assistant}],
        "videos": [dict(v) for v in row["videos"]],
    }


def stratified_sample(rows, n, rng):
    """Sample ~n rows spread across ADOS items."""
    by_item = collections.defaultdict(list)
    for r in rows:
        by_item[r["extra_info"]["item"]].append(r)
    for v in by_item.values():
        rng.shuffle(v)
    picked, idx = [], 0
    while len(picked) < min(n, len(rows)):
        for item in sorted(by_item):
            if idx < len(by_item[item]):
                picked.append(by_item[item][idx])
                if len(picked) >= n:
                    break
        idx += 1
        if idx > max(len(v) for v in by_item.values()):
            break
    return picked


def write_jsonl(rows, path):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[prep] wrote {path} ({len(rows)} rows)")


def write_parquet(records, path):
    import pandas as pd

    pd.DataFrame(records).to_parquet(str(path), index=False)
    print(f"[prep] wrote {path} ({len(records)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/scratch/emorfin/childplay_dataset")
    ap.add_argument("--out", default="/scratch/dvdai/childplay_dataset")
    ap.add_argument("--max-frames", type=int, default=16)
    ap.add_argument("--smoke-max-frames", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--balance-cap",
        type=int,
        default=0,
        help="Optional per-(item,score) cap on train rows (score 0 dominates); 0 = off",
    )
    ap.add_argument("--skip-rsync", action="store_true", help="Assume raw data already copied")
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    rng = random.Random(args.seed)
    rubric = load_rubric(Path(__file__).parent / "ados2_module1.json")

    if not args.skip_rsync:
        rsync_raw(src, out)

    rows, stats = build_rows(out, rubric)
    for split in ("train", "val"):
        rng.shuffle(rows[split])
        set_frames(rows[split], args.max_frames)

    if args.balance_cap > 0:
        seen = collections.Counter()
        capped = []
        for r in rows["train"]:
            key = (r["extra_info"]["item"], r["extra_info"]["answer"])
            if seen[key] < args.balance_cap:
                seen[key] += 1
                capped.append(r)
        print(f"[prep] balance cap {args.balance_cap}: train {len(rows['train'])} -> {len(capped)}")
        rows["train"] = capped

    for i, split_rows in enumerate([rows["train"], rows["val"]]):
        for j, r in enumerate(split_rows):
            r["extra_info"]["index"] = j

    write_jsonl(rows["train"], out / "childplay_ados_train.jsonl")
    write_jsonl(rows["val"], out / "childplay_ados_val.jsonl")

    val_mini = stratified_sample(rows["val"], 256, rng)
    write_jsonl(val_mini, out / "childplay_ados_val_mini.jsonl")

    smoke_train = set_frames(
        [json.loads(json.dumps(r)) for r in stratified_sample(rows["train"], 64, rng)],
        args.smoke_max_frames,
    )
    smoke_val = set_frames(
        [json.loads(json.dumps(r)) for r in stratified_sample(rows["val"], 32, rng)],
        args.smoke_max_frames,
    )
    write_jsonl(smoke_train, out / "childplay_ados_smoke_train.jsonl")
    write_jsonl(smoke_val, out / "childplay_ados_smoke_val.jsonl")

    write_parquet([to_sft_record(r) for r in rows["train"]], out / "childplay_ados_sft_train.parquet")
    write_parquet([to_sft_record(r) for r in rows["val"]], out / "childplay_ados_sft_val.parquet")

    stats_out = {
        "train_rows": len(rows["train"]),
        "val_rows": len(rows["val"]),
        "chunks_seen": stats["chunks_seen"],
        "chunks_missing_video": stats["chunks_missing_video"],
        "chunks_unknown_group": stats["chunks_unknown_group"],
        "dropped_scores": dict(stats["dropped_scores"]),
        "score_hist": dict(stats["score_hist"]),
        "kept_per_item": dict(stats["kept_per_item"]),
        "max_frames": args.max_frames,
        "seed": args.seed,
        "balance_cap": args.balance_cap,
    }
    (out / "stats.json").write_text(json.dumps(stats_out, indent=2))
    print(f"[prep] stats: train={stats_out['train_rows']} val={stats_out['val_rows']} "
          f"dropped={stats_out['dropped_scores']}")
    print(f"[prep] done -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
