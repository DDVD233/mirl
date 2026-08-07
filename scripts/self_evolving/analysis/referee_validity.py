"""How often is the REFEREE wrong? Measured against physician-written rubrics. Zero GPU.

H (see verl/utils/reward_score/spec_gap.py) is the disagreement between a generated
rubric and a rubric-blind referee. On its own it cannot say WHICH of the two is wrong,
so an H of 0.35 is uninterpretable until the referee's own error rate is known.

This measures it. The 525 HealthBench-Professional tasks come with rubrics written by
physicians, so on those the rubric is gold BY CONSTRUCTION and any referee
disagreement is referee error. Same computation as the training-time statistic,
opposite reading — and the asymmetry is licensed by who wrote the rubric, not by
anything about the models.

The material is free: saved validation dumps already carry each answer AND its
official per-criterion grading, and different steps produced different answers to the
same 525 questions. So one question's answers across several dumps form a group the
referee can rank, exactly as it ranks a GRPO group during training.

THE LENGTH TRAP, which inverts the naive check. The official metric subtracts 0.0147
per 500 characters over a 2000-character centre, and these runs' answers ran to 9000+
characters, so on the official score length is a strong NEGATIVE predictor. A referee
that did nothing but prefer shorter answers would therefore look like it AGREES with
the physicians. Raw agreement is descriptive only; the number that matters is
agreement on pairs of SIMILAR length, where that shortcut is unavailable.

Two comparisons, because they answer different questions:
  vs acc_raw            the rubric's content judgement, no length term. This is the
                        fair test of the referee, which is told to ignore length.
  vs acc_len_adj_signed the official reported metric, length term included.

Usage:
  python scripts/self_evolving/analysis/referee_validity.py \
      --dumps <val_generations/exp/0.jsonl,.../20.jsonl,.../60.jsonl> \
      --referee-base $TRAPI_BASE --referee-key $(cat $S/.trapi_key) \
      --referee-model gpt-chat-latest_2026-05-28 --provider trapi \
      --items 120 --out $S/referee_validity.json
"""

import argparse
import itertools
import json
import os
import random
import re
import statistics
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from verl.utils.reward_score.spec_gap import measure_groups_sync  # noqa: E402


def _strip_thinking(text: str) -> str:
    t = re.sub(r"<tool_call>.*?(?:</tool_call>|\Z)|<tool_response>.*?(?:</tool_response>|\Z)",
               "", text or "", flags=re.S)
    if "</think>" in t:
        return t.rsplit("</think>", 1)[-1].strip()
    return "" if "<think>" in t else t.strip()


def _last_user_turn(prompt_text: str) -> str:
    parts = re.split(r"<\|im_start\|>|<\|start_header_id\|>", prompt_text or "")
    turns = [p for p in parts if p.strip().startswith("user")]
    if turns:
        body = re.sub(r"^user\s*(<\|im_sep\|>|<\|end_header_id\|>)?", "", turns[-1].strip())
        return re.sub(r"<\|im_end\|>.*$", "", body, flags=re.S).strip()[:8000]
    return (prompt_text or "")[:8000]


def load(paths: list) -> dict:
    """val item index -> [{answer, acc_raw, acc_la, chars, tag}, ...].

    Dumps are index-aligned to the val parquet, so row i of every dump is the same
    physician-written task. Answers differ because the policy differed.
    """
    per_item: dict = {}
    task_of: dict = {}
    for p in paths:
        tag = f"{os.path.basename(os.path.dirname(p))}@{os.path.basename(p).split('.')[0]}"
        for i, line in enumerate(open(p)):
            if not line.strip():
                continue
            r = json.loads(line)
            ans = _strip_thinking(r.get("output", ""))
            if len(ans.strip()) < 64:
                continue
            task_of.setdefault(i, _last_user_turn(r.get("input", "")))
            per_item.setdefault(i, []).append({
                "answer": ans, "chars": len(ans), "tag": tag,
                "acc_raw": float(r.get("acc_raw", 0.0)),
                "acc_la": float(r.get("acc_len_adj_signed", r.get("acc_raw", 0.0))),
            })
    return per_item, task_of


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", required=True, help="comma-separated val dump jsonls")
    ap.add_argument("--referee-base", required=True)
    ap.add_argument("--referee-key", default="")
    ap.add_argument("--referee-model", required=True)
    ap.add_argument("--provider", default="trapi")
    ap.add_argument("--items", type=int, default=120, help="val items to sample")
    ap.add_argument("--margin", type=float, default=0.15,
                    help="min |delta| on the gold score for a pair to be decisive")
    ap.add_argument("--matched-length-frac", type=float, default=0.15,
                    help="pairs within this relative char difference are length-matched")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    paths = [p.strip() for p in a.dumps.split(",") if p.strip() and os.path.exists(p.strip())]
    if len(paths) < 2:
        raise SystemExit("need >= 2 existing dumps so one task has several answers")
    per_item, task_of = load(paths)
    usable = [i for i, v in per_item.items() if len(v) >= 2]
    random.Random(a.seed).shuffle(usable)
    usable = usable[:a.items]
    print(f"{len(paths)} dumps -> {len(per_item)} items, {len(usable)} sampled "
          f"({statistics.mean(len(per_item[i]) for i in usable):.1f} answers/item)", flush=True)

    # The referee ranks each item's answers, blind to the rubric -- the same call the
    # trainer makes on a GRPO group. Scores passed in are the OFFICIAL content score,
    # so `group_stats` internals are unused here; we only want the tier verdicts.
    payload = []
    for i in usable:
        rows = per_item[i]
        payload.append({
            "uid": f"item{i}", "task": task_of[i], "rows": list(range(len(rows))),
            "answers": {j: r["answer"] for j, r in enumerate(rows)},
            "scores": {j: r["acc_raw"] for j, r in enumerate(rows)},
            "lens": {j: r["chars"] for j, r in enumerate(rows)},
        })
    stats = measure_groups_sync(
        payload, a.referee_base, a.referee_key, a.referee_model, provider=a.provider,
        concurrency=a.concurrency, swap=True, margin=0.0, min_pairs=1, step=a.seed,
        timeout_s=180)

    # Pair-level accounting against the physician rubric.
    tot = {k: 0 for k in ("pairs", "decisive", "stable", "err", "matched", "matched_err",
                          "ref_longer", "ref_first", "conflict", "sides_gold", "unjudged")}
    per_item_err = []
    for i in usable:
        st = stats.get(f"item{i}")
        rows = per_item[i]
        if st is None or not st.judged or not st.tier_of:
            tot["unjudged"] += 1
            continue
        n_dec = n_err = 0
        tot["pairs"] += len(rows) * (len(rows) - 1) // 2
        # stable_pairs() applies the same referee tie + swap-consistency filter H
        # uses, so e and H are measured on the same population and the debias
        # arithmetic between them is valid.
        for x, y, ref_x_better in st.stable_pairs():
            d_raw = rows[x]["acc_raw"] - rows[y]["acc_raw"]
            if abs(d_raw) < a.margin:
                continue          # physicians' rubric does not separate them
            tot["decisive"] += 1
            n_dec += 1
            gold_x_better = d_raw > 0
            wrong = ref_x_better != gold_x_better
            if wrong:
                tot["err"] += 1
                n_err += 1
            # Length diagnostics, on every decisive pair.
            cx, cy = rows[x]["chars"], rows[y]["chars"]
            if (cx > cy) == ref_x_better:
                tot["ref_longer"] += 1
            # Length-matched subset: the shortcut of "prefer shorter" is unavailable.
            if abs(cx - cy) <= a.matched_length_frac * max(cx, cy):
                tot["matched"] += 1
                if wrong:
                    tot["matched_err"] += 1
            # Three-way witness: where the content score and the official
            # length-adjusted score disagree, which one does the referee back?
            d_la = rows[x]["acc_la"] - rows[y]["acc_la"]
            if d_raw * d_la < 0 and abs(d_la) >= a.margin:
                tot["conflict"] += 1
                if ref_x_better == (d_la > 0):
                    tot["sides_gold"] += 1
        if n_dec:
            per_item_err.append(n_err / n_dec)

    d, m = tot["decisive"], tot["matched"]
    e = tot["err"] / d if d else float("nan")
    e_matched = tot["matched_err"] / m if m else float("nan")
    res = {
        "dumps": paths, "n_items": len(usable), "margin": a.margin,
        "n_pairs_total": tot["pairs"], "n_decisive": d, "n_unjudged_items": tot["unjudged"],
        "referee_error_rate": e,
        "referee_error_rate_matched_length": e_matched,
        "n_length_matched": m,
        "length_pref": tot["ref_longer"] / d if d else float("nan"),
        "sides_with_official_on_conflict": (tot["sides_gold"] / tot["conflict"]
                                           if tot["conflict"] else float("nan")),
        "n_conflict": tot["conflict"],
        "per_item_error_sd": statistics.pstdev(per_item_err) if len(per_item_err) > 1 else 0.0,
    }

    print("\n=== referee validity vs PHYSICIAN-written rubrics ===")
    print(f"  decisive pairs                  {d}  (of {tot['pairs']} possible; "
          f"{tot['unjudged']} items got no usable verdict)")
    print(f"  REFEREE ERROR RATE  e =         {e:.3f}"
          "        <-- the number H has to be read against")
    print(f"  ... on length-MATCHED pairs     {e_matched:.3f}  (n={m})"
          "   <-- the honest one; see the length trap")
    print(f"  referee prefers the longer       {res['length_pref']:.3f}"
          "        [0.5 = indifferent]")
    if tot["conflict"]:
        print(f"  content vs official-metric conflicts: {tot['conflict']} pairs, referee "
              f"sides with the official metric {res['sides_with_official_on_conflict']:.2f}")
    print("\n  READING IT:")
    if e_matched != e_matched:  # nan
        print("    not enough length-matched pairs to conclude; widen --matched-length-frac")
    elif e_matched >= 0.30:
        print(f"    e={e_matched:.2f} is close to the H={0.35:.2f} we measured on GENERATED")
        print("    rubrics, so that H is mostly this referee being wrong. The measurement")
        print("    does not support a claim about the generated rubrics. STOP and fix the")
        print("    referee before spending GPU on it.")
    elif e_matched <= 0.15:
        print(f"    e={e_matched:.2f} is well below the H~0.35 on generated rubrics, so most")
        print("    of that H is real specification failure, not referee error. Debiased:")
        print(f"    H_true ~ (0.35 - {e_matched:.2f}) / (1 - 2*{e_matched:.2f}) = "
              f"{(0.35 - e_matched) / (1 - 2 * e_matched):.2f}")
    else:
        print(f"    e={e_matched:.2f} is middling: real signal in H, but the confidence")
        print(f"    interval widens by 1/(1-2e) = {1 / (1 - 2 * e_matched):.2f}x. Usable,")
        print("    and the error rate must be reported alongside every H.")

    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
