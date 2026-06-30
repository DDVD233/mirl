#!/usr/bin/env python3
"""Offline harness to design / validate the reward-evolution meta-prompts on real data.

It exercises the *production* evolution path (`reward_evolution.evolve_once`) against a
previous training/val dump and a live judge (the gpu5 teacher), so we can confirm BEFORE
wiring anything into training that the judge:
  (a) emits a well-formed <prompt>...</prompt> that keeps the \\boxed{0-10} contract, and
  (b) writes a non-trivial, COMPILING `function_reward` that scores responses sensibly.

Run on a node that can reach the judge endpoint and the dump (e.g. remote node 2333):

    CHAT_PROVIDER=vllm python scripts/self_evolving/eval/test_reward_evolution.py \
        --dump /scratch/sheng/self_evolving/logs_si_qwen36_kl/val_generations/mimiciv_rare_qwen36_27b_selfimprove_kl/20.jsonl \
        --api-base http://point.dd.works:18184/v1 \
        --api-key-file /scratch/sheng/self_evolving/.climb_teacher_key \
        --model Qwen/Qwen3.6-27B \
        --n-examples 6 --rounds 2 --verbose

The dump rows carry: input, output, gts, score, acc, judge_acc_*, answer_quality,
reasoning_quality, format_ok, embed_sim, char_bleu, extracted_answer.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile

# Make `verl` importable when run from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from verl.utils.reward_score import reward_evolution as R  # noqa: E402


def _split_question(input_text: str) -> str:
    """Drop the system preamble; keep the user clinical case if delimited."""
    if not input_text:
        return ""
    for sep in ("\nuser\n", "\n\nuser\n", "<|im_start|>user\n"):
        if sep in input_text:
            return input_text.split(sep, 1)[1].strip()
    return input_text.strip()


def load_examples(dump_path: str, limit: int | None = None) -> list[dict]:
    examples = []
    with open(dump_path) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            question = _split_question(row.get("input", ""))
            response = row.get("output", "")
            gt = row.get("gts", "") or row.get("ground_truth", "")
            if not response or not gt:
                continue
            examples.append(R.make_example(question, response, gt, row))
    return examples


def _hr(title: str) -> None:
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")


async def run_round(args, store: R.EvolutionStore, examples: list[dict], step: int) -> dict:
    sel = R.select_contrastive(examples, args.n_examples)
    _hr(f"ROUND {step}: evolving from current artifacts using {len(sel)} contrastive examples")
    cur_prompt, cur_fn = store.read_current()
    print(f"[current judge prompt] {len(cur_prompt)} chars; [current fn] {len(cur_fn)} chars")
    print("[example combined rewards fed in]:",
          [round(float(e["combined_reward"]), 3) for e in sel])

    metrics = await R.evolve_once(
        api_base=args.api_base,
        api_key=args.api_key,
        model_name=args.model,
        evolve_dir=store.root,
        examples=sel,
        step=step,
        design_max_tokens=args.max_tokens,
    )
    _hr(f"ROUND {step}: evolution metrics")
    print(json.dumps(metrics, indent=2))

    new_prompt, new_fn = store.read_current()
    _hr(f"ROUND {step}: NEW judge prompt (current/)")
    print(new_prompt)
    _hr(f"ROUND {step}: NEW function_reward (current/)")
    print(new_fn)

    # Show the function's output on each selected example + the judge_reward.
    _hr(f"ROUND {step}: scoring selected examples with NEW artifacts")
    fn = R.load_function(new_fn)
    rows = []
    judge_tasks = [
        R.score_with_judge_prompt(
            args.api_base, args.api_key, args.model, new_prompt,
            e["question"], e["response"], e["ground_truth"],
        )
        for e in sel
    ]
    judge_scores = await asyncio.gather(*judge_tasks)
    for e, js in zip(sel, judge_scores):
        f_out = R.safe_call_function(fn, e["question"], e["response"], e["ground_truth"])
        rows.append((e.get("sub_rewards", {}).get("acc", "?"),
                     round(float(e["combined_reward"]), 3),
                     e.get("extracted_answer", "")[:24], e["ground_truth"][:32],
                     round(js, 3), round(f_out, 3)))
    print(f"{'acc':>4} {'oldR':>6} {'extracted':>24} {'ground_truth':>32} {'judgeR':>7} {'funcR':>6}")
    for r in rows:
        print(f"{str(r[0]):>4} {r[1]:>6} {r[2]:>24} {r[3]:>32} {r[4]:>7} {r[5]:>6}")

    return metrics


async def amain(args) -> None:
    examples = load_examples(args.dump, limit=args.max_rows)
    if not examples:
        print(f"No usable examples in {args.dump}", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(examples)} examples from {args.dump}")
    print(f"Judge: {args.model} @ {args.api_base}  (CHAT_PROVIDER={os.environ.get('CHAT_PROVIDER', 'vllm')})")

    tmpdir = args.evolve_dir or tempfile.mkdtemp(prefix="reward_evolve_test_")
    store = R.EvolutionStore(tmpdir)
    store.init_if_needed()
    print(f"Evolution store: {tmpdir}")

    # Sanity: does the STARTER judge prompt already score sensibly (correlate with acc)?
    _hr("BASELINE: starter judge prompt scores vs. exact-match acc")
    sel0 = R.select_contrastive(examples, args.n_examples)
    base_scores = await asyncio.gather(*[
        R.score_with_judge_prompt(args.api_base, args.api_key, args.model, R.STARTER_JUDGE_PROMPT,
                                   e["question"], e["response"], e["ground_truth"])
        for e in sel0
    ])
    for e, s in zip(sel0, base_scores):
        print(f"  acc={e.get('sub_rewards', {}).get('acc', '?')}  judgeR={s:.3f}  "
              f"extracted={e.get('extracted_answer', '')[:24]!r}  gt={e['ground_truth'][:32]!r}")

    for step in range(1, args.rounds + 1):
        await run_round(args, store, examples, step)

    _hr("DONE")
    print(f"Artifacts + per-step history under: {tmpdir}")
    print("  step_000 = starters; step_NNN = each round; current/ = latest valid")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", required=True, help="path to a val/rollout dump jsonl")
    ap.add_argument("--api-base", default=os.environ.get("TEACHER_BASE", "http://point.dd.works:18184/v1"))
    ap.add_argument("--api-key", default=os.environ.get("JUDGE_KEY", ""))
    ap.add_argument("--api-key-file", default="/scratch/sheng/self_evolving/.climb_teacher_key")
    ap.add_argument("--model", default=os.environ.get("JUDGE_MODEL", "Qwen/Qwen3.6-27B"))
    ap.add_argument("--n-examples", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=8000)
    ap.add_argument("--max-rows", type=int, default=400, help="cap rows read from the dump")
    ap.add_argument("--evolve-dir", default="", help="persist the store here (default: tmp)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if not args.api_key and args.api_key_file and os.path.exists(args.api_key_file):
        with open(args.api_key_file) as f:
            args.api_key = f.read().strip()
    if not args.api_key:
        args.api_key = "EMPTY"
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
