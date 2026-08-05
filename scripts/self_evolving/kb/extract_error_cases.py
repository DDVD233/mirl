"""Extract and structure the worst rollouts from a verl generation dump.

Turns a rollout/validation dump (written by trainer.rollout_data_dir /
trainer.validation_data_dir) into a compact JSON of failure cases with the four
things a post-mortem actually needs separated out:

    input      the clinician task
    retrieval  the passages the tool returned (with their source labels), since
               the graded phase-2 prompt carries them inline
    thinking   the model's reasoning
    output     the graded answer (everything after the last </think>)

plus the per-criterion rubric results when the reward wrote them. This exists
because reading the raw dump costs thousands of tokens per case and the four
dimensions are interleaved inside one prompt string.

    python scripts/self_evolving/kb/extract_error_cases.py \
        --dump  .../rollouts/<exp>/<step>.jsonl \
        --n 20 --out /tmp/error_cases.json
"""

import argparse
import json
import re

_PASSAGE_RE = re.compile(r"\[passage \d+ \| source=([a-z0-9_]+)\]\n(.*?)(?=\n\[passage \d+ \| source=|\Z)", re.DOTALL)
_TOOL_SPAN_RE = re.compile(r"<tool_call>.*?</tool_call>|<tool_response>.*?</tool_response>", re.DOTALL | re.I)
_REF_BLOCK = "Reference passages retrieved for this question:"
_HARD_ANSWER = "The search tool is now CLOSED"


def split_input(inp: str) -> tuple[str, list[dict]]:
    """Separate the clinician task from the injected retrieval block."""
    text = inp or ""
    passages = []
    idx = text.find(_REF_BLOCK)
    if idx != -1:
        head, tail = text[:idx], text[idx + len(_REF_BLOCK):]
        # The hard-answer instruction follows the passages; cut it off the tail
        # first or it gets absorbed into the final passage's text.
        k = tail.find(_HARD_ANSWER)
        if k != -1:
            tail = tail[:k]
        for src, body in _PASSAGE_RE.findall(tail):
            passages.append({"source": src, "text": body.strip()})
        if not passages:  # block present but unparsed — keep it raw for inspection
            passages.append({"source": "?", "text": tail.strip()[:2000]})
        text = head
    # Drop the trailing hard-answer instruction; it is identical on every case.
    j = text.find(_HARD_ANSWER)
    if j != -1:
        text = text[:j]
    return text.strip(), passages


def split_output(out: str) -> tuple[str, str]:
    """(thinking, graded answer) — mirrors the reward's _strip_thinking."""
    t = _TOOL_SPAN_RE.sub("", out or "")
    low = t.lower()
    close = low.rfind("</think>")
    if close != -1:
        think = t[:close]
        for tag in ("<think>", "</think>"):
            think = think.replace(tag, "")
        return think.strip(), t[close + len("</think>"):].strip()
    open_i = low.find("<think>")
    if open_i != -1:
        return t[open_i + len("<think>"):].strip(), ""
    return "", t.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--n", type=int, default=20, help="worst-N cases to extract")
    ap.add_argument("--max_score", type=float, default=None,
                    help="only cases at or below this score (default: take worst N)")
    ap.add_argument("--truncate", type=int, default=4000, help="max chars per field")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = []
    with open(args.dump, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    if not rows:
        raise SystemExit(f"no rows in {args.dump}")

    def score_of(r):
        for k in ("score", "acc_len_adj_signed", "reward"):
            v = r.get(k)
            if isinstance(v, (int, float)):
                return float(v)
        return 0.0

    rows.sort(key=score_of)
    picked = [r for r in rows if args.max_score is None or score_of(r) <= args.max_score][: args.n]

    cases = []
    for i, r in enumerate(picked):
        task, passages = split_input(r.get("input", ""))
        think, answer = split_output(r.get("output", ""))
        case = {
            "case_id": i,
            "score": round(score_of(r), 4),
            "retrieved": bool(passages),
            "n_passages": len(passages),
            "input": task[: args.truncate],
            "retrieval": [{"source": p["source"], "text": p["text"][: args.truncate // 2]} for p in passages],
            "thinking": think[: args.truncate],
            "output": answer[: args.truncate],
            "think_chars": len(think),
            "answer_chars": len(answer),
        }
        for k in ("rubric_met", "acc_raw", "acc_len_adj", "acc_len_adj_signed",
                  "format_ok", "think_closed", "use_case", "question_id"):
            if k in r:
                case[k] = r[k]
        cases.append(case)

    all_scores = [score_of(r) for r in rows]
    summary = {
        "dump": args.dump,
        "n_rows": len(rows),
        "mean_score": round(sum(all_scores) / len(all_scores), 4),
        "n_extracted": len(cases),
        "extracted_score_range": [cases[0]["score"], cases[-1]["score"]] if cases else [],
        "retrieval_rate_all": round(
            sum(1 for r in rows if _REF_BLOCK in (r.get("input") or "")) / len(rows), 4),
        "retrieval_rate_failures": round(
            sum(1 for c in cases if c["retrieved"]) / max(len(cases), 1), 4),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "cases": cases}, f, indent=1)
    print(json.dumps(summary, indent=1))
    print(f"wrote {len(cases)} cases -> {args.out}")


if __name__ == "__main__":
    main()
