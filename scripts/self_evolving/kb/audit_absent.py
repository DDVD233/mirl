"""Hand-audit of the ABSENT bucket: re-query live Milvus with criterion-targeted
queries (plain = shipped policy, raw = pure dense over all 57M rows).

Usage:
    python scripts/self_evolving/kb/audit_absent.py --idx 3,5,10 --out /tmp/a.txt
    python scripts/self_evolving/kb/audit_absent.py --idx-file sample_idx.json --out ...

Two queries per record: (1) the probe's own missing_fact sentence, (2) an extra
hand-written probe string supplied via --extra "idx=query;idx=query".
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kb.kbq import raw_search  # noqa: E402
from kb.retrieval import RetrieveConfig, local_search  # noqa: E402

CAT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe_out", "cat_ABSENT.json")


def run(query, k_plain=8, k_raw=12, chars=460):
    out = []
    try:
        ps, _ = local_search(query, top_k=k_plain, cfg=RetrieveConfig())
    except Exception as e:
        ps = []
        out.append(f"  PLAIN ERROR {e}")
    out.append(f"  -- PLAIN k={k_plain} ({len(ps)} hits)")
    for i, p in enumerate(ps):
        out.append(f"  P{i+1} [{p['source']} {p['score']:.3f}] " + " ".join(p["text"][:chars].split()))
    try:
        rs = raw_search(query, k_raw)
    except Exception as e:
        rs = []
        out.append(f"  RAW ERROR {e}")
    out.append(f"  -- RAW k={k_raw}")
    for i, p in enumerate(rs):
        out.append(f"  R{i+1} [{p['source']} {p['score']:.3f}] " + " ".join(p["text"][:chars].split()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx", default="")
    ap.add_argument("--idx-file", default="")
    ap.add_argument("--extra", default="", help="idx=query;;idx=query")
    ap.add_argument("--only-extra", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--kraw", type=int, default=12)
    ap.add_argument("--chars", type=int, default=460)
    args = ap.parse_args()

    d = json.load(open(CAT))
    if args.idx_file:
        idxs = json.load(open(args.idx_file))
    else:
        idxs = [int(x) for x in args.idx.split(",") if x.strip()]

    extra = {}
    for chunk in args.extra.split(";;"):
        if "=" in chunk:
            k, v = chunk.split("=", 1)
            extra.setdefault(int(k), []).append(v)

    lines = []
    for i in idxs:
        r = d[i]
        lines.append("=" * 100)
        lines.append(f"### {i} pts={r['points']} met={r['met_rate']} conf={r['confidence']}")
        lines.append("CRIT: " + " ".join((r["criterion_text"] or "")[:260].split()))
        lines.append("MISS: " + " ".join((r["missing_fact"] or "")[:300].split()))
        queries = [] if args.only_extra else [r["missing_fact"]]
        queries += extra.get(i, [])
        for q in queries:
            lines.append(f"  QUERY: {q[:180]}")
            lines += run(q, k_raw=args.kraw, chars=args.chars)
        print(f"done {i}", file=sys.stderr)

    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
