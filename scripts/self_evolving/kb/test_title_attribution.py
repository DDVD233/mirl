#!/usr/bin/env python3
"""End-to-end check: does a labelled passage actually produce an attributed brief?

Three things have to hold for the title join to be worth anything, and only the first two
are covered by unit tests:

  1. attach_titles() puts the title in the passage header.
  2. format_passages() renders it where the summarizer is told to look.
  3. the SUMMARIZER ACTUALLY CARRIES IT into the evidence brief.

(3) is the one that can silently fail. The brief is written by a 9B under a 400-word cap
with eight competing rules, and before this change the rules said nothing about attribution
while the [p3] tag replaced provenance with an index. A prompt edit is a request, not a
guarantee, so it has to be measured against a control.

So this runs the REAL pipeline -- real embed, real Milvus, real ranking, the real
SUMMARY_SYSTEM parsed out of generation_server.py -- twice on identical passages: once with
titles attached and once without. If the titled brief names sources and the untitled one
does not, the mechanism works. If neither does, the summarizer prompt needs another pass and
the title join is inert.

Titles come from the same MedRAG chunk files the real build reads, fetched for only the
handful of ids this query returns, so the test needs no completed database.

  python3 test_title_attribution.py
  python3 test_title_attribution.py --question "..." --summarizer http://host:port/v1
"""

import argparse
import ast
import json
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
GEN_SERVER = os.path.join(HERE, "..", "generation_server.py")
CHUNK_BASE = "https://huggingface.co/datasets/MedRAG/pubmed/resolve/main/chunk"

DEFAULT_Q = ("A patient has recurrent bleeding after endoscopic treatment of a peptic "
             "ulcer. What does the evidence say about high-dose intravenous omeprazole, "
             "and which guideline or trial supports it?")


def summary_prompt() -> str:
    """Read SUMMARY_SYSTEM out of the real server with ast, not by importing it.

    Importing generation_server pulls fastapi/pymilvus/httpx and builds app state; parsing
    is both cheaper and guarantees we are testing the deployed literal rather than a copy
    retyped into this file, which is how a test starts passing against its own fiction.
    """
    tree = ast.parse(open(GEN_SERVER).read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", "") == "SUMMARY_SYSTEM" for t in node.targets):
            return ast.literal_eval(node.value)
    raise SystemExit("FATAL: SUMMARY_SYSTEM not found in generation_server.py")


def summary_user(question: str, passages_block: str) -> str:
    tree = ast.parse(open(GEN_SERVER).read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", "") == "SUMMARY_USER" for t in node.targets):
            tmpl = ast.literal_eval(node.value)
            return tmpl.format(question=question, passages=passages_block)
    raise SystemExit("FATAL: SUMMARY_USER not found")


def fetch_titles(entry_ids: list[str]) -> dict:
    """Titles for these ids only, from the chunk files that contain them."""
    need = {}
    for eid in entry_ids:
        m = re.match(r"(pubmed23n\d+)_\d+$", eid or "")
        if m:
            need.setdefault(m.group(1) + ".jsonl", set()).add(eid)
    out = {}
    for fname, ids in need.items():
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=True) as tf:
            try:
                subprocess.run(["curl", "-sSL", "--retry", "2", "-m", "300",
                                "-o", tf.name, f"{CHUNK_BASE}/{fname}"], check=True)
            except subprocess.CalledProcessError as e:
                print(f"  ! could not fetch {fname}: {e}")
                continue
            with open(tf.name, errors="ignore") as fh:
                for line in fh:
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if d.get("id") in ids and (d.get("title") or "").strip():
                        out[d["id"]] = (d["title"].strip(), str(d.get("PMID") or ""))
    return out


def call_summarizer(base: str, model: str, system: str, user: str) -> str:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "max_tokens": 1024, "temperature": 0.2,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(f"{base.rstrip('/')}/chat/completions", data=body,
                                headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"] or ""


# Does the brief attribute anything? Deliberately generous: any issuing body, any journal,
# any "per/according to <Proper Noun>", or a quoted title fragment counts.
ATTRIB = re.compile(
    r"\b(per the|according to|guideline|American \w+|European \w+|National \w+|"
    r"World Health|ACC|AHA|ESC|NICE|WHO|USPSTF|IDSA|ASCO|NCCN|AUA|ACG|KDIGO|"
    r"NEJM|New England|Lancet|JAMA|BMJ|PMID|et al)\b", re.I)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", default=DEFAULT_Q)
    ap.add_argument("--summarizer", default="http://point.dd.works:18186/v1")
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--milvus", default="http://mib.media.mit.edu:19531")
    ap.add_argument("--embed", default="http://mib.media.mit.edu:18001/v1")
    ap.add_argument("--top_k", type=int, default=8)
    a = ap.parse_args()

    import retrieval as R

    print("=== 1. real retrieval (embed + Milvus + source-aware ranking) ===")
    passages, _ = R.local_search(a.question, top_k=a.top_k, milvus_uri=a.milvus,
                                 embed_api_base=a.embed)
    if not passages:
        print("FATAL: retrieval returned nothing"); return 1
    print(f"  {len(passages)} passages; sources: "
          f"{sorted({p['source'] for p in passages})}")

    print("\n=== 2. titles for exactly these ids, from the same chunk files ===")
    got = fetch_titles([p.get("entry_id", "") for p in passages])
    print(f"  resolved {len(got)} titles for "
          f"{sum(1 for p in passages if str(p.get('entry_id','')).startswith('pubmed'))} "
          f"pubmed passages")
    for eid, (t, pmid) in list(got.items())[:3]:
        print(f"    {eid}  pmid={pmid}  {t[:90]!r}")

    db = os.path.join(tempfile.mkdtemp(), "t.sqlite")
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE titles (id TEXT PRIMARY KEY, title TEXT, pmid TEXT) "
                "WITHOUT ROWID")
    con.executemany("INSERT INTO titles VALUES (?,?,?)",
                    [(k, v[0], v[1]) for k, v in got.items()])
    con.commit(); con.close()

    print("\n=== 3. format WITHOUT titles (control) and WITH titles ===")
    import copy
    plain = copy.deepcopy(passages)
    block_plain = R.format_passages(plain)

    os.environ["PUBMED_TITLES_DB"] = db
    import importlib
    R = importlib.reload(R)
    titled = copy.deepcopy(passages)
    n = R.attach_titles(titled)
    block_titled = R.format_passages(titled)
    print(f"  attach_titles labelled {n}/{len(titled)} passages")
    print(f"  control header: {block_plain.splitlines()[0][:100]}")
    print(f"  titled  header: {block_titled.splitlines()[0][:130]}")
    if n == 0:
        print("FATAL: no passage got a title; the join is not working"); return 1
    if "title=" not in block_titled:
        print("FATAL: title missing from the formatted block"); return 1

    sysmsg = summary_prompt()
    has_rule = "SOURCE ATTRIBUTION" in sysmsg
    print(f"\n=== 4. SUMMARY_SYSTEM carries the attribution rule? {has_rule} ===")
    if not has_rule:
        print("FATAL: the deployed prompt has no attribution rule"); return 1

    print("\n=== 5. real summarizer, identical passages, titles on vs off ===")
    brief_plain = call_summarizer(a.summarizer, a.model, sysmsg,
                                  summary_user(a.question, block_plain))
    brief_titled = call_summarizer(a.summarizer, a.model, sysmsg,
                                   summary_user(a.question, block_titled))

    for name, brief in (("CONTROL (no titles)", brief_plain),
                        ("TITLED", brief_titled)):
        hits = sorted({m.group(0).lower() for m in ATTRIB.finditer(brief)})
        print(f"\n  --- {name}: {len(brief)} chars, "
              f"{len(hits)} attribution markers {hits[:8]} ---")
        print("   " + brief[:700].replace("\n", "\n   "))

    n_plain = len({m.group(0).lower() for m in ATTRIB.finditer(brief_plain)})
    n_titled = len({m.group(0).lower() for m in ATTRIB.finditer(brief_titled)})
    # Did any actual title text survive into the brief? The strongest evidence.
    quoted = sum(1 for t, _ in got.values()
                 if any(w in brief_titled.lower()
                        for w in [w for w in t.lower().split() if len(w) > 7][:4]))

    print(f"\n=== VERDICT ===")
    print(f"  attribution markers: control={n_plain}  titled={n_titled}")
    print(f"  titles whose wording appears in the titled brief: {quoted}/{len(got)}")
    if n_titled > n_plain or quoted > 0:
        print("  PASS: labelling the passages changed what the brief attributes.")
        return 0
    print("  FAIL: the summarizer is dropping the titles. The prompt needs another pass;\n"
          "        the join alone does nothing if the brief discards it.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
