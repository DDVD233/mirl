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


# Does the brief attribute anything? Organisation and journal names are matched
# CASE-SENSITIVELY: a case-insensitive \bWHO\b matched the English word "who" and scored a
# brief that attributed nothing, which is how the first version of this test reported a pass
# on a failing pipeline.
ATTRIB = re.compile(
    r"\b(per the|according to|guideline|American \w+|European \w+|National \w+|"
    r"World Health|ACC|AHA|ESC|NICE|WHO|USPSTF|IDSA|ASCO|NCCN|AUA|ACG|KDIGO|"
    r"NEJM|New England|Lancet|JAMA|BMJ|PMID|et al)\b")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", default=DEFAULT_Q)
    ap.add_argument("--summarizer", default="http://point.dd.works:18186/v1")
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--milvus", default="http://mib.media.mit.edu:19531")
    ap.add_argument("--embed", default="http://mib.media.mit.edu:18001/v1")
    ap.add_argument("--top_k", type=int, default=8)
    ap.add_argument("--real", action="store_true",
                    help="Use the PUBLISHED database via retrieval.py's own default path, "
                         "with no env var set -- i.e. exactly what a training run gets. "
                         "Without this, titles are fetched per-id into a temp DB so the "
                         "test can run before the build finishes.")
    a = ap.parse_args()

    import retrieval as R

    print("=== 1. real retrieval (embed + Milvus + source-aware ranking) ===")
    passages, _ = R.local_search(a.question, top_k=a.top_k, milvus_uri=a.milvus,
                                 embed_api_base=a.embed)
    if not passages:
        print("FATAL: retrieval returned nothing"); return 1
    print(f"  {len(passages)} passages; sources: "
          f"{sorted({p['source'] for p in passages})}")

    import copy
    import importlib
    n_pubmed = sum(1 for p in passages
                   if str(p.get("entry_id", "")).startswith("pubmed"))

    if a.real:
        # No env var, no temp DB: whatever a training run would see.
        os.environ.pop("PUBMED_TITLES_DB", None)
        R = importlib.reload(R)
        print(f"\n=== 2. PUBLISHED database via the default path (no env var) ===")
        print(f"  {R._TITLES_DB}")
        if not os.path.exists(R._TITLES_DB):
            print("FATAL: the published DB does not exist at the default path"); return 1
        got = {}
    else:
        print("\n=== 2. titles for exactly these ids, from the same chunk files ===")
        got = fetch_titles([p.get("entry_id", "") for p in passages])
        print(f"  resolved {len(got)} titles for {n_pubmed} pubmed passages")
        for eid, (t, pmid) in list(got.items())[:3]:
            print(f"    {eid}  pmid={pmid}  {t[:90]!r}")
        db = os.path.join(tempfile.mkdtemp(), "t.sqlite")
        con = sqlite3.connect(db)
        con.execute("CREATE TABLE titles (id TEXT PRIMARY KEY, title TEXT, pmid TEXT) "
                    "WITHOUT ROWID")
        con.executemany("INSERT INTO titles VALUES (?,?,?)",
                        [(k, v[0], v[1]) for k, v in got.items()])
        con.commit(); con.close()
        os.environ["PUBMED_TITLES_DB"] = db
        R = importlib.reload(R)

    print("\n=== 3. format WITHOUT titles (control) and WITH titles ===")
    plain = copy.deepcopy(passages)
    # A pristine copy for the control: attach_titles mutates in place, so formatting the
    # same list twice would leak the labels into the control block.
    block_plain = "\n\n".join(
        f"[passage {i + 1} | source={p['source']}]\n{p['text']}"
        for i, p in enumerate(plain))
    titled = copy.deepcopy(passages)
    n = R.attach_titles(titled)
    block_titled = R.format_passages(titled)
    if a.real:
        # Report what the REAL db actually supplied, which is the thing being verified.
        got = {p["entry_id"]: (p.get("title", ""), p.get("pmid", ""))
               for p in titled if p.get("title")}
        for p in titled:
            if p.get("title"):
                print(f"    {p['entry_id']}  pmid={p.get('pmid')}  "
                      f"cite={p.get('citation')!r}\n      title={p['title'][:88]!r}")
        print(f"  labelled {n}/{len(titled)} passages ({n_pubmed} are pubmed)")
    if n == 0:
        print("FATAL: no passage got a title; the join is not working"); return 1
    if "title=" not in block_titled:
        print("FATAL: title missing from the formatted block"); return 1

    # THE RAW BLOCK IS ITSELF AN ANSWER-TIME INPUT, not just an intermediate: /retrieve
    # serves it verbatim whenever summarization is disabled or both summarizer endpoints
    # fail, so if the labels were missing here a summarizer outage would silently strip
    # every citation from the run.
    print("\n=== 3b. the RAW tool response the model sees (labelled headers) ===")
    for line in block_titled.splitlines():
        if line.startswith("[passage"):
            print(f"  {line[:190]}")
    raw_titles = sum(1 for line in block_titled.splitlines() if "title=" in line)
    raw_cites = sum(1 for line in block_titled.splitlines() if "cite=" in line)
    raw_pmids = sum(1 for line in block_titled.splitlines() if "pmid=" in line)
    print(f"  raw block carries: {raw_titles} title=, {raw_cites} cite=, {raw_pmids} pmid=")

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

    # /retrieve appends the source list ITSELF -- the summarizer is told not to write one,
    # because when asked it produced a Sources section on one sample and none on the next.
    # So the tool response is brief + appended block, and that is what must be verified;
    # checking only the model's own output would test a behaviour we deliberately removed.
    appended = R.sources_block(titled)
    if appended:
        brief_titled = f"{brief_titled}\n\n{appended}"
    print(f"\n  appended by /retrieve ({len(appended)} chars):")
    print("   " + (appended or "(nothing -- no passage was labelled)").replace("\n", "\n   "))

    for name, brief in (("CONTROL (no titles)", brief_plain),
                        ("TITLED", brief_titled)):
        hits = sorted({m.group(0).lower() for m in ATTRIB.finditer(brief)})
        print(f"\n  --- {name}: {len(brief)} chars, "
              f"{len(hits)} attribution markers {hits[:8]} ---")
        print("   " + brief[:700].replace("\n", "\n   "))

    n_plain = len({m.group(0) for m in ATTRIB.finditer(brief_plain)})
    n_titled = len({m.group(0) for m in ATTRIB.finditer(brief_titled)})

    # The decisive test: a Sources section naming a real title. NOT "do the title's words
    # appear in the brief" -- an abstract's conclusion and its title say the same thing
    # ("High-dose omeprazole reduces recurrent bleeding..."), so word overlap is satisfied
    # by the passage body alone and scored the previous version of this test as a pass while
    # the pipeline attributed nothing.
    def sources_block(b):
        m = re.search(r"^\s*Sources?\s*:\s*$(.*)", b, re.M | re.S)
        return (m.group(1) if m else "").strip()

    src_titled, src_plain = sources_block(brief_titled), sources_block(brief_plain)
    # A title counts as cited when a CONTIGUOUS run of it appears inside that section.
    # An earlier version built its needle by dropping short words -- "effect esomeprazole
    # recurrent bleeding" -- a string that appears in no real title, so it reported 0/3
    # against a Sources block that in fact carried all three titles verbatim.
    def norm(x):
        return re.sub(r"\s+", " ", x).strip().lower()

    src_norm = norm(src_titled)
    named = [t for t, _ in got.values() if norm(t)[:60] in src_norm]
    pmids = [p for _, p in got.values() if p and p in src_titled]

    print("\n=== VERDICT ===")
    print(f"  attribution markers:   control={n_plain}  titled={n_titled}")
    print(f"  Sources section:       control={'yes' if src_plain else 'no'}  "
          f"titled={'yes' if src_titled else 'no'}")
    print(f"  titles named in it:    {len(named)}/{len(got)}")
    print(f"  PMIDs carried:         {len(pmids)}/{sum(1 for _, p in got.values() if p)}")
    if src_titled:
        print("  --- Sources block as written ---")
        print("   " + src_titled[:400].replace("\n", "\n   "))
    # PMIDs count as a pass too. They are the harder signal, not the softer one: a PMID
    # cannot be produced from the passage body, so its presence proves the header survived,
    # whereas title wording can leak in from the abstract's own conclusion.
    if named or pmids:
        print("  PASS: the brief carries its sources' identifiers, which it could not "
              "before -- a PMID in particular cannot come from the passage body.")
        return 0
    print("  FAIL: the summarizer is still dropping the labels. The join is inert until\n"
          "        the brief carries them -- fix the prompt, not the database.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
