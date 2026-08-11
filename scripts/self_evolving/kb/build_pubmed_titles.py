#!/usr/bin/env python3
"""Re-attach article titles to the PubMed half of medical_knowledge_v2.

WHY. 23.9M of the collection's 57.2M rows are `source_dataset='medrag_pubmed'`, and their
`text_content` holds the abstract body with the bibliographic metadata stripped: there is no
title, author, journal or year field in the schema at all. Probed directly, the KB returns
the ACORN trial's own abstract, the SOAP II abstract, and Lau et al.'s NEJM conclusion as
top hits for their respective questions -- while the strings "ACORN", "SOAP" and "Lau" never
appear. So retrieval finds exactly the right evidence and the policy still cannot name it.

Measured consequence: of the 118 HealthBench-Pro positive criteria that demand a source, 15
name a specific work (a trial acronym, an author, a journal) and are UNSATISFIABLE from this
corpus at any level of retrieval skill; a further 22 name only an issuing organisation and
are satisfiable, because organisations survive in prose.

WHAT A TITLE DOES AND DOES NOT BUY -- checked against the real data, not assumed. The Lau
et al. paper comes back as "Effect of intravenous omeprazole on recurrent bleeding after
endoscopic treatment of bleeding peptic ulcers": the paper is identifiable, but a title
never carries the author, the journal or the year. Trial acronyms survive only where the
original title carried them ("Coronary artery surgery study (CASS): a randomized trial"),
which modern trials usually do and older ones often do not. So titles reach the "identify
the evidence" criteria and the acronym subset; they do NOT reach "the 2000 NEJM trial by
Lau et al.". PMID is stored for exactly that gap -- it is the canonical handle from which
author/journal/year can be resolved later, for the handful of passages actually retrieved
rather than for all 23.9M rows.

WHY THIS IS A JOIN AND NOT A REBUILD. The rows keep their MedRAG chunk ids
(`pubmed23n0792_18`), and the upstream MedRAG/pubmed corpus ships a `title` per chunk under
the same id. So titles come back via a lookup keyed on `entry_id` -- no re-embedding, no
Milvus schema change, and nothing about ranking moves. Retrieval returns the same passages
in the same order; they simply arrive labelled.

DISK. /scratch is a shared NFS mount that has been sitting at 96-97% full, and the corpus is
55.3 GB across 997 files. So this streams: fetch one ~35 MB file, keep (id, title, PMID), delete
it. Peak extra disk is one chunk file plus the database. The DB is built on LOCAL disk
because SQLite over NFS is both slow and unsafe under concurrent access, then published to
/scratch at the end so every pod can read it.

Resumable by design: a 997-file job WILL be interrupted. Completed files are recorded in the
DB itself, so a re-run continues rather than restarting.

  python3 build_pubmed_titles.py --out /scratch/sheng/self_evolving/kb/pubmed_titles.sqlite
  python3 build_pubmed_titles.py --verify-only    # sanity-check an existing DB
"""

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time

REPO = "MedRAG/pubmed"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main"
LIST_URL = f"https://huggingface.co/api/datasets/{REPO}/tree/main?recursive=1"

# The identifiers whose absence motivated the job. Checked at the end against the real DB:
# if these do not come back, the join did not do what it claims.
# NOTE on the second entry: this row is NOT SOAP II. It is an older
# norepinephrine-vs-dopamine paper that outranked it for that query -- a reminder that a
# top hit resembling the asked-for trial is not the asked-for trial, and that a criterion
# naming a trial needs the NAME, not merely the topic.
VERIFY = [
    ("pubmed23n0859_11771", ("acorn", "kidney")),
    ("pubmed23n0280_2499", ("dopamine", "norepinephrine")),
    ("pubmed23n0558_23219", ("omeprazole",)),
]


def _curl_json(url: str, timeout: int = 120):
    out = subprocess.run(["curl", "-sSL", "-m", str(timeout), url],
                         capture_output=True, check=True).stdout
    return json.loads(out)


def _open_db(path: str) -> sqlite3.Connection:
    db = sqlite3.connect(path)
    # This artifact is rebuildable from the source corpus, so durability guarantees are
    # not worth their cost across 23.9M inserts.
    db.execute("PRAGMA journal_mode=OFF")
    db.execute("PRAGMA synchronous=OFF")
    db.execute("PRAGMA cache_size=-200000")          # ~200 MB page cache
    # WITHOUT ROWID keeps one b-tree instead of a table plus an index: the id IS the key,
    # and at this row count that is several GB of difference.
    #
    # PMID is captured alongside the title even though nothing reads it yet. Titles alone
    # do NOT satisfy every criterion: measured on the real data, the Lau et al. paper's
    # title is "Effect of intravenous omeprazole on recurrent bleeding after endoscopic
    # treatment of bleeding peptic ulcers" -- the paper is identifiable, but the author,
    # journal and year appear nowhere in a title, ever. Acronyms survive only when the
    # original title carried them ("Coronary artery surgery study (CASS): ..."). PMID is
    # the canonical handle that lets author/journal/year be resolved later for the handful
    # of passages actually retrieved. It costs ~8 bytes a row here and saves re-downloading
    # 55 GB to get it.
    db.execute("CREATE TABLE IF NOT EXISTS titles "
               "(id TEXT PRIMARY KEY, title TEXT, pmid TEXT) WITHOUT ROWID")
    db.execute("CREATE TABLE IF NOT EXISTS done (fname TEXT PRIMARY KEY)")
    db.commit()
    return db


def build(args) -> int:
    os.makedirs(os.path.dirname(args.local) or ".", exist_ok=True)
    db = _open_db(args.local)
    done = {r[0] for r in db.execute("SELECT fname FROM done")}

    files = [f["path"] for f in _curl_json(LIST_URL)
             if f.get("type") == "file" and f["path"].endswith(".jsonl")]
    files.sort()
    todo = [f for f in files if f not in done]
    print(f"{len(files)} chunk files, {len(done)} already done, {len(todo)} to go",
          flush=True)

    tmp = args.tmp
    os.makedirs(tmp, exist_ok=True)
    t0 = time.time()
    rows_total = db.execute("SELECT count(*) FROM titles").fetchone()[0]

    for i, path in enumerate(todo, 1):
        dest = os.path.join(tmp, os.path.basename(path))
        try:
            subprocess.run(["curl", "-sSL", "--retry", "3", "--retry-delay", "2",
                            "-m", "600", "-o", dest, f"{BASE}/{path}"], check=True)
            batch = []
            with open(dest, errors="ignore") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    cid = d.get("id")
                    title = (d.get("title") or "").strip()
                    pmid = str(d.get("PMID") or "").strip()
                    # A chunk with no title contributes nothing; storing an empty string
                    # would make lookups return "" and read as "titled, but blank".
                    if cid and title:
                        batch.append((cid, title[:400], pmid))
            db.executemany("INSERT OR REPLACE INTO titles VALUES (?,?,?)", batch)
            db.execute("INSERT OR REPLACE INTO done VALUES (?)", (path,))
            db.commit()
            rows_total += len(batch)
        except subprocess.CalledProcessError as e:
            # One bad file must not lose 996 good ones. It stays absent from `done`, so a
            # re-run retries exactly the gaps.
            print(f"  FAILED {path}: {e}", flush=True)
            continue
        finally:
            if os.path.exists(dest):
                os.remove(dest)

        if i % 25 == 0 or i == len(todo):
            el = time.time() - t0
            print(f"  [{i}/{len(todo)}] {rows_total:,} titles  "
                  f"{el/60:.1f}min elapsed  eta {(el/i)*(len(todo)-i)/60:.0f}min",
                  flush=True)

    n = db.execute("SELECT count(*) FROM titles").fetchone()[0]
    print(f"built {n:,} titles at {args.local} "
          f"({os.path.getsize(args.local)/1e9:.2f} GB)", flush=True)
    db.close()
    return n


def verify(path: str) -> bool:
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    n = db.execute("SELECT count(*) FROM titles").fetchone()[0]
    print(f"titles in DB: {n:,}")
    ok = True
    for cid, needles in VERIFY:
        row = db.execute("SELECT title, pmid FROM titles WHERE id=?", (cid,)).fetchone()
        t = (row[0] if row else "") or ""
        hit = any(x in t.lower() for x in needles)
        print(f"  {cid:24} {'OK ' if hit else 'MISS'} pmid={row[1] if row else '-'} {t[:95]!r}")
        ok = ok and bool(row)
    db.close()
    return ok and n > 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/scratch/sheng/self_evolving/kb/pubmed_titles.sqlite",
                    help="Final published location, readable by every pod.")
    ap.add_argument("--local", default="/root/pubmed_titles.sqlite",
                    help="Build location. Keep it on LOCAL disk: SQLite over NFS is slow "
                         "and unsafe under concurrent access.")
    ap.add_argument("--tmp", default="/root/_pmchunks",
                    help="Scratch dir for one chunk file at a time.")
    ap.add_argument("--verify-only", action="store_true")
    a = ap.parse_args()

    if a.verify_only:
        return 0 if verify(a.out if os.path.exists(a.out) else a.local) else 1

    build(a)
    if not verify(a.local):
        print("FATAL: verification failed; not publishing", file=sys.stderr)
        return 1
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    # Publish atomically: a half-copied DB on NFS would be read by a live gen server.
    tmp_out = a.out + ".partial"
    shutil.copy2(a.local, tmp_out)
    os.replace(tmp_out, a.out)
    print(f"published -> {a.out}")
    shutil.rmtree(a.tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
