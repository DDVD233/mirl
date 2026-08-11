#!/usr/bin/env python3
"""Attach journal, year and first author to PubMed passages, keyed by PMID.

WHY, ON TOP OF build_pubmed_titles.py. That job restored the article TITLE, which makes a
passage identifiable and recovers trial acronyms where the original title carried one. It
does not recover the author, the journal or the year, because a title never contains them --
so a rubric criterion asking for "the 2000 NEJM trial by Lau et al." was still unreachable.
Those three fields are what turn an identifiable passage into a citable one.

SOURCE. The MedRAG chunk files carry only (id, title, content, contents, PMID), so the
metadata has to come from PubMed itself. Two ways in:

  - esummary: 23.9M PMIDs at 200 per request is ~120k requests, rate-limited to 3/s without
    an API key (~11 hours), and makes the KB depend on an external service.
  - the BASELINE XML: 1334 files, parsed locally, no rate limit, no train-time dependency.

The baseline wins for the same reason the summarizer is frozen: a reproducible corpus beats
a live lookup. Note the 2023 baseline the MedRAG ids were cut from is no longer hosted --
NCBI keeps only the current year (pubmed26n0001..1334) -- which does not matter, because the
baseline is CUMULATIVE over all of PubMed and the join key is the PMID, not the file number.

SHARDED, then merged. XML parsing dominates the wall clock here (unlike the title job, which
was download-bound), so files are split across worker processes, each writing its own TSV
shard, and the shards are folded into SQLite at the end. Concurrent SQLite writers would
serialise the very thing being parallelised.

Streams like its sibling: fetch one ~25 MB file, parse, delete. Peak disk is a few files
plus the shards, which matters on a /scratch that sits at 97%.

  python3 build_pubmed_meta.py                      # all 1334 files, 12 workers
  python3 build_pubmed_meta.py --workers 8 --limit 20   # quick smoke over 20 files
  python3 build_pubmed_meta.py --verify-only
"""

import argparse
import glob
import gzip
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from multiprocessing import Pool

BASE = "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline"
LISTING = BASE + "/"

# PMIDs whose citation we know independently, so a wrong join is visible rather than
# plausible. Lau et al. 2000 NEJM is the exact paper a HealthBench criterion asks for.
VERIFY = {
    "10891516": ("N Engl J Med", "2000"),
    "17201879": ("J Gastroenterol Hepatol", "2007"),
}


def list_files() -> list[str]:
    out = subprocess.run(["curl", "-sSL", "-m", "120", LISTING],
                         capture_output=True, check=True).stdout.decode("utf8", "ignore")
    import re
    return sorted(set(re.findall(r"pubmed\d+n\d+\.xml\.gz", out)))


def parse_one(args) -> tuple[str, int, str]:
    """Download, parse and delete ONE baseline file; append rows to this worker's shard.

    Returns (fname, n_rows, error). Never raises: one unreadable file out of 1334 must not
    lose the other 1333, and it stays out of the shard's done-list so a re-run retries it.
    """
    fname, shard_dir, tmp_dir = args
    dest = os.path.join(tmp_dir, fname)
    shard = os.path.join(shard_dir, fname + ".tsv")
    if os.path.exists(shard + ".done"):
        return fname, 0, "skipped"
    n = 0
    try:
        subprocess.run(["curl", "-sSL", "--retry", "3", "--retry-delay", "2",
                        "-m", "900", "-o", dest, f"{BASE}/{fname}"], check=True)
        with gzip.open(dest, "rb") as fh, open(shard, "w") as out:
            # iterparse + clear(): the decompressed file is ~250 MB and building a full
            # DOM for each of 1334 of them would trade the whole point of streaming.
            for _, el in ET.iterparse(fh, events=("end",)):
                if el.tag != "PubmedArticle":
                    continue
                try:
                    pmid_el = el.find("./MedlineCitation/PMID")
                    art = el.find("./MedlineCitation/Article")
                    if pmid_el is None or art is None:
                        continue
                    pmid = (pmid_el.text or "").strip()
                    j = art.find("./Journal")
                    iso = j.findtext("./ISOAbbreviation", "") if j is not None else ""
                    full = j.findtext("./Title", "") if j is not None else ""
                    year = ""
                    if j is not None:
                        pd = j.find("./JournalIssue/PubDate")
                        if pd is not None:
                            # MedlineDate covers the ranged forms ("1998 Nov-Dec") that
                            # have no <Year> element at all.
                            year = (pd.findtext("./Year", "")
                                    or (pd.findtext("./MedlineDate", "") or "")[:4])
                    a1 = art.find("./AuthorList/Author")
                    author = ""
                    if a1 is not None:
                        last = a1.findtext("./LastName", "") or ""
                        init = a1.findtext("./Initials", "") or ""
                        author = (f"{last} {init}".strip()
                                  or a1.findtext("./CollectiveName", "") or "")
                    n_auth = len(art.findall("./AuthorList/Author"))
                    if pmid:
                        # Tabs and newlines would corrupt the shard; journal titles
                        # occasionally contain neither, but "occasionally" is not "never".
                        row = [pmid, iso or full, year, author, str(n_auth)]
                        out.write("\t".join(c.replace("\t", " ").replace("\n", " ")
                                            for c in row) + "\n")
                        n += 1
                finally:
                    el.clear()
        open(shard + ".done", "w").close()
    except Exception as e:  # noqa: BLE001
        return fname, n, f"{type(e).__name__}: {str(e)[:120]}"
    finally:
        if os.path.exists(dest):
            os.remove(dest)
    return fname, n, ""


def fold_shards(shard_dir: str, db_path: str) -> int:
    db = sqlite3.connect(db_path)
    db.execute("PRAGMA journal_mode=OFF")
    db.execute("PRAGMA synchronous=OFF")
    db.execute("PRAGMA cache_size=-200000")
    db.execute("CREATE TABLE IF NOT EXISTS meta (pmid TEXT PRIMARY KEY, journal TEXT, "
               "year TEXT, author TEXT, n_authors TEXT) WITHOUT ROWID")
    total = 0
    for tsv in sorted(glob.glob(os.path.join(shard_dir, "*.tsv"))):
        batch = []
        with open(tsv, errors="ignore") as fh:
            for line in fh:
                p = line.rstrip("\n").split("\t")
                if len(p) == 5 and p[0]:
                    batch.append(tuple(p))
        if batch:
            db.executemany("INSERT OR REPLACE INTO meta VALUES (?,?,?,?,?)", batch)
            db.commit()
            total += len(batch)
    n = db.execute("SELECT count(*) FROM meta").fetchone()[0]
    db.close()
    print(f"folded {total:,} rows -> {n:,} distinct PMIDs", flush=True)
    return n


def verify(db_path: str) -> bool:
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        n = db.execute("SELECT count(*) FROM meta").fetchone()[0]
    except sqlite3.OperationalError:
        print("no meta table"); return False
    print(f"PMIDs with metadata: {n:,}")
    ok = True
    for pmid, (want_j, want_y) in VERIFY.items():
        r = db.execute("SELECT journal, year, author FROM meta WHERE pmid=?",
                       (pmid,)).fetchone()
        if not r:
            print(f"  {pmid}  MISSING"); ok = False; continue
        good = want_j.lower()[:12] in (r[0] or "").lower() and r[1] == want_y
        print(f"  {pmid}  {'OK  ' if good else 'WRONG'} journal={r[0]!r} year={r[1]!r} "
              f"author={r[2]!r}   (expected ~{want_j} {want_y})")
        ok = ok and good
    db.close()
    return ok and n > 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="/root/pubmed_titles.sqlite",
                    help="Same DB the titles live in: one file to publish and one to read.")
    ap.add_argument("--out", default="/scratch/sheng/self_evolving/kb/pubmed_titles.sqlite")
    ap.add_argument("--shards", default="/root/_pmmeta")
    ap.add_argument("--tmp", default="/root/_pmxml")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0, help="First N files only (smoke test).")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--no-publish", action="store_true")
    a = ap.parse_args()

    if a.verify_only:
        return 0 if verify(a.out if os.path.exists(a.out) else a.db) else 1

    os.makedirs(a.shards, exist_ok=True)
    os.makedirs(a.tmp, exist_ok=True)
    files = list_files()
    if a.limit:
        files = files[:a.limit]
    print(f"{len(files)} baseline files, {a.workers} workers", flush=True)

    t0, done, rows, errs = time.time(), 0, 0, []
    with Pool(a.workers) as pool:
        for fname, n, err in pool.imap_unordered(
                parse_one, [(f, a.shards, a.tmp) for f in files], chunksize=1):
            done += 1
            rows += n
            if err and err != "skipped":
                errs.append((fname, err))
            if done % 25 == 0 or done == len(files):
                el = time.time() - t0
                print(f"  [{done}/{len(files)}] {rows:,} articles  {el/60:.1f}min  "
                      f"eta {(el/done)*(len(files)-done)/60:.0f}min  errs={len(errs)}",
                      flush=True)
    for f, e in errs[:10]:
        print(f"  FAILED {f}: {e}", flush=True)

    n = fold_shards(a.shards, a.db)
    if not verify(a.db):
        print("FATAL: verification failed; not publishing", file=sys.stderr)
        return 1
    if a.no_publish:
        return 0
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    tmp_out = a.out + ".partial"
    shutil.copy2(a.db, tmp_out)
    os.replace(tmp_out, a.out)                     # atomic: a live server may be reading
    print(f"published -> {a.out} ({os.path.getsize(a.out)/1e9:.2f} GB)")
    shutil.rmtree(a.tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
