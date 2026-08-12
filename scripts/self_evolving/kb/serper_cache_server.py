"""Serper web-search cache: the one place a paid SERP query happens.

Why this exists. The web-search arm gives the SOLVER a `web_search` tool whose
results come straight from the Serper API (Google SERP) with no model anywhere
on the path -- the anti-leak property of that arm is precisely that nothing
conditioned on the case composes the evidence. But a training step fires
hundreds of rollouts at once and the same tasks recur across steps and epochs,
so an uncached run would buy the same query thousands of times. Every tool call
lands here first: exact-match sqlite cache, fetch on miss, and a periodic
snapshot to the shared /scratch NFS so a pod recreation keeps the results
already paid for (same pattern as evidence_cache_server.py, which stays
untouched as the GPT-lookup arm's artifact).

Results are cached with a TTL (default 30 days) because SERPs drift; a stale
entry is re-fetched and, if the re-fetch fails, served stale rather than empty
-- yesterday's guideline link beats an error string in a training rollout.

Usage:
  SERPER_API_KEY=... python3 serper_cache_server.py --port 8056 --restore

Endpoints:
  POST /search {"query": str, "num": int?} -> {"text": str, "results": [...],
       "cached": bool, "hits": int}
  GET  /healthz, /stats
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import shutil
import sqlite3
import tempfile
import time
from urllib.parse import urlparse

import httpx
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("serper_cache")

SCHEMA = """
CREATE TABLE IF NOT EXISTS searches (
    key        TEXT PRIMARY KEY,
    query      TEXT NOT NULL,
    num        INTEGER NOT NULL,
    json       TEXT NOT NULL,
    fetched_at REAL NOT NULL,
    hits       INTEGER NOT NULL DEFAULT 0
);
"""

SERPER_URL = "https://google.serper.dev/search"

app = FastAPI()
STATE: dict = {"snapshots": 0, "fetches": 0, "hits": 0, "stale_served": 0, "errors": 0}


def _norm(query: str) -> str:
    return re.sub(r"\s+", " ", query.strip().lower())


class Store:
    def __init__(self, path: str):
        self.path = path
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.executescript(SCHEMA)
        self.db.commit()
        self.lock = asyncio.Lock()

    def get(self, key: str) -> tuple[str, float] | None:
        row = self.db.execute(
            "SELECT json, fetched_at FROM searches WHERE key=?", (key,)
        ).fetchone()
        return (row[0], row[1]) if row else None

    async def put(self, key: str, query: str, num: int, payload: str) -> None:
        async with self.lock:
            self.db.execute(
                "INSERT INTO searches (key, query, num, json, fetched_at, hits)"
                " VALUES (?,?,?,?,?,0) ON CONFLICT(key) DO UPDATE SET"
                " json=excluded.json, fetched_at=excluded.fetched_at",
                (key, query, num, payload, time.time()),
            )
            self.db.commit()

    async def bump(self, key: str) -> None:
        async with self.lock:
            self.db.execute("UPDATE searches SET hits=hits+1 WHERE key=?", (key,))
            self.db.commit()

    def counts(self) -> dict:
        n, paid = self.db.execute(
            "SELECT COUNT(*), COALESCE(SUM(hits),0) FROM searches"
        ).fetchone()
        # Whitelist, never dump STATE: it holds the API key and live objects.
        counters = ("snapshots", "fetches", "hits", "stale_served", "errors")
        return {"entries": n, "cache_hits_alltime": paid,
                **{k: STATE.get(k, 0) for k in counters}}


def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc.removeprefix("www.")
    except Exception:
        return ""


def format_results(query: str, data: dict) -> tuple[str, int]:
    """Verbatim SERP blocks. No model writes a word of this: titles, links and
    snippets exactly as the API returned them, numbered for citation."""
    blocks: list[str] = []
    # The answer box is a featured snippet FROM a source page, so it is carried
    # as a regular result block attributed to its link, not as an "answer".
    ab = data.get("answerBox") or {}
    if ab.get("snippet") and ab.get("link"):
        blocks.append((ab.get("title") or "", ab["link"], ab["snippet"], ab.get("date")))
    for r in data.get("organic") or []:
        if r.get("link"):
            blocks.append((r.get("title") or "", r["link"], r.get("snippet") or "", r.get("date")))
    if not blocks:
        return (f'No web results for "{query}".', 0)
    out = [f'Web search results for "{query}":']
    for i, (title, link, snippet, date) in enumerate(blocks, 1):
        src = _domain(link)
        head = f"[w{i}] {title}" + (f" ({src}" + (f", {date})" if date else ")") if src else "")
        out.append(f"{head}\n{link}\n{snippet}")
    return ("\n\n".join(out), len(blocks))


class SearchIn(BaseModel):
    query: str
    num: int | None = None


@app.post("/search")
async def search(p: SearchIn):
    num = min(max(int(p.num or STATE["num_default"]), 1), 20)
    key = f"{_norm(p.query)}|{num}"
    st: Store = STATE["store"]
    row = st.get(key)
    now = time.time()
    if row and now - row[1] < STATE["ttl_s"]:
        STATE["hits"] += 1
        await st.bump(key)
        data = json.loads(row[0])
        text, hits = format_results(p.query, data)
        return {"text": text, "results": data.get("organic") or [], "cached": True, "hits": hits}

    paid = False
    async with STATE["sem"]:
        # Re-check under the semaphore: a burst of identical queries queues here
        # and only the first should pay.
        row2 = st.get(key)
        if row2 and row2[1] > now:
            row, data = row2, json.loads(row2[0])
        else:
            paid = True
            try:
                async with httpx.AsyncClient(timeout=20) as client:
                    resp = await client.post(
                        SERPER_URL,
                        headers={"X-API-KEY": STATE["api_key"], "Content-Type": "application/json"},
                        json={"q": p.query, "num": num},
                    )
                    resp.raise_for_status()
                    data = resp.json()
                await st.put(key, p.query, num, json.dumps(data))
                STATE["fetches"] += 1
            except Exception as e:
                STATE["errors"] += 1
                logger.warning("serper fetch failed for %r: %s: %s", p.query[:80], type(e).__name__, e)
                if row:  # serve stale over serving nothing
                    STATE["stale_served"] += 1
                    data = json.loads(row[0])
                else:
                    return {"text": "Web search is temporarily unavailable.",
                            "results": [], "cached": False, "hits": 0, "error": str(e)[:200]}
    text, hits = format_results(p.query, data)
    return {"text": text, "results": data.get("organic") or [], "cached": not paid, "hits": hits}


@app.get("/healthz")
async def healthz():
    return {"ok": True, "entries": STATE["store"].counts()["entries"]}


@app.get("/stats")
async def stats():
    return STATE["store"].counts()


async def _snapshotter():
    """Copy the local db to /scratch periodically. Same rationale as the evidence
    cache: the pod's local disk dies with the pod, and the snapshot is what makes
    paid fetches survive recreation. sqlite backup() gives a consistent copy."""
    while True:
        await asyncio.sleep(STATE["snapshot_s"])
        dest = STATE["snapshot_path"]
        if not dest:
            continue
        try:
            tmp = tempfile.mktemp(dir=os.path.dirname(dest), prefix=".serper_snap_")
            async with STATE["store"].lock:
                out = sqlite3.connect(tmp)
                STATE["store"].db.backup(out)
                out.close()
            shutil.move(tmp, dest)
            STATE["snapshots"] += 1
            logger.info("snapshot -> %s (%s)", dest, STATE["store"].counts())
        except Exception as e:
            logger.warning("snapshot failed: %s: %s", type(e).__name__, e)


@app.on_event("startup")
async def _startup():
    STATE["sem"] = asyncio.Semaphore(STATE["concurrency"])
    STATE["snap_task"] = asyncio.create_task(_snapshotter())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="/root/search_cache.sqlite",
                    help="local sqlite path (fast disk; snapshotted to --snapshot)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8056)
    ap.add_argument("--key_file", default="/scratch/sheng/self_evolving/.serper_key",
                    help="file holding the Serper API key; SERPER_API_KEY env wins")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--num_default", type=int, default=8)
    ap.add_argument("--ttl_days", type=float, default=30.0)
    ap.add_argument("--snapshot",
                    default="/scratch/sheng/self_evolving/kb/search_cache.sqlite")
    ap.add_argument("--snapshot_s", type=int, default=600)
    ap.add_argument("--restore", action="store_true",
                    help="seed the local db from --snapshot when the local db is missing")
    args = ap.parse_args()

    key = os.getenv("SERPER_API_KEY", "")
    if not key and os.path.exists(args.key_file):
        key = open(args.key_file).read().strip()
    if not key:
        logger.error("no Serper key: set SERPER_API_KEY or provide --key_file")
        return 1

    if args.restore and not os.path.exists(args.db) and os.path.exists(args.snapshot):
        shutil.copyfile(args.snapshot, args.db)
        logger.info("restored %s from %s", args.db, args.snapshot)

    STATE.update(api_key=key, store=Store(args.db), concurrency=args.concurrency,
                 num_default=args.num_default, ttl_s=args.ttl_days * 86400,
                 snapshot_path=args.snapshot, snapshot_s=args.snapshot_s)
    logger.info("serper cache up on :%d db=%s (%s)", args.port, args.db, STATE["store"].counts())
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
