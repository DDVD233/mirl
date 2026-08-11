#!/usr/bin/env python3
"""Shared, persistent cache and fetch queue for web-search evidence.

WHY A SERVICE AND NOT AN IN-PROCESS DICT. The generation server is single-process
(`uvicorn.run(app, ...)` with no `workers=`; `--workers 20` spawns asyncio tasks), so an
in-process cache is already shared by every generation worker and every rollout worker that
calls /retrieve over HTTP. What that does NOT give is sharing across HOSTS (2335, 2336, future
arms) or across SESSIONS, and it puts the fetch queue, the rate limit and the breaker in
whichever process happens to own them. This service owns all four.

WHY SQLITE AND NOT JSONL. Requirement: keep EVERYTHING from each call so a later summarisation
never has to re-fetch. One measured web_search call is ~13k tokens, so a complete record is
tens of KB; at 100k entries a JSONL store is several GB to parse at startup. SQLite gives an
indexed key, lazy reads and a single file. WAL mode is safe here because the file lives on
LOCAL disk -- SQLite over NFS is not safe under concurrent writers, which is why the titles DB
is opened read-only.

THE QUEUE HANDLES THE TAIL, NOT THE LOAD. Measured latency is 6.0-12.3s (median 6.6s), so
/retrieve calls web search SYNCHRONOUSLY: 800 calls/step at 6.6s with concurrency 16 is ~5.5
min/step. An earlier lone call timed out at 90s and I wrongly treated that as typical, which
had made a fully asynchronous design look mandatory; it is not. What the queue is still for is
the tail -- a call that times out or trips the breaker is enqueued here, so the next rollout
asking the same question gets it from cache instead of paying again. It also serves offline
pre-warming.

  python3 evidence_cache_server.py --port 8055 --db /root/evidence_cache.sqlite
  curl -s localhost:8055/stats
  curl -s -XPOST localhost:8055/lookup -d '{"query":"..."}' -H 'content-type: application/json'
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sqlite3
import sys
import time
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from fastapi import FastAPI                                    # noqa: E402
from pydantic import BaseModel                                 # noqa: E402
from web_evidence import WebEvidence, cache_key, normalize_query  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evidence_cache")

SCHEMA = """
CREATE TABLE IF NOT EXISTS cache (
    key         TEXT PRIMARY KEY,
    query       TEXT NOT NULL,
    text        TEXT,
    sources     TEXT,          -- JSON list of bibliographic sources
    raw         BLOB,          -- zlib(JSON) of the ENTIRE response: never re-fetch
    model       TEXT,
    tokens      INTEGER,
    fetched_at  TEXT           -- unused today; a refresh pass would expire on it
);
CREATE TABLE IF NOT EXISTS queue (
    key        TEXT PRIMARY KEY,
    query      TEXT NOT NULL,
    enqueued_at TEXT,
    tries      INTEGER DEFAULT 0,
    last_error TEXT
);
"""


class LookupIn(BaseModel):
    query: str
    enqueue: bool = True          # a miss schedules a background fetch by default


class PutIn(BaseModel):
    query: str
    text: str = ""
    sources: list = []
    raw: dict | None = None
    model: str = ""
    tokens: int = 0


class Store:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        # WAL: concurrent readers alongside one writer, and it survives a crash. Safe because
        # this file is on local disk and only this process writes it.
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.executescript(SCHEMA)
        self.db.commit()
        self.lock = asyncio.Lock()

    def get(self, query: str) -> dict | None:
        r = self.db.execute(
            "SELECT query, text, sources, fetched_at, tokens FROM cache WHERE key=?",
            (cache_key(query),)).fetchone()
        if not r:
            return None
        return {"query": r[0], "text": r[1] or "",
                "sources": json.loads(r[2] or "[]"), "fetched_at": r[3] or "",
                "tokens": r[4] or 0}

    def get_raw(self, query: str) -> dict | None:
        """The complete original response. Kept so a later re-summarisation of an
        over-long brief never has to spend another call."""
        r = self.db.execute("SELECT raw FROM cache WHERE key=?",
                            (cache_key(query),)).fetchone()
        if not r or not r[0]:
            return None
        try:
            return json.loads(zlib.decompress(r[0]).decode("utf8"))
        except Exception:  # noqa: BLE001
            return None

    async def put(self, q: str, text: str, sources: list, raw: dict | None,
                  model: str, tokens: int) -> None:
        blob = None
        if raw is not None:
            blob = zlib.compress(json.dumps(raw, ensure_ascii=False).encode("utf8"), 6)
        async with self.lock:
            self.db.execute(
                "INSERT OR REPLACE INTO cache "
                "(key, query, text, sources, raw, model, tokens, fetched_at) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (cache_key(q), q[:4000], text, json.dumps(sources, ensure_ascii=False),
                 blob, model, int(tokens or 0),
                 time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
            self.db.execute("DELETE FROM queue WHERE key=?", (cache_key(q),))
            self.db.commit()

    async def enqueue(self, q: str) -> bool:
        k = cache_key(q)
        async with self.lock:
            if self.db.execute("SELECT 1 FROM cache WHERE key=?", (k,)).fetchone():
                return False
            cur = self.db.execute(
                "INSERT OR IGNORE INTO queue (key, query, enqueued_at) VALUES (?,?,?)",
                (k, q[:4000], time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
            self.db.commit()
            return cur.rowcount > 0

    async def take(self, n: int) -> list[tuple[str, str]]:
        """Oldest-first, so the queue drains in the order rollouts asked."""
        async with self.lock:
            rows = self.db.execute(
                "SELECT key, query FROM queue WHERE tries < 3 "
                "ORDER BY enqueued_at LIMIT ?", (n,)).fetchall()
            for k, _ in rows:
                self.db.execute("UPDATE queue SET tries = tries + 1 WHERE key=?", (k,))
            self.db.commit()
        return [(k, q) for k, q in rows]

    async def fail(self, key: str, err: str) -> None:
        async with self.lock:
            self.db.execute("UPDATE queue SET last_error=? WHERE key=?", (err[:200], key))
            self.db.commit()

    def counts(self) -> dict:
        c = self.db.execute("SELECT count(*), coalesce(sum(tokens),0) FROM cache").fetchone()
        q = self.db.execute("SELECT count(*) FROM queue WHERE tries < 3").fetchone()[0]
        dead = self.db.execute("SELECT count(*) FROM queue WHERE tries >= 3").fetchone()[0]
        try:
            size = os.path.getsize(self.path)
        except OSError:
            size = 0
        return {"entries": c[0], "tokens_saved": c[1], "queued": q, "dead": dead,
                "db_bytes": size}


app = FastAPI()
STATE: dict = {}


@app.post("/lookup")
async def lookup(p: LookupIn):
    """The HOT PATH. Milliseconds: one indexed read, plus an enqueue on a miss."""
    st: Store = STATE["store"]
    rec = st.get(p.query)
    if rec:
        STATE["hits"] += 1
        return {"hit": True, **rec}
    STATE["misses"] += 1
    queued = await st.enqueue(p.query) if p.enqueue else False
    return {"hit": False, "queued": queued}


@app.post("/put")
async def put(p: PutIn):
    st: Store = STATE["store"]
    await st.put(p.query, p.text, p.sources, p.raw, p.model, p.tokens)
    return {"ok": True}


@app.get("/raw")
async def raw(query: str):
    """Everything the call returned, for re-summarising without re-fetching."""
    return {"raw": STATE["store"].get_raw(query)}


@app.get("/stats")
async def stats():
    st: Store = STATE["store"]
    tot = STATE["hits"] + STATE["misses"]
    out = {"hits": STATE["hits"], "misses": STATE["misses"],
           "hit_rate": round(STATE["hits"] / tot, 4) if tot else None,
           "fetched": STATE["fetched"], "fetch_failures": STATE["fetch_failures"]}
    out.update(st.counts())
    out["snapshots"] = STATE.get("snapshots", 0)
    we: WebEvidence | None = STATE.get("web")
    if we is not None:
        out["breaker_open"] = we.stats().get("web_breaker_open")
    return out


@app.get("/healthz")
async def healthz():
    return {"ok": True, "entries": STATE["store"].counts()["entries"]}


async def _snapshotter():
    """Copy the DB to shared /scratch periodically.

    The store lives on LOCAL disk because SQLite WAL over NFS is unsafe, but these pods are
    preemptible -- a cache that exists only locally is lost with the pod, which would throw
    away exactly the accumulated fetches this service exists to preserve. sqlite3's backup()
    takes a CONSISTENT copy of a live database; `cp` of a WAL database does not.
    """
    st: Store = STATE["store"]
    dest = STATE["snapshot_path"]
    every = float(os.environ.get("EVIDENCE_SNAPSHOT_EVERY_S", "600"))
    if not dest:
        return
    while True:
        await asyncio.sleep(every)
        try:
            os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
            tmp = dest + ".partial"
            async with st.lock:
                out = sqlite3.connect(tmp)
                try:
                    st.db.backup(out)
                finally:
                    out.close()
            os.replace(tmp, dest)          # atomic: a reader never sees a half copy
            STATE["snapshots"] += 1
            logger.info("snapshot -> %s (%s)", dest, st.counts())
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            logger.warning("snapshot failed: %s: %s", type(e).__name__, e)


async def _drainer():
    """Background fetcher. The ONLY place a slow web call happens."""
    st: Store = STATE["store"]
    we: WebEvidence = STATE["web"]
    per_min = int(os.environ.get("EVIDENCE_FETCH_PER_MIN", "60"))
    batch = max(1, int(os.environ.get("EVIDENCE_FETCH_BATCH", "4")))
    while True:
        try:
            items = await st.take(batch)
            if not items:
                await asyncio.sleep(5)
                continue
            # Rate limited on purpose: this shares the TRAPI request cap with the reward
            # judge and the task generator, and starving the reward to warm a cache would
            # be a bad trade.
            async def one(k, q):
                res = await we.search(q)
                if res.error or not (res.text or res.sources):
                    STATE["fetch_failures"] += 1
                    await st.fail(k, res.error or "empty")
                    return
                await st.put(q, res.text, [s.__dict__ for s in res.sources],
                             res.raw, we.model, res.tokens)
                STATE["fetched"] += 1

            await asyncio.gather(*(one(k, q) for k, q in items))
            await asyncio.sleep(max(0.0, 60.0 * len(items) / max(1, per_min)))
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            logger.warning("drainer loop error: %s: %s", type(e).__name__, e)
            await asyncio.sleep(10)


@app.on_event("startup")
async def _startup():
    STATE["task"] = asyncio.create_task(_drainer())
    STATE["snap_task"] = asyncio.create_task(_snapshotter())
    logger.warning("evidence cache ready: %s", STATE["store"].counts())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="/root/evidence_cache.sqlite",
                    help="LOCAL disk. SQLite WAL over NFS is unsafe under concurrency.")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8055)
    ap.add_argument("--api_base", default="http://point.dd.works:18890/v1")
    ap.add_argument("--model", default="gpt-chat-latest_2026-05-28")
    ap.add_argument("--key_file", default="/scratch/sheng/self_evolving/.trapi_key")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--snapshot",
                    default="/scratch/sheng/self_evolving/kb/evidence_cache.sqlite",
                    help="Shared copy, refreshed periodically so a preempted pod does not "
                         "lose the cache. Empty disables.")
    ap.add_argument("--restore", action="store_true",
                    help="Seed a fresh local DB from the snapshot before serving.")
    a = ap.parse_args()

    # Restoring is how the cache survives a pod recreation: the snapshot on /scratch becomes
    # the new local DB. Refuses to clobber an existing local DB, since that one is newer.
    if a.restore and a.snapshot and os.path.exists(a.snapshot) and not os.path.exists(a.db):
        import shutil
        os.makedirs(os.path.dirname(a.db) or ".", exist_ok=True)
        shutil.copy2(a.snapshot, a.db)
        logger.warning("restored local cache from snapshot %s", a.snapshot)

    key = ""
    if os.path.exists(a.key_file):
        key = open(a.key_file).read().strip()
    STATE.update(store=Store(a.db), hits=0, misses=0, fetched=0, fetch_failures=0,
                 snapshot_path=a.snapshot, snapshots=0,
                 web=WebEvidence(api_base=a.api_base, api_key=key, model=a.model,
                                 concurrency=a.concurrency, use_cache=False))
    import uvicorn
    uvicorn.run(app, host=a.host, port=a.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
