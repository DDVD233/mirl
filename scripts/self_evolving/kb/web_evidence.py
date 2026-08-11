#!/usr/bin/env python3
"""Web-search evidence for /retrieve, with a persistent exact-match cache.

WHY. The embedded corpus effectively ends in 2019: 1.0M papers from 2018, 702k from 2019,
then 39k / 5k / 324 / 205 for 2020-2023. Every work the rubrics name from the last five years
is therefore absent -- ACORN (2023), the 2022 ACG GERD guideline, 2024 AUA/SUFU, 2025
ESC/EACTS -- so no amount of ranking, titles or prompt work could make the policy cite them.
TRAPI's /v1/responses endpoint does support a real `web_search` tool (chat/completions does
not; it accepts only 'function' and 'custom'), and it returns url_citation annotations.

CITATIONS, NOT URLS. A web citation is rendered bibliographically, exactly like a Milvus
passage: "Qian ET et al., JAMA 2023 -- Cefepime vs Piperacillin-Tazobactam ...". That works
without any extra request because the metadata table was built from the 2026 PubMed baseline
and holds 4.84M PMIDs from 2023-2025, even though the EMBEDDED corpus stops at 2019. So a
pubmed.ncbi.nlm.nih.gov/<pmid> annotation becomes author/journal/year by local lookup.

COST AND LATENCY, MEASURED. Five timed calls: 6.0 / 6.5 / 6.6 / 7.3 / 12.3 s, median 6.6s,
~8.8k tokens each. Latency tracks how many sub-searches the backend runs (one ~6.6s, two
~12.3s), not a fixed floor. An earlier single call timed out at 90s and I wrongly read that as
typical; it was one outlier in ~8 and it hit a 90s ceiling of my own making.

So this IS viable synchronously on the retrieve path: 800 calls/step at 6.6s with concurrency
16 is ~5.5 min/step and ~1.2 req/s against the ~2000 req/60s TRAPI cap shared with the reward
judge and task generator. The cache is therefore a cost saver (~8.8k tokens a call, ~7M/step
uncached), not a prerequisite -- a repeat query measured 6.0s, so the backend gives us no
caching of its own.

WHY NOT MILVUS FOR THE CACHE. Milvus is approximate-nearest-neighbour, so "exact or very
close" means thresholding cosine -- which is precisely how a cache serves the WRONG evidence
for a different question. A cache false positive is worse than a miss: a miss costs a call, a
false hit silently answers question A with question B's evidence. Keys here are a hash of the
normalised question, exact by construction.

STORAGE lives in evidence_cache_server.py: SQLite WAL on local disk behind a small HTTP API,
because the record keeps the ENTIRE response (tens of KB) so a later re-summarisation never
re-fetches, and a JSONL store would be gigabytes to parse at startup. The local JSONL cache in
this module is only for standalone offline use -- pre-warming and one-off probes.

Each record carries `fetched_at`. Nothing reads it yet; it is what a future refresh pass would
use to expire stale guidance.

  from web_evidence import WebEvidence
  we = WebEvidence(api_base=..., api_key=..., model=...)
  res = await we.search("does piperacillin-tazobactam cause more AKI than cefepime?")
  res.text, res.sources, res.cached
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import socket
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_WARNED: set = set()


def _warned_once(key: str) -> bool:
    """One-shot latch so a degraded dependency logs once, not once per retrieval."""
    if key in _WARNED:
        return True
    _WARNED.add(key)
    return False

CACHE_DIR = os.environ.get("WEB_EVIDENCE_CACHE_DIR",
                           "/scratch/sheng/self_evolving/kb/web_cache")
# Same DB the passage titles come from: one file, one lookup path, one thing to publish.
TITLES_DB = os.environ.get("PUBMED_TITLES_DB",
                           "/scratch/sheng/self_evolving/kb/pubmed_titles.sqlite")

_PMID_RE = re.compile(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)")
_PMC_RE = re.compile(r"pmc\.ncbi\.nlm\.nih\.gov/articles/(PMC\d+)")
# Annotation titles arrive with the source appended by the search backend.
_TITLE_TAIL = re.compile(r"\s*[-|]\s*(PubMed|PMC|NCBI|Food and Drug Administration"
                         r"|UpToDate|Medscape|.{0,24}\.(gov|org|com))\s*$", re.I)

SYSTEM = (
    "You are a clinical evidence retriever. Search the web and report ONLY what the sources "
    "state, as compact bullets a physician can use. Copy every number verbatim -- doses, "
    "thresholds, cutoffs, intervals, percentages. Prefer PubMed, society guidelines and "
    "regulatory labels over secondary summaries, and cite the PubMed record when one exists. "
    "Name the issuing organisation, the guideline or trial, and the year for each claim. "
    "If the sources do not address part of the request, say 'Not covered: <topic>'. "
    "Do not give advice or answer beyond what the sources support. Max 350 words."
)


@dataclass
class WebSource:
    """One citable source, rendered the way a Milvus passage is."""
    title: str = ""
    pmid: str = ""
    journal: str = ""
    year: str = ""
    author: str = ""
    n_authors: str = ""
    url: str = ""            # kept for provenance/debugging; never shown to the model

    def citation(self) -> str:
        """"Qian ET et al., JAMA 2023" -- or "" when no bibliographic fields are known."""
        try:
            many = int(self.n_authors or 0) > 1
        except (TypeError, ValueError):
            many = False
        who = f"{self.author} et al." if (self.author and many) else (self.author or "")
        where = " ".join(x for x in (self.journal, self.year) if x)
        return ", ".join(x for x in (who, where) if x)

    def render(self) -> str:
        """A source line for the brief. Deliberately URL-FREE: a citation is
        author/journal/year/title, and a bare link is not something a clinician can check
        against a rubric that asks for 'the 2022 ACG guideline'."""
        cite = self.citation()
        bits = " -- ".join(x for x in (cite, self.title) if x)
        if self.pmid:
            bits += f" (PMID {self.pmid})"
        return bits or self.title


@dataclass
class WebResult:
    text: str = ""
    sources: list[WebSource] = field(default_factory=list)
    cached: bool = False
    fetched_at: str = ""
    error: str = ""
    # THE COMPLETE RESPONSE, kept verbatim. A brief can always be re-summarised from this
    # later; it can never be un-thrown-away. One call costs ~13k tokens, so re-fetching
    # because we stored only a summary would be the expensive mistake.
    raw: dict | None = None
    tokens: int = 0


def normalize_query(q: str) -> str:
    """Canonical form for cache keys: case, whitespace and edge punctuation only.

    Deliberately conservative. Anything cleverer (stemming, stopword removal, synonyms)
    starts merging questions that differ in a clinically material way -- "eGFR below 30" and
    "eGFR below 45" must never share a cache entry.
    """
    s = (q or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip(" .?!,:;\"'")


def cache_key(q: str) -> str:
    return hashlib.sha256(normalize_query(q).encode("utf8")).hexdigest()[:32]


class WebCache:
    """Append-only JSONL cache, shared across hosts and sessions.

    One shard per host so two gen servers never interleave writes into one file; every shard
    is read by everyone at startup, so a lookup sees what other hosts fetched. Appends are
    single short lines opened in append mode, which is the one write pattern that is safe on
    this NFS mount without locking.
    """

    def __init__(self, directory: str = CACHE_DIR):
        self.dir = directory
        self.mem: dict[str, dict] = {}
        self.hits = 0
        self.misses = 0
        self.writes = 0
        host = re.sub(r"\W+", "_", socket.gethostname())[:40] or "unknown"
        self.shard = os.path.join(self.dir, f"cache_{host}.jsonl")
        self._load()

    def _load(self) -> None:
        try:
            os.makedirs(self.dir, exist_ok=True)
        except OSError as e:
            logger.warning("web cache dir unusable (%s); running memory-only", e)
            return
        n_files = 0
        for name in sorted(os.listdir(self.dir)):
            if not name.startswith("cache_") or not name.endswith(".jsonl"):
                continue
            n_files += 1
            path = os.path.join(self.dir, name)
            try:
                with open(path, errors="ignore") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            continue          # a torn final line is expected; skip it
                        k = rec.get("key")
                        if k:
                            self.mem[k] = rec
            except OSError as e:
                logger.warning("could not read web cache shard %s: %s", name, e)
        logger.warning("web evidence cache: %d entries from %d shard(s) in %s",
                       len(self.mem), n_files, self.dir)

    def get(self, q: str) -> dict | None:
        rec = self.mem.get(cache_key(q))
        if rec is None:
            self.misses += 1
            return None
        self.hits += 1
        return rec

    def put(self, q: str, text: str, sources: list[dict]) -> None:
        rec = {
            "key": cache_key(q),
            "query": q[:2000],
            "text": text,
            "sources": sources,
            # Unused today. A refresh pass would use it to expire guidance that has moved on.
            "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        self.mem[rec["key"]] = rec
        try:
            with open(self.shard, "a") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self.writes += 1
        except OSError as e:
            # An unwritable cache must not fail a rollout; the entry still serves in-memory.
            logger.warning("web cache append failed (%s)", e)

    def stats(self) -> dict:
        tot = self.hits + self.misses
        return {"web_cache_entries": len(self.mem), "web_cache_hits": self.hits,
                "web_cache_misses": self.misses, "web_cache_writes": self.writes,
                "web_cache_hit_rate": round(self.hits / tot, 4) if tot else None}


class WebEvidence:
    """Search the web through TRAPI /v1/responses, cache the result, cite it properly."""

    def __init__(self, api_base: str, api_key: str, model: str,
                 concurrency: int | None = None, cache_dir: str = CACHE_DIR,
                 timeout_s: float | None = None, titles_db: str = TITLES_DB,
                 use_cache: bool = True):
        self.api_base = (api_base or "").rstrip("/")
        self.api_key = api_key
        self.model = model
        # The evidence cache SERVICE owns persistence and passes use_cache=False; the local
        # file cache exists only for standalone/offline use (pre-warming, one-off probes).
        self.cache = WebCache(cache_dir) if use_cache else None
        self.titles_db = titles_db
        self.sem = asyncio.Semaphore(
            int(concurrency or os.environ.get("WEB_EVIDENCE_CONCURRENCY", "8")))
        self.timeout_s = float(timeout_s
                               or os.environ.get("WEB_EVIDENCE_TIMEOUT", "60"))
        # CIRCUIT BREAKER. Web search shares the TRAPI request cap with the reward judge and
        # the task generator, so a rate-limit storm here would degrade the reward itself.
        # After `_breaker_trip` consecutive failures the tool is skipped for `_breaker_cool`
        # seconds and /retrieve serves Milvus alone -- the same degrade-not-fail shape as the
        # summarizer fallback.
        self._fails = 0
        self._open_until = 0.0
        self._breaker_trip = int(os.environ.get("WEB_EVIDENCE_BREAKER_FAILS", "5"))
        self._breaker_cool = float(os.environ.get("WEB_EVIDENCE_BREAKER_COOL", "120"))
        self.calls = 0
        self.failures = 0
        self.breaker_skips = 0
        self._db = None

    # -- provenance -------------------------------------------------------
    def _conn(self):
        """One reused read-only connection, not one per PMID.

        Reopening a 4.86 GB database on NFS per citation is pure latency on a path that runs
        per retrieval.
        """
        if self._db is not None:
            return self._db or None
        import sqlite3
        if not os.path.exists(self.titles_db):
            self._db = False
            return None
        try:
            self._db = sqlite3.connect(f"file:{self.titles_db}?mode=ro", uri=True,
                                       check_same_thread=False, timeout=5)
        except Exception as e:  # noqa: BLE001
            logger.warning("titles DB unusable for web citations (%s)", e)
            self._db = False
            return None
        return self._db

    def _meta(self, pmid: str) -> dict:
        """journal/year/author for a PMID. meta.pmid is the PRIMARY KEY, so this is an index
        lookup.

        It deliberately does NOT read titles.title by pmid: that table is keyed on the chunk
        id, so a pmid predicate is a FULL SCAN of 19.5M rows over NFS. Doing it per citation
        blocked the cache service's event loop hard enough that /stats stopped answering while
        a fetch was in flight. The annotation already carries a usable title, so the canonical
        one is not worth an unindexed scan.
        """
        if not pmid:
            return {}
        con = self._conn()
        if not con:
            return {}
        try:
            r = con.execute("SELECT journal, year, author, n_authors FROM meta WHERE pmid=?",
                            (pmid,)).fetchone()
        except Exception as e:  # noqa: BLE001
            logger.debug("web citation meta lookup failed for %s: %s", pmid, e)
            return {}
        if not r:
            return {}
        return {"journal": r[0] or "", "year": r[1] or "", "author": r[2] or "",
                "n_authors": r[3] or ""}

    async def _sources_from(self, annotations: list[dict]) -> list[WebSource]:
        seen, out = set(), []
        for a in annotations:
            url = (a.get("url") or "").strip()
            title = _TITLE_TAIL.sub("", (a.get("title") or "").strip())
            m = _PMID_RE.search(url)
            pmid = m.group(1) if m else ""
            key = pmid or url or title
            if not key or key in seen:
                continue
            seen.add(key)
            s = WebSource(title=title, pmid=pmid, url=url)
            if pmid:
                # to_thread: a handful of index lookups is fast, but "fast" on an NFS-backed
                # 4.86 GB file is not a promise worth making on an event loop that also
                # serves /lookup for every rollout.
                meta = await asyncio.to_thread(self._meta, pmid)
                s.journal = meta.get("journal", "")
                s.year = meta.get("year", "")
                s.author = meta.get("author", "")
                s.n_authors = meta.get("n_authors", "")
            out.append(s)
        return out

    # -- search -----------------------------------------------------------
    async def search(self, question: str) -> WebResult:
        """Never raises. A failure returns an empty WebResult with `error` set, so the
        caller keeps its Milvus passages and the rollout survives."""
        if not question or not self.api_base:
            return WebResult(error="not configured")

        hit = self.cache.get(question) if self.cache else None
        if hit is not None:
            return WebResult(text=hit.get("text") or "",
                             sources=[WebSource(**{k: v for k, v in s.items()
                                                   if k in WebSource.__annotations__})
                                      for s in (hit.get("sources") or [])],
                             cached=True, fetched_at=hit.get("fetched_at") or "")

        if time.time() < self._open_until:
            self.breaker_skips += 1
            return WebResult(error="breaker open")

        try:
            # The budget covers QUEUE WAIT PLUS the call, not just the call. With the
            # semaphore acquired inside the deadline, a request that arrives while 16 calls
            # are in flight would wait an unbounded time before its own 60s even started --
            # precisely the "rate limited, so we hang" case this bound exists to prevent.
            # A timeout is not a failure of retrieval: the caller keeps its Milvus passages
            # and summarises from those, and the miss is already enqueued on the cache
            # service, so the next asker gets a hit instead of paying again.
            text, anns, raw, tokens = await asyncio.wait_for(
                self._call_guarded(question), timeout=self.timeout_s)
        except Exception as e:  # noqa: BLE001
            self.failures += 1
            self._fails += 1
            if self._fails >= self._breaker_trip:
                self._open_until = time.time() + self._breaker_cool
                self._fails = 0
                logger.warning("web evidence breaker OPEN for %.0fs after %d failures "
                               "(%s) -- /retrieve continues on Milvus alone",
                               self._breaker_cool, self._breaker_trip, type(e).__name__)
            return WebResult(error=f"{type(e).__name__}: {str(e)[:120]}")

        self._fails = 0
        self.calls += 1
        sources = await self._sources_from(anns)
        if self.cache:
            self.cache.put(question, text, [s.__dict__ for s in sources])
        return WebResult(text=text, sources=sources, cached=False, raw=raw, tokens=tokens)

    async def _call_guarded(self, question: str) -> tuple[str, list[dict], dict, int]:
        """Concurrency-limited call, inside the caller's deadline so queueing counts too."""
        async with self.sem:
            return await self._call(question)

    async def _call(self, question: str) -> tuple[str, list[dict], dict, int]:
        import httpx
        body = {"model": self.model,
                "instructions": SYSTEM,
                "input": question,
                "tools": [{"type": "web_search"}]}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            r = await client.post(f"{self.api_base}/responses", json=body, headers=headers)
            r.raise_for_status()
            d = r.json()
        if d.get("error"):
            raise RuntimeError(str(d["error"])[:200])
        texts, anns = [], []
        for item in d.get("output") or []:
            for c in item.get("content") or []:
                if c.get("type") in ("output_text", "text"):
                    texts.append(c.get("text") or "")
                anns.extend(c.get("annotations") or [])
        tokens = int((d.get("usage") or {}).get("total_tokens") or 0)
        return "\n".join(t for t in texts if t).strip(), anns, d, tokens

    def stats(self) -> dict:
        s = self.cache.stats() if self.cache else {}
        s.update(web_calls=self.calls, web_failures=self.failures,
                 web_breaker_skips=self.breaker_skips,
                 web_breaker_open=time.time() < self._open_until)
        return s


class EvidenceCacheClient:
    """Thin client for evidence_cache_server: shared by the GENERATOR and the SOLVER.

    Both go through one service so a question fetched while minting a task is already cached
    when a rollout later retrieves for it, and vice versa. That sharing is most of the value:
    the curriculum and the policy ask about the same clinical topics by construction, since
    the tasks ARE the topics.

    Every method degrades instead of raising. If the service is down, `get` returns None and
    the caller proceeds on Milvus alone -- retrieval sits on the generation critical path, so a
    cache outage must cost evidence quality, never a rollout.
    """

    def __init__(self, base_url: str = "", timeout_s: float = 10.0):
        self.base = (base_url or os.environ.get("EVIDENCE_CACHE_URL", "")).rstrip("/")
        self.timeout_s = timeout_s
        self.enabled = bool(self.base)
        self.hits = 0
        self.misses = 0
        self.errors = 0

    async def get(self, query: str, enqueue: bool = True) -> WebResult | None:
        """Cache lookup. None means "not cached" -- including when the service is down."""
        if not self.enabled or not query:
            return None
        try:
            import httpx
            async with httpx.AsyncClient(timeout=self.timeout_s) as c:
                r = await c.post(f"{self.base}/lookup",
                                 json={"query": query, "enqueue": enqueue})
                r.raise_for_status()
                d = r.json()
        except Exception as e:  # noqa: BLE001
            self.errors += 1
            if not _warned_once("cache_client"):
                logger.warning("evidence cache unreachable at %s (%s: %s); continuing "
                               "without web evidence", self.base, type(e).__name__, e)
            return None
        if not d.get("hit"):
            self.misses += 1
            return None
        self.hits += 1
        return WebResult(
            text=d.get("text") or "",
            sources=[WebSource(**{k: v for k, v in s.items()
                                  if k in WebSource.__annotations__})
                     for s in (d.get("sources") or [])],
            cached=True, fetched_at=d.get("fetched_at") or "",
            tokens=int(d.get("tokens") or 0))

    async def put(self, query: str, res: WebResult, model: str = "") -> None:
        """Store a result fetched directly (synchronous path), so the next asker is free."""
        if not self.enabled or not query or res.error:
            return
        try:
            import httpx
            async with httpx.AsyncClient(timeout=self.timeout_s) as c:
                await c.post(f"{self.base}/put", json={
                    "query": query, "text": res.text,
                    "sources": [s.__dict__ for s in res.sources],
                    "raw": res.raw, "model": model, "tokens": res.tokens})
        except Exception as e:  # noqa: BLE001
            self.errors += 1
            logger.debug("evidence cache put failed: %s", e)

    def stats(self) -> dict:
        tot = self.hits + self.misses
        return {"evidence_cache_hits": self.hits, "evidence_cache_misses": self.misses,
                "evidence_cache_errors": self.errors,
                "evidence_cache_hit_rate": round(self.hits / tot, 4) if tot else None}
