"""Source-aware ranking for the medical-knowledge retriever.

Single definition of the retrieval-quality policy, imported by BOTH the
generation server's ``/retrieve`` endpoint (what the RL rollouts call) and the
offline tools (trace generation, evaluation). Keeping one copy is the point:
if the passages SFT traces are built from differ from the passages RL rollouts
see, the warm start teaches a distribution that does not exist at train time.

The policy exists because dense cosine alone ranks this KB badly. Of its ~57M
rows, ~53% are general (non-medical) Wikipedia, 99% of the wikidoc rows are bare
titles carrying no facts, and ~2.85M rows are MCQ/VQA stems (several referring to
an image the model is never shown). Every retrieval configuration measured before
this change scored BELOW the no-tool baseline, and the per-case analysis blamed
passage quality first. So: exclude the structurally useless rows in the query,
over-fetch, re-rank by cosine * source prior, and cap per-source so one source
cannot fill the whole budget.
"""

import json
import logging
import os
import re
from collections import Counter

logger = logging.getLogger(__name__)

# Relative trust per source (multiplicative on cosine; 1.0 = trust fully).
SOURCE_WEIGHTS: dict[str, float] = {
    # Curated clinical references — added specifically because HealthBench rewards
    # current guidance, exact dosing, patient-facing explanation and codes.
    "statpearls": 1.00,     # peer-reviewed clinical reviews, sectioned
    "dailymed": 1.00,       # FDA labels: dosing / contraindications / warnings
    "icd10cm": 1.00,        # official code tables
    "medlineplus": 1.00,    # NIH consumer-health topics
    "medrag_textbook": 1.00,
    "wikidoc": 0.95,        # only enriched articles survive EXCLUDE_EXPR
    "medrag_pubmed": 0.85,  # abstract fragments: narrow, often a single study
    "pubmedqa": 0.80,
    "medrag_wiki": 0.55,    # general-domain Wikipedia — the biggest diluter
}
DEFAULT_WEIGHT = 0.75

# Applied inside the Milvus query so junk never consumes the fetch budget:
#   content_type == "title"  -> 187,987 bare wikidoc titles (median 25 chars)
#   mirage / climb / pmc_vqa -> MCQ stems and image-grounded VQA rows
EXCLUDE_EXPR = 'content_type != "title" and source_dataset not in ["mirage", "climb", "pmc_vqa"]'

FETCH_MULT = 8        # over-fetch factor before re-ranking
MAX_PER_SOURCE = 3    # diversity cap within the returned top-k
MIN_CHARS = 80        # below this a passage states no usable fact

# Multi-query defaults. Measured on 93 LATENT criteria (kb/multiquery_probe.py):
# the shipped single query supplies the graded fact 10.8% of the time; 2 queries at
# k=12 reach 35.9% (McNemar p<1e-4); 6 queries at k=4 reach 34.8% for 28% fewer
# tokens than 1 query at k=24 (30.4%). The gain comes from asking for DIFFERENT
# FACTS, not from a deeper single list — but >=2 queries is necessary, depth alone
# does not get there.
PER_QUERY_K = int(os.environ.get("RETRIEVE_TOP_K", "5"))       # depth per sub-query
MERGE_TOTAL = int(os.environ.get("RETRIEVE_TOTAL", "16"))      # merged budget
MERGE_PER_SOURCE = int(os.environ.get("RETRIEVE_MERGE_PER_SOURCE", "5"))
DEDUP_JACCARD = float(os.environ.get("RETRIEVE_DEDUP_JACCARD", "0.6"))
MAX_QUERIES = int(os.environ.get("RETRIEVE_MAX_QUERIES", "4"))

# Per-passage character caps in the tool response. Curated rows are written as
# self-contained units (a dosing section, a code block), so truncating them at the
# snippet cap would cut the very fact they were added for.
PASSAGE_CHARS = 600
WIKIDOC_CHARS = 1600
CURATED_CHARS = 1600
CURATED_SOURCES = {"statpearls", "dailymed", "medlineplus", "icd10cm"}


class RetrieveConfig:
    """Env-overridable knobs, re-read per call so a running server can be retuned."""

    def __init__(self):
        self.weights = dict(SOURCE_WEIGHTS)
        raw = os.environ.get("RETRIEVE_SOURCE_WEIGHTS", "").strip()
        if raw:
            try:
                self.weights.update({str(k): float(v) for k, v in json.loads(raw).items()})
            except Exception as e:
                logger.warning(f"bad RETRIEVE_SOURCE_WEIGHTS ({e}); using defaults")
        self.exclude_expr = os.environ.get("RETRIEVE_EXCLUDE_EXPR", EXCLUDE_EXPR)
        self.fetch_mult = max(1, int(os.environ.get("RETRIEVE_FETCH_MULT", FETCH_MULT)))
        self.max_per_source = max(1, int(os.environ.get("RETRIEVE_MAX_PER_SOURCE", MAX_PER_SOURCE)))
        self.min_chars = int(os.environ.get("RETRIEVE_MIN_CHARS", MIN_CHARS))


def passage_cap(source: str) -> int:
    if source in CURATED_SOURCES:
        return CURATED_CHARS
    if source == "wikidoc":
        return WIKIDOC_CHARS
    return PASSAGE_CHARS


def rank_hits(hits: list[dict], top_k: int, cfg: RetrieveConfig | None = None) -> list[dict]:
    """Re-rank raw Milvus hits by cosine * source prior, with a per-source cap.

    `hits` entries use the generation-server shape: {source, text, answer,
    question, score, ...}. Returns at most `top_k` passage dicts."""
    cfg = cfg or RetrieveConfig()
    scored = []
    for h in hits:
        text = (h.get("text") or h.get("answer") or h.get("question") or "").strip()
        if len(text) < cfg.min_chars:
            continue
        src = h.get("source", "?")
        w = cfg.weights.get(src, DEFAULT_WEIGHT)
        scored.append((float(h.get("score", 0.0)) * w, h, text, src))
    scored.sort(key=lambda t: t[0], reverse=True)

    out: list[dict] = []
    per_source: dict[str, int] = {}
    for adj, h, text, src in scored:
        if len(out) >= top_k:
            break
        if per_source.get(src, 0) >= cfg.max_per_source:
            continue
        per_source[src] = per_source.get(src, 0) + 1
        out.append({
            "source": src,
            # entry_id is carried through so multi-query merging can dedup EXACTLY.
            # Without it the merge falls back to text matching, which is both slower
            # and wrong for passages truncated at different caps.
            "entry_id": h.get("entry_id", ""),
            "text": text[:passage_cap(src)],
            "score": h.get("score", 0.0),
            "adjusted_score": round(adj, 4),
        })
    return out


# --------------------------------------------------------------------------
# Multi-query merge. Single definition shared by /retrieve (RL rollouts), the
# offline trace generator, and kb/multiquery_probe.py (where the recovery numbers
# above were measured) — if these diverge, the warm start teaches a passage
# distribution that does not exist at train time.
# --------------------------------------------------------------------------
def _norm(t: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", (t or "").lower())


def shingles(t: str, n: int = 8) -> set:
    w = _norm(t).split()
    if len(w) < n:
        return {" ".join(w)}
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def is_stub(text: str) -> bool:
    """Heading/TOC row: many short colon-terminated lines, no sentences.

    6.5% of returned passages were these — bare section headings that clear
    EXCLUDE_EXPR (content_type != "title") and MIN_CHARS as multi-line lists while
    stating no fact. One audited rollout burned 3 of its 6 slots on byte-identical
    copies of one such row.
    """
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return True
    colonish = sum(1 for ln in lines if ln.endswith(":") or len(ln.split()) <= 6)
    body = " ".join(lines)
    n_sent = len(re.findall(r"[a-z]{3,}[.!?](?:\s|$)", body))
    return (colonish >= max(2, 0.7 * len(lines))) and n_sent <= 1


def merge_ranked(per_query: list[list[dict]], total: int = 0,
                 per_source_cap: int = 0, dedup_jaccard: float = DEDUP_JACCARD,
                 drop_stubs: bool = True) -> tuple[list[dict], dict]:
    """Round-robin across sub-queries, then global dedup and a global source cap.

    Round-robin by RANK DEPTH (every query's #1 before any query's #2) is what makes
    the facets share the budget; concatenating and re-sorting by score would let one
    query's long tail crowd the others out. Returns (passages, merge_stats)."""
    total = total or MERGE_TOTAL
    per_source_cap = per_source_cap or MERGE_PER_SOURCE
    out: list[dict] = []
    seen_sh: list[set] = []
    seen_id: set = set()
    per_src: Counter = Counter()
    stats: Counter = Counter()
    depth = max((len(x) for x in per_query), default=0)
    for d in range(depth):
        for q in per_query:
            if d >= len(q) or len(out) >= total:
                continue
            p = q[d]
            txt = p.get("text") or ""
            if p.get("entry_id") and p["entry_id"] in seen_id:
                stats["dup_exact"] += 1
                continue
            if drop_stubs and is_stub(txt):
                stats["stub"] += 1
                continue
            sh = shingles(txt)
            if any(len(sh & prev) / max(1, min(len(sh), len(prev))) >= dedup_jaccard
                   for prev in seen_sh):
                stats["dup_near"] += 1
                continue
            if per_src[p.get("source", "?")] >= per_source_cap:
                stats["src_cap"] += 1
                continue
            per_src[p.get("source", "?")] += 1
            seen_sh.append(sh)
            if p.get("entry_id"):
                seen_id.add(p["entry_id"])
            out.append(p)
    stats["kept"] = len(out)
    return out, dict(stats)


def format_passages(passages: list[dict]) -> str:
    """The exact block handed back to the model as the tool response."""
    if not passages:
        return "No relevant passages found in the medical knowledge base."
    return "\n\n".join(
        f"[passage {i + 1} | source={p['source']}]\n{p['text']}"
        for i, p in enumerate(passages)
    )


# --------------------------------------------------------------------------
# Offline helper: embed + search + rank without the generation server.
# --------------------------------------------------------------------------
def local_search(query: str, top_k: int = 5,
                 milvus_uri: str = "http://localhost:19531",
                 milvus_token: str = "root:Milvus",
                 collection: str = "medical_knowledge_v2",
                 embed_api_base: str = "http://localhost:18001/v1",
                 embed_api_key: str = "EMPTY",
                 embed_model: str = "Qwen/Qwen3-VL-Embedding-2B",
                 cfg: RetrieveConfig | None = None) -> tuple[list[dict], str]:
    """Same ranking as /retrieve, run directly against Milvus.

    Used by offline trace generation so traces are built from exactly the
    passages a rollout would have seen."""
    import httpx
    from pymilvus import MilvusClient

    cfg = cfg or RetrieveConfig()
    r = httpx.post(f"{embed_api_base.rstrip('/')}/embeddings",
                   json={"model": embed_model, "input": [query[:2000]]},
                   headers={"Authorization": f"Bearer {embed_api_key}"}, timeout=60)
    r.raise_for_status()
    vec = r.json()["data"][0]["embedding"]

    client = MilvusClient(uri=milvus_uri, token=milvus_token)
    res = client.search(
        collection_name=collection, data=[vec], limit=top_k * cfg.fetch_mult,
        filter=cfg.exclude_expr,
        output_fields=["source_dataset", "modality", "content_type",
                       "text_content", "question", "answer", "entry_id"],
    )
    hits = []
    for hit_list in res:
        for hit in hit_list:
            e = hit["entity"]
            hits.append({
                "source": e.get("source_dataset", ""),
                "content_type": e.get("content_type", ""),
                "text": e.get("text_content", ""),
                "question": e.get("question", ""),
                "answer": e.get("answer", ""),
                "entry_id": e.get("entry_id", ""),
                "score": hit["distance"],
            })
    passages = rank_hits(hits, top_k, cfg)
    return passages, format_passages(passages)
