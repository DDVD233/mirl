"""A/B the OLD vs NEW retrieval policy on real HealthBench-style queries.

OLD = what every measured retrieval run used: plain dense top-k over all 57M rows,
      no source filter, no re-ranking, 600-char cap (1600 for wikidoc).
NEW = kb/retrieval.py: junk rows excluded in-query, over-fetch, re-rank by
      cosine * source prior, per-source diversity cap, larger cap for curated rows.

Reports per-query source mixes plus a "usable passage" rate, where a passage counts
as usable if it is long enough to state a fact and does not come from a source that
is structurally noise for a text-only clinical question. This is the check that was
missing before: retrieval was shipped on cosine alone and every configuration
scored below the no-tool baseline.

    python scripts/self_evolving/kb/compare_retrieval.py [--top_k 5]
"""

import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kb.retrieval import RetrieveConfig, format_passages, rank_hits  # noqa: E402

# Real clinician turns sampled every 45th row from healthbench_pro_val.parquet
# (used ONLY to inspect retrieval quality — nothing here is trained on).
QUERIES = [
    "Under hvilke omstændigheder vil en 1,5 årig dreng med hyppige AOM blive tilbudt en polypoperation?",
    "Discharge instructions: 24-year-old female, severe malaria with early 3rd trimester pregnancy, "
    "artemisinin injections, folic acid 5mg daily, primaquine 15mg daily for 14 days",
    "Resume las guías para el manejo de dislipidemia ACC/AHA, medidas preventivas, inicio temprano "
    "de medicamentos y laboratorios de diagnóstico y seguimiento en adultos",
    "Treatment chart for a child admitted with infected eczema of face/neck/torso, background flexural "
    "eczema, culture positive for fully sensitive staph",
    "R-ICE protocol including dosing and side effects",
    "Inpatient CHF on diuretics with reduced urine output and rising creatinine",
    "Latest guidelines for peripheral facial nerve palsy",
    "Vesicovaginal fistula after emergency caesarean with subtotal hysterectomy, continuous urine "
    "leakage, referral for CT urogram and cystoscopy",
    "Open fracture of the left tibia after a motorcycle accident: antibiotic regimen, dosage and duration",
    "6-month-old exclusively breastfed with bloody mucoid watery stool, postprandial vomiting, fever",
    "Explain to a patient why antidepressant benefit persists after stopping",
    "Exact adult dose of amoxicillin for streptococcal pharyngitis and the penicillin-allergic alternative",
    "ICD-10-CM code for type 2 diabetes mellitus with diabetic polyneuropathy",
    "Warfarin INR 7.2 without bleeding: vitamin K dose and monitoring interval",
]

# Sources that cannot help a text-only clinical question: image-grounded VQA rows,
# bare MCQ stems, and general-domain Wikipedia.
NOISE_SOURCES = {"climb", "pmc_vqa", "mirage", "medrag_wiki"}
USABLE_MIN_CHARS = 200


def embed(query, base, key, model):
    import httpx
    r = httpx.post(f"{base.rstrip('/')}/embeddings",
                   json={"model": model, "input": [query[:2000]]},
                   headers={"Authorization": f"Bearer {key}"}, timeout=60)
    r.raise_for_status()
    return r.json()["data"][0]["embedding"]


def raw_hits(client, collection, vec, limit, filter_expr=""):
    kw = dict(collection_name=collection, data=[vec], limit=limit,
              output_fields=["source_dataset", "content_type", "text_content",
                             "question", "answer", "entry_id"])
    if filter_expr:
        kw["filter"] = filter_expr
    out = []
    for hl in client.search(**kw):
        for h in hl:
            e = h["entity"]
            out.append({
                "source": e.get("source_dataset", ""),
                "content_type": e.get("content_type", ""),
                "text": e.get("text_content", ""),
                "question": e.get("question", ""),
                "answer": e.get("answer", ""),
                "score": h["distance"],
            })
    return out


def old_policy(hits, top_k):
    """Plain dense top-k with the original per-passage caps."""
    out = []
    for h in hits[:top_k]:
        text = (h.get("text") or h.get("answer") or h.get("question") or "").strip()
        if not text:
            continue
        cap = 1600 if h.get("source") == "wikidoc" else 600
        out.append({"source": h.get("source", "?"), "text": text[:cap],
                    "score": h.get("score", 0.0)})
    return out


def usable(p):
    return p["source"] not in NOISE_SOURCES and len(p["text"]) >= USABLE_MIN_CHARS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--milvus_uri", default="http://localhost:19531")
    ap.add_argument("--milvus_token", default="root:Milvus")
    ap.add_argument("--collection", default="medical_knowledge_v2")
    ap.add_argument("--embed_api_base", default="http://localhost:18001/v1")
    ap.add_argument("--embed_api_key", default="EMPTY")
    ap.add_argument("--embed_model", default="Qwen/Qwen3-VL-Embedding-2B")
    ap.add_argument("--show", type=int, default=3, help="queries to print passages for")
    args = ap.parse_args()

    from pymilvus import MilvusClient
    client = MilvusClient(uri=args.milvus_uri, token=args.milvus_token)
    cfg = RetrieveConfig()

    old_src, new_src = Counter(), Counter()
    old_ok = new_ok = total = 0
    old_chars = new_chars = 0

    for qi, q in enumerate(QUERIES):
        vec = embed(q, args.embed_api_base, args.embed_api_key, args.embed_model)
        old = old_policy(raw_hits(client, args.collection, vec, args.top_k), args.top_k)
        new = rank_hits(raw_hits(client, args.collection, vec,
                                 args.top_k * cfg.fetch_mult, cfg.exclude_expr),
                        args.top_k, cfg)
        for p in old:
            old_src[p["source"]] += 1
            old_ok += usable(p)
            old_chars += len(p["text"])
        for p in new:
            new_src[p["source"]] += 1
            new_ok += usable(p)
            new_chars += len(p["text"])
        total += args.top_k

        if qi < args.show:
            print(f"\n{'=' * 78}\nQUERY: {q[:110]}")
            print(f"--- OLD ({sum(usable(p) for p in old)}/{len(old)} usable) ---")
            for p in old:
                print(f"  {p['source']:16s} {len(p['text']):5d}c  {p['text'][:88]!r}")
            print(f"--- NEW ({sum(usable(p) for p in new)}/{len(new)} usable) ---")
            for p in new:
                print(f"  {p['source']:16s} {len(p['text']):5d}c  {p['text'][:88]!r}")

    print(f"\n{'=' * 78}\nAGGREGATE over {len(QUERIES)} queries x top-{args.top_k} = {total} passages")
    print(f"  usable passages : OLD {old_ok:3d}/{total} ({old_ok / total:.0%})   "
          f"NEW {new_ok:3d}/{total} ({new_ok / total:.0%})")
    print(f"  mean chars/passage: OLD {old_chars / total:,.0f}   NEW {new_chars / total:,.0f}")
    print(f"  OLD sources: {dict(old_src.most_common())}")
    print(f"  NEW sources: {dict(new_src.most_common())}")


if __name__ == "__main__":
    main()
