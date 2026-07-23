"""Enrich retrieved WikiDoc entries with full article text (post-run pass).

During SFT/RL, generation_server logs every retrieved wikidoc title to
`<log_dir>/wikidoc_retrieved_titles.jsonl`. After a run, this script:

  1. dedups those titles,
  2. for each, enumerates its WikiDoc sub-pages (MediaWiki allpages apprefix) and
     fetches + cleans their wikitext, concatenating into one plain-text article
     (sub-pages are short, so we return them all at once),
  3. saves the full original to `--fulltext_dir/<title>.txt`,
  4. if the cleaned article exceeds the Milvus text_content limit (VARCHAR 2000),
     summarizes it to a compact clinical reference with the same teacher model and
     stores the SUMMARY; otherwise stores the cleaned text as-is,
  5. re-embeds on (title + stored text) and upserts the wikidoc row so future
     retrieval matches on CONTENT, not just the bare title.

Only the handful of titles that actually surfaced during the run are fetched, so
this builds a "needed articles" WikiDoc corpus rather than crawling all ~200k.

Run where wikidoc.org + mib (embed/Milvus) + the teacher are reachable (server 1):
  python scripts/self_evolving/wikidoc_enrich.py \
    --titles_file /scratch/.../logs_healthbench_rubric/<EXP>/wikidoc_retrieved_titles.jsonl \
    --milvus_uri http://mib.media.mit.edu:19531 --milvus_token root:Milvus \
    --milvus_collection medical_knowledge_v2 \
    --embed_api_base http://mib.media.mit.edu:18001/v1 --embed_model Qwen/Qwen3-VL-Embedding-2B \
    --api_base http://point.dd.works:18184/v1 --api_key $(cat .../.climb_teacher_key) \
    --model_name Qwen/Qwen3.6-27B \
    --fulltext_dir /scratch/sheng/self_evolving/wikidoc_fulltext
"""

import argparse
import asyncio
import json
import os
import re
import sys
import urllib.parse

import httpx

WIKI_API = "https://www.wikidoc.org/api.php"
DB_TEXT_CAP = 1900          # keep under the VARCHAR(2000) text_content limit
SUMMARY_SYS = (
    "You are a medical reference editor. Summarize the following WikiDoc article into a "
    "COMPACT clinical reference of at most 320 words. Preserve concrete facts: definition, "
    "causes/pathophysiology, diagnosis, management (drugs, doses, thresholds), and key "
    "cautions/contraindications. Plain prose, no markup, no headers list. Be information-dense."
)


# ---- wikitext cleaning ------------------------------------------------------- #
_RE_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_RE_REF = re.compile(r"<ref[^>]*>.*?</ref>|<ref[^>]*/>", re.DOTALL | re.IGNORECASE)
_RE_TAG = re.compile(r"<[^>]+>")
_RE_FILE = re.compile(r"\[\[(?:File|Image):[^\]]*\]\]", re.IGNORECASE)
_RE_LINK = re.compile(r"\[\[(?:[^\]|]*\|)?([^\]]+)\]\]")
_RE_EXTLINK = re.compile(r"\[https?://[^\s\]]+\s*([^\]]*)\]")
_RE_HEADER = re.compile(r"^\s*={2,}\s*(.*?)\s*={2,}\s*$", re.MULTILINE)
_RE_BOILER = re.compile(r"__[A-Z]+__|\{\{[^}]*\}\}")


def _strip_templates(s: str) -> str:
    # remove {{...}} including simple nesting
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r"\{\{[^{}]*\}\}", "", s)
    return s


def _strip_tables(s: str) -> str:
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r"\{\|[^{}]*?\|\}", "", s, flags=re.DOTALL)
    return s


def clean_wikitext(wt: str) -> str:
    if not wt:
        return ""
    s = _RE_COMMENT.sub("", wt)
    s = _RE_REF.sub("", s)
    s = _strip_tables(s)
    s = _strip_templates(s)
    s = _RE_FILE.sub("", s)
    s = _RE_EXTLINK.sub(r"\1", s)
    s = _RE_LINK.sub(r"\1", s)
    s = _RE_HEADER.sub(r"\n\1: ", s)
    s = _RE_TAG.sub("", s)
    s = _RE_BOILER.sub("", s)
    s = re.sub(r"''+", "", s)                # bold/italic markers
    s = re.sub(r"^[\*#:;]+\s*", "", s, flags=re.MULTILINE)  # list bullets
    s = re.sub(r"\n{2,}", "\n", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    # drop nav lines
    lines = [ln.strip() for ln in s.splitlines()
             if ln.strip() and "click here" not in ln.lower()
             and "patient information" not in ln.lower()]
    return "\n".join(lines).strip()


async def _api(client, params):
    params = {**params, "format": "json"}
    r = await client.get(WIKI_API, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


async def subpage_titles(client, title, max_subpages):
    """All namespace-0 pages whose title starts with `title` (the article + subpages)."""
    try:
        d = await _api(client, {"action": "query", "list": "allpages",
                                "apprefix": title, "apnamespace": 0, "aplimit": max_subpages})
        pages = [p["title"] for p in d.get("query", {}).get("allpages", [])]
    except Exception:
        pages = []
    if title not in pages:
        pages = [title] + pages
    return pages[:max_subpages]


async def fetch_wikitext(client, title):
    try:
        d = await _api(client, {"action": "parse", "page": title,
                                "prop": "wikitext", "redirects": 1})
        return d.get("parse", {}).get("wikitext", {}).get("*", "")
    except Exception:
        return ""


async def build_article(client, title, subs):
    parts = []
    for st in subs:
        wt = await fetch_wikitext(client, st)
        ct = clean_wikitext(wt)
        if len(ct) < 40:
            continue
        # label sub-page sections (strip the shared article prefix)
        label = st[len(title):].strip(" -/").strip() or "overview"
        parts.append(f"[{label}] {ct}" if st != title else ct)
    return "\n\n".join(parts).strip()


async def summarize(client, args, title, text):
    try:
        r = await client.post(
            f"{args.api_base.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {args.api_key}"},
            # Summarization needs no reasoning channel; disabling thinking makes the
            # model emit the summary directly (else it burns the budget in <think>).
            json={"model": args.model_name,
                  "messages": [{"role": "system", "content": SUMMARY_SYS},
                               {"role": "user", "content": f"Article: {title}\n\n{text[:16000]}"}],
                  "max_tokens": 900, "temperature": 0.2,
                  "chat_template_kwargs": {"enable_thinking": False}},
            timeout=180)
        r.raise_for_status()
        m = r.json()["choices"][0]["message"]
        out = (m.get("content") or m.get("reasoning_content") or "").strip()
        if "</think>" in out:
            out = out.split("</think>")[-1].strip()
        return out or text[:DB_TEXT_CAP]
    except Exception as e:
        print(f"  summarize failed for {title!r}: {e}", file=sys.stderr)
        return text[:DB_TEXT_CAP]


async def embed(client, args, text):
    r = await client.post(f"{args.embed_api_base.rstrip('/')}/embeddings",
                          headers={"Authorization": f"Bearer {args.api_key}"},
                          json={"model": args.embed_model, "input": [text[:2000]]}, timeout=60)
    r.raise_for_status()
    return r.json()["data"][0]["embedding"]


def _safe_name(title):
    return re.sub(r"[^A-Za-z0-9._-]", "_", title)[:180]


def _clip(text, cap):
    """Truncate to <= cap chars at the last sentence/word boundary (no mid-word cut)."""
    if len(text) <= cap:
        return text
    s = text[:cap]
    dot = s.rfind(". ")
    if dot > cap * 0.6:
        return s[:dot + 1]
    sp = s.rfind(" ")
    return (s[:sp] if sp > 0 else s).rstrip()


def load_titles(path):
    seen, out = set(), []
    with open(path) as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            t = (d.get("title") or "").strip()
            k = t.lower()
            if t and k not in seen:
                seen.add(k)
                out.append((t, d.get("entry_id", "")))
    return out


async def main_async(args):
    from pymilvus import MilvusClient
    mc = MilvusClient(uri=args.milvus_uri, token=args.milvus_token)
    os.makedirs(args.fulltext_dir, exist_ok=True)
    titles = load_titles(args.titles_file)
    if args.limit:
        titles = titles[:args.limit]
    print(f"enriching {len(titles)} unique wikidoc titles (dry_run={args.dry_run})")
    stats = {"fetched": 0, "empty": 0, "summarized": 0, "upserted": 0}
    sem = asyncio.Semaphore(args.concurrency)

    async with httpx.AsyncClient(headers={"User-Agent": "mirl-wikidoc-enrich/1.0"}) as client:
        def _find_sibling_entry_ids(subs):
            """entry_ids of the bare-title rows for this topic's sub-pages, so we can
            replace the ~N title rows with one enriched article row (else the short
            title embeddings outrank the long article for keyword queries)."""
            ids = []
            for st in subs:
                try:
                    rows = mc.query(collection_name=args.milvus_collection,
                                    filter=f'source_dataset == "wikidoc" and text_content == {json.dumps(st)}',
                                    output_fields=["entry_id"], limit=8)
                    ids += [r["entry_id"] for r in rows if r.get("entry_id")]
                except Exception:
                    pass
            return list(dict.fromkeys(ids))

        async def one(title, entry_id):
            async with sem:
                subs = await subpage_titles(client, title, args.max_subpages)
                article = await build_article(client, title, subs)
                if len(article) < 80:
                    stats["empty"] += 1
                    return
                stats["fetched"] += 1
                with open(os.path.join(args.fulltext_dir, _safe_name(title) + ".txt"), "w") as fo:
                    fo.write(article)
                if len(article) > DB_TEXT_CAP:
                    stored = _clip(await summarize(client, args, title, article), DB_TEXT_CAP)
                    stats["summarized"] += 1
                else:
                    stored = _clip(article, DB_TEXT_CAP)
                if args.dry_run:
                    print(f"\n=== {title}  (full={len(article)}c stored={len(stored)}c "
                          f"{'SUMMARY' if len(article) > DB_TEXT_CAP else 'FULL'}) ===")
                    print(stored[:600])
                    return
                emb = await embed(client, args, f"{title}\n{stored}")
                if not entry_id:
                    entry_id = ("wikidoc_enriched_" + _safe_name(title))[:200]
                row = {
                    "entry_id": entry_id,
                    "source_dataset": "wikidoc",
                    "modality": "text",
                    "content_type": "article",
                    "text_content": stored,
                    "question": title[:1000],
                    "answer": "",
                    "image_path": "",
                    "embedding": emb,
                }
                # Consolidate: delete the bare-title rows for this topic's sub-pages
                # (PK `id` is auto_id, so we can't upsert — we delete + insert), then
                # insert the single enriched article row.
                sib_ids = _find_sibling_entry_ids(subs)
                if entry_id not in sib_ids:
                    sib_ids.append(entry_id)
                if not args.keep_titles and sib_ids:
                    try:
                        idlist = ", ".join(json.dumps(i) for i in sib_ids)
                        mc.delete(collection_name=args.milvus_collection,
                                  filter=f"entry_id in [{idlist}]")
                    except Exception as e:
                        print(f"  delete siblings failed for {title!r}: {e}", file=sys.stderr)
                mc.insert(collection_name=args.milvus_collection, data=[row])
                stats["upserted"] += 1
                stats.setdefault("titles_removed", 0)
                stats["titles_removed"] += len(sib_ids)
                if stats["upserted"] % 10 == 0:
                    print(f"  upserted {stats['upserted']}  stats={stats}")

        await asyncio.gather(*[one(t, e) for t, e in titles])
    print(f"DONE stats={stats}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--titles_file", required=True)
    p.add_argument("--milvus_uri", default="http://mib.media.mit.edu:19531")
    p.add_argument("--milvus_token", default="root:Milvus")
    p.add_argument("--milvus_collection", default="medical_knowledge_v2")
    p.add_argument("--embed_api_base", default="http://mib.media.mit.edu:18001/v1")
    p.add_argument("--embed_model", default="Qwen/Qwen3-VL-Embedding-2B")
    p.add_argument("--api_base", default="http://point.dd.works:18184/v1")
    p.add_argument("--api_key", default="EMPTY")
    p.add_argument("--model_name", default="Qwen/Qwen3.6-27B")
    p.add_argument("--fulltext_dir", default="/scratch/sheng/self_evolving/wikidoc_fulltext")
    p.add_argument("--max_subpages", type=int, default=40)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--keep_titles", action="store_true",
                   help="Do NOT delete the topic's bare-title rows (default: consolidate).")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
