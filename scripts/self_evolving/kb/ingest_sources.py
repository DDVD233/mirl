"""Ingest new high-yield medical sources into the `medical_knowledge_v2` Milvus KB.

Why
---
The KB the retrieval rollouts search is ~53% general Wikipedia + 42% PubMed
abstract fragments, and 99% of its wikidoc rows are bare titles. HealthBench-
Professional rewards *current guideline text, exact drug dosing, consumer-health
guidance and codes* — content the KB simply does not hold, which is the main
reason every retrieval config scored below the no-tool baseline
(RETRIEVAL_INVESTIGATION_2026-07-24.md). These four sources target those gaps:

    statpearls   ~9.6k peer-reviewed clinical review articles (NCBI OA, 01/2026)
    dailymed     FDA drug labels -> dosing / contraindications / warnings
    medlineplus  ~2k NIH consumer-health topic summaries (patient register)
    icd10cm      the 2026 ICD-10-CM code table, grouped by category

Row schema matches build_medical_knowledge_v2.py / add_wikidoc_titles.py exactly
so the new rows rank against the same queries in the same vector space:

    entry_id  source_dataset  modality="text"  content_type  text_content(<=2000)
    question=""  answer=""  image_path=""  embedding(2048, COSINE)

Chunking
--------
Unlike the 512-char mid-sentence MedRAG chunks (a known failure mode: facts split
across chunks), chunks here are sentence-aware, ~1500 chars, and every chunk is
prefixed with its article/section heading so an isolated passage still says what
it is about — the retriever sees the heading text too, which is what makes a
"dose of X" query land on the dosing section rather than a random paragraph.

Run (on MIB, where Milvus + the embed server are local)
------------------------------------------------------
    /home/dvd/miniconda3/envs/new2/bin/python \
      scripts/self_evolving/kb/ingest_sources.py --source statpearls
    ... --source medlineplus | icd10cm | dailymed

--dry_run parses + chunks + prints stats and samples without touching Milvus.
Resume is automatic: entry_ids in the per-source checkpoint file are skipped.
"""

import argparse
import asyncio
import glob
import hashlib
import html
import io
import json
import logging
import os
import re
import sys
import time
import zipfile
from collections import Counter

logger = logging.getLogger("kb_ingest")

COLLECTION = "medical_knowledge_v2"
EMBED_DIM = 2048
MAX_TEXT = 2000          # text_content field limit
TARGET_CHUNK = 1500      # aim; a chunk never exceeds MAX_TEXT
MIN_CHUNK = 120          # drop slivers (the wikidoc "bare title" failure mode)
OVERLAP_SENTS = 1        # sentence overlap between consecutive chunks

SRC_DIR_DEFAULT = "/scratch/dvd/medkb_sources"


# --------------------------------------------------------------------------
# chunking
# --------------------------------------------------------------------------
_SENT_RE = re.compile(r"(?<=[.!?;:])\s+(?=[A-Z0-9(\[])|\n+")


def _clean(text: str) -> str:
    text = html.unescape(text or "")
    text = re.sub(r"[ \t\xa0]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def fit(text: str, limit: int = MAX_TEXT) -> str:
    """Truncate to `limit` UTF-8 BYTES, not characters.

    Milvus enforces varchar limits in bytes. Headings here use em dashes (3 bytes
    each) and clinical text carries µ/±/°/Greek letters, so a 2000-character slice
    can be 2014 bytes and the whole insert batch is rejected."""
    raw = text.encode("utf-8")
    if len(raw) <= limit:
        return text
    return raw[:limit].decode("utf-8", errors="ignore")


def chunk_text(body: str, heading: str) -> list[str]:
    """Sentence-aware chunks of ~TARGET_CHUNK chars, each prefixed with `heading`.

    The heading is repeated on every chunk so a passage returned in isolation is
    self-describing (and so the embedding carries the topic, not just the prose)."""
    body = _clean(body)
    heading = _clean(heading)
    if not body:
        return []
    prefix = f"{heading}\n" if heading else ""
    room = MAX_TEXT - len(prefix)
    if room < MIN_CHUNK:                       # pathological heading; truncate it
        prefix = prefix[: MAX_TEXT // 2] + "\n"
        room = MAX_TEXT - len(prefix)
    target = min(TARGET_CHUNK, room)

    sents = [s.strip() for s in _SENT_RE.split(body) if s and s.strip()]
    chunks, cur = [], []
    cur_len = 0
    for s in sents:
        s = s[:room]                           # a single monster sentence still fits
        if cur and cur_len + 1 + len(s) > target:
            chunks.append(" ".join(cur))
            cur = cur[-OVERLAP_SENTS:] if OVERLAP_SENTS else []
            cur_len = sum(len(x) + 1 for x in cur)
        cur.append(s)
        cur_len += len(s) + 1
    if cur:
        chunks.append(" ".join(cur))
    out = []
    for c in chunks:
        c = c.strip()
        if len(c) >= MIN_CHUNK or (len(chunks) == 1 and len(c) >= 40):
            out.append(fit(prefix + c))
    return out


# --------------------------------------------------------------------------
# XML helpers (JATS/BITS and HL7 SPL both carry namespaces we don't care about)
# --------------------------------------------------------------------------
def _strip_ns(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _itertext(el) -> str:
    parts = []
    for node in el.iter():
        t = _strip_ns(node.tag)
        if t in ("table-wrap", "graphic", "inline-graphic", "media", "xref"):
            continue
        if node.text:
            parts.append(node.text)
        if node.tail:
            parts.append(node.tail)
    return " ".join(parts)


# --------------------------------------------------------------------------
# source: statpearls  (NCBI Bookshelf OA, JATS/BITS .nxml)
# --------------------------------------------------------------------------
# Sections that are navigation/boilerplate rather than clinical content.
_SP_SKIP_SECS = {
    "continuing education activity", "objectives", "review questions",
    "disclosure", "references", "media", "figures", "tables",
    "publication details", "copyright", "questions",
}


def parse_statpearls(src_dir: str):
    import xml.etree.ElementTree as ET

    files = sorted(glob.glob(os.path.join(src_dir, "statpearls", "**", "*.nxml"), recursive=True))
    logger.info(f"statpearls: {len(files):,} nxml files")
    for path in files:
        stem = os.path.splitext(os.path.basename(path))[0]
        try:
            root = ET.parse(path).getroot()
        except Exception as e:
            logger.debug(f"parse fail {stem}: {e}")
            continue
        # Article title lives in book-part-meta/title-group/title.
        title = ""
        for tg in root.iter():
            if _strip_ns(tg.tag) == "title-group":
                for ch in tg:
                    if _strip_ns(ch.tag) in ("title", "article-title"):
                        title = _clean(_itertext(ch))
                        break
                if title:
                    break
        if not title or title.lower() == "statpearls":
            continue
        body = None
        for el in root.iter():
            if _strip_ns(el.tag) == "body":
                body = el
                break
        if body is None:
            continue
        for i, sec in enumerate(body):
            if _strip_ns(sec.tag) != "sec":
                continue
            sec_title = ""
            for ch in sec:
                if _strip_ns(ch.tag) == "title":
                    sec_title = _clean(_itertext(ch))
                    break
            if sec_title.lower().strip() in _SP_SKIP_SECS:
                continue
            text = _clean(" ".join(
                _itertext(p) for p in sec.iter()
                if _strip_ns(p.tag) == "p"
            ))
            if len(text) < MIN_CHUNK:
                continue
            heading = f"StatPearls: {title}" + (f" — {sec_title}" if sec_title else "")
            for j, ch in enumerate(chunk_text(text, heading)):
                yield f"statpearls_{stem}_{i}_{j}", "article", ch


# --------------------------------------------------------------------------
# source: medlineplus  (NIH consumer-health topic XML)
# --------------------------------------------------------------------------
def parse_medlineplus(src_dir: str):
    import xml.etree.ElementTree as ET

    path = os.path.join(src_dir, "medlineplus_topics.xml")
    root = ET.parse(path).getroot()
    n = 0
    for topic in root:
        if _strip_ns(topic.tag) != "health-topic":
            continue
        if (topic.get("language") or "English") != "English":
            continue
        title = _clean(topic.get("title") or "")
        tid = topic.get("id") or str(n)
        if not title:
            continue
        aliases, summary, groups = [], "", []
        for ch in topic:
            t = _strip_ns(ch.tag)
            if t == "also-called":
                aliases.append(_clean(_itertext(ch)))
            elif t == "full-summary":
                # full-summary is HTML escaped inside the XML text.
                summary = re.sub(r"<[^>]+>", " ", html.unescape(_itertext(ch)))
            elif t == "group":
                groups.append(_clean(_itertext(ch)))
        summary = _clean(summary)
        if len(summary) < MIN_CHUNK:
            continue
        head = f"MedlinePlus (patient information): {title}"
        if aliases:
            head += f" (also called: {', '.join(aliases[:6])})"
        if groups:
            head += f" [{', '.join(groups[:3])}]"
        for j, ch in enumerate(chunk_text(summary, head)):
            yield f"medlineplus_{tid}_{j}", "topic", ch
        n += 1
    logger.info(f"medlineplus: {n:,} English topics")


# --------------------------------------------------------------------------
# source: icd10cm  (2026 code table, grouped by 3-char category)
# --------------------------------------------------------------------------
def parse_icd10cm(src_dir: str):
    """One row per ICD-10-CM category (e.g. E11), listing every child code with
    its full description. Grouping matters: 74k isolated code strings would
    reproduce the wikidoc bare-title failure (nothing to match a clinical query
    against), whereas a category block reads like a lookup table and carries the
    disease vocabulary a question actually uses."""
    path = os.path.join(src_dir, "icd10", "icd10cm_order_2026.txt")
    cats: dict[str, dict] = {}
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if len(line) < 78:
                continue
            code = line[6:13].strip()
            billable = line[14:15].strip()
            long_desc = line[77:].strip()
            if not code or not long_desc:
                continue
            cat = code[:3]
            d = cats.setdefault(cat, {"header": "", "codes": []})
            if billable == "0" and len(code) == 3:
                d["header"] = long_desc
            else:
                # Format as the dotted code clinicians and coders actually write.
                dotted = code if len(code) <= 3 else f"{code[:3]}.{code[3:]}"
                d["codes"].append(f"{dotted} — {long_desc}")
    logger.info(f"icd10cm: {len(cats):,} categories")
    for cat, d in sorted(cats.items()):
        header = d["header"] or cat
        heading = f"ICD-10-CM 2026 category {cat}: {header}"
        body = "\n".join(d["codes"]) if d["codes"] else header
        # Codes are line-oriented, not prose: pack lines rather than sentences.
        lines, cur, cur_len, blocks = body.split("\n"), [], 0, []
        room = MAX_TEXT - len(heading) - 2
        for ln in lines:
            if cur and cur_len + len(ln) + 1 > min(TARGET_CHUNK, room):
                blocks.append("\n".join(cur))
                cur, cur_len = [], 0
            cur.append(ln)
            cur_len += len(ln) + 1
        if cur:
            blocks.append("\n".join(cur))
        for j, b in enumerate(blocks):
            yield f"icd10cm_{cat}_{j}", "codeblock", fit(f"{heading}\n{b}")


# --------------------------------------------------------------------------
# source: dailymed  (FDA SPL drug labels -> the clinically actionable sections)
# --------------------------------------------------------------------------
# LOINC section codes worth keeping. Everything else on a label (how supplied,
# packaging, inactive ingredients, ...) is retrieval noise for our tasks.
_SPL_SECTIONS = {
    "34068-7": "Dosage and Administration",
    "34067-9": "Indications and Usage",
    "34070-3": "Contraindications",
    "43685-7": "Warnings and Precautions",
    "34066-1": "Warnings",
    "34084-4": "Adverse Reactions",
    "34073-7": "Drug Interactions",
    "43684-0": "Use in Specific Populations",
    "34090-1": "Clinical Pharmacology",
    "34088-5": "Overdosage",
}


def _spl_drug_name(root) -> str:
    """Best-effort brand + generic name for the labeled product."""
    brand, generic = "", ""
    for el in root.iter():
        if _strip_ns(el.tag) != "manufacturedProduct":
            continue
        for ch in el.iter():
            t = _strip_ns(ch.tag)
            if t == "name" and not brand:
                brand = _clean(ch.text or "")
            if t == "activeMoiety" and not generic:
                for nm in ch.iter():
                    if _strip_ns(nm.tag) == "name" and (nm.text or "").strip():
                        generic = _clean(nm.text)
                        break
        if brand:
            break
    if brand and generic and generic.lower() not in brand.lower():
        return f"{brand} ({generic})"
    return brand or generic


def _parse_spl_bytes(data: bytes, entry_stem: str):
    import xml.etree.ElementTree as ET

    try:
        root = ET.parse(io.BytesIO(data)).getroot()
    except Exception:
        return
    drug = _spl_drug_name(root)
    if not drug:
        return
    for sec in root.iter():
        if _strip_ns(sec.tag) != "section":
            continue
        code_el = None
        for ch in sec:
            if _strip_ns(ch.tag) == "code":
                code_el = ch
                break
        if code_el is None:
            continue
        loinc = (code_el.get("code") or "").strip()
        label = _SPL_SECTIONS.get(loinc)
        if not label:
            continue
        text = _clean(" ".join(
            _itertext(t) for t in sec
            if _strip_ns(t.tag) in ("text", "paragraph", "list", "table")
        ))
        if len(text) < MIN_CHUNK:
            continue
        heading = f"FDA drug label — {drug} — {label}"
        for j, ch in enumerate(chunk_text(text, heading)):
            yield f"{entry_stem}_{loinc}_{j}", "druglabel", ch, f"{drug}|{label}"


def parse_dailymed(src_dir: str, max_per_key: int = 1):
    """Walk the DailyMed release zips (zip-of-zips) and emit label-section chunks.

    Dedup: labels are massively redundant (dozens of generic labelers ship the
    same text). Keep at most `max_per_key` distinct labels per
    (drug name, section) so one ingredient does not flood retrieval."""
    zips = sorted(glob.glob(os.path.join(src_dir, "dm_rx_part*.zip")))
    logger.info(f"dailymed: {len(zips)} release archives")
    seen_key: Counter = Counter()
    seen_hash: set[str] = set()
    for zpath in zips:
        try:
            outer = zipfile.ZipFile(zpath)
        except Exception as e:
            logger.warning(f"bad zip {zpath}: {e}")
            continue
        names = [n for n in outer.namelist() if n.lower().endswith(".zip")]
        logger.info(f"  {os.path.basename(zpath)}: {len(names):,} inner SPL zips")
        for i, inner_name in enumerate(names):
            try:
                inner = zipfile.ZipFile(io.BytesIO(outer.read(inner_name)))
                xmls = [n for n in inner.namelist() if n.lower().endswith(".xml")]
                if not xmls:
                    continue
                data = inner.read(xmls[0])
            except Exception:
                continue
            stem = "dailymed_" + os.path.splitext(os.path.basename(inner_name))[0][:40]
            label_keys = set()
            for eid, ctype, chunk, key in _parse_spl_bytes(data, stem):
                label_keys.add(key)
                if seen_key[key] >= max_per_key:
                    continue
                h = hashlib.md5(chunk.encode()).hexdigest()
                if h in seen_hash:
                    continue
                seen_hash.add(h)
                yield eid, ctype, chunk
            # Count this label once per (drug, section) it covered, so the next
            # labeler shipping the same text is skipped wholesale.
            for k in label_keys:
                seen_key[k] += 1
            if (i + 1) % 5000 == 0:
                logger.info(f"    {i + 1:,}/{len(names):,} labels; {len(seen_hash):,} chunks kept")
        outer.close()


PARSERS = {
    "statpearls": parse_statpearls,
    "medlineplus": parse_medlineplus,
    "icd10cm": parse_icd10cm,
    "dailymed": parse_dailymed,
}


# --------------------------------------------------------------------------
# embed + insert
# --------------------------------------------------------------------------
async def embed_batch(http, args, texts):
    resp = await http.post(
        f"{args.embed_api_base}/embeddings",
        json={"model": args.embed_model, "input": [fit(t) for t in texts]},
        headers={"Authorization": f"Bearer {args.embed_api_key}"},
    )
    resp.raise_for_status()
    data = sorted(resp.json()["data"], key=lambda d: d.get("index", 0))
    return [d["embedding"] for d in data]


def load_done(checkpoint_file) -> set:
    done = set()
    if checkpoint_file and os.path.isfile(checkpoint_file):
        with open(checkpoint_file) as f:
            for line in f:
                if line.strip():
                    done.add(line.strip())
    return done


async def run(args):
    import httpx

    parser = PARSERS[args.source]
    kwargs = {}
    if args.source == "dailymed":
        kwargs["max_per_key"] = args.max_per_key

    if args.dry_run:
        n, chars = 0, 0
        samples = []
        for eid, ctype, text in parser(args.src_dir, **kwargs):
            n += 1
            chars += len(text)
            if len(samples) < 5:
                samples.append((eid, ctype, text))
            if args.limit and n >= args.limit:
                break
        logger.info(f"dry_run {args.source}: {n:,} chunks, mean {chars / max(n, 1):,.0f} chars")
        for eid, ctype, text in samples:
            logger.info(f"--- {eid} [{ctype}] ---\n{text[:700]}\n")
        return

    from pymilvus import MilvusClient
    client = MilvusClient(uri=args.milvus_uri, token=args.milvus_token)
    if not client.has_collection(COLLECTION):
        logger.error(f"{COLLECTION} missing")
        sys.exit(1)

    done = load_done(args.checkpoint_file)
    logger.info(f"resume: {len(done):,} entry_ids already ingested — skipping")

    counts = Counter()
    t0 = time.time()
    ckpt = open(args.checkpoint_file, "a") if args.checkpoint_file else None
    sem = asyncio.Semaphore(args.embed_concurrency)
    limits = httpx.Limits(max_connections=max(args.embed_concurrency * 4, 64))

    async with httpx.AsyncClient(limits=limits, timeout=httpx.Timeout(180.0)) as http:

        async def flush(wave):
            """Embed `wave` (list of batches) then insert every resulting row."""
            async def do_one(batch):
                async with sem:
                    try:
                        vecs = await embed_batch(http, args, [t for _, _, t in batch])
                        if len(vecs) != len(batch):
                            raise ValueError("embed count mismatch")
                        return [{
                            "entry_id": eid[:200],
                            "source_dataset": args.source,
                            "modality": "text",
                            "content_type": ctype[:20],
                            "text_content": fit(text),
                            "question": "",
                            "answer": "",
                            "image_path": "",
                            "embedding": v,
                        } for (eid, ctype, text), v in zip(batch, vecs)]
                    except Exception as e:
                        counts["embed_err"] += len(batch)
                        logger.debug(f"embed err: {type(e).__name__}: {e}")
                        return []

            rows = [r for br in await asyncio.gather(*(do_one(b) for b in wave)) for r in br]
            for k in range(0, len(rows), args.insert_batch):
                chunk = rows[k:k + args.insert_batch]
                try:
                    client.insert(collection_name=COLLECTION, data=chunk)
                    counts["inserted"] += len(chunk)
                    if ckpt:
                        for r in chunk:
                            ckpt.write(r["entry_id"] + "\n")
                        ckpt.flush()
                except Exception as e:
                    counts["insert_err"] += len(chunk)
                    logger.warning(f"insert err: {type(e).__name__}: {e}")

        batch, wave = [], []
        for eid, ctype, text in parser(args.src_dir, **kwargs):
            counts["parsed"] += 1
            if eid in done:
                counts["skipped"] += 1
                continue
            batch.append((eid, ctype, text))
            if len(batch) >= args.embed_batch:
                wave.append(batch)
                batch = []
            if len(wave) >= args.embed_concurrency:
                await flush(wave)
                wave = []
                if counts["inserted"] % (args.insert_batch * 20) < args.insert_batch:
                    rate = counts["inserted"] / max(time.time() - t0, 1)
                    logger.info(f"  inserted {counts['inserted']:,} ({rate:,.0f}/s) "
                                f"parsed {counts['parsed']:,}")
            if args.limit and counts["parsed"] >= args.limit:
                break
        if batch:
            wave.append(batch)
        if wave:
            await flush(wave)

    if ckpt:
        ckpt.close()
    logger.info(f"== {args.source} summary ==")
    for k, v in sorted(counts.items()):
        logger.info(f"  {k}: {v:,}")
    logger.info(f"done in {time.time() - t0:,.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=sorted(PARSERS))
    ap.add_argument("--src_dir", default=SRC_DIR_DEFAULT)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="stop after N parsed chunks (debug)")
    ap.add_argument("--max_per_key", type=int, default=1,
                    help="dailymed: max labels kept per (drug, section)")
    ap.add_argument("--milvus_uri", default=os.environ.get("MILVUS_URI", "http://localhost:19531"))
    ap.add_argument("--milvus_token", default=os.environ.get("MILVUS_TOKEN", "root:Milvus"))
    ap.add_argument("--embed_api_base", default=os.environ.get("EMBED_API_BASE", "http://localhost:18001/v1"))
    ap.add_argument("--embed_api_key", default=os.environ.get("EMBED_API_KEY", "EMPTY"))
    ap.add_argument("--embed_model", default=os.environ.get("EMBED_MODEL", "Qwen/Qwen3-VL-Embedding-2B"))
    ap.add_argument("--embed_batch", type=int, default=32)
    ap.add_argument("--embed_concurrency", type=int, default=16)
    ap.add_argument("--insert_batch", type=int, default=500)
    ap.add_argument("--checkpoint_file", default="")
    args = ap.parse_args()
    if not args.checkpoint_file:
        args.checkpoint_file = f"/scratch/dvd/medkb_sources/.done_{args.source}.txt"

    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
