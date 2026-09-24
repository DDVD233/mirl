"""Word statistics of the stage-1 knowledge base, per source (paper Table 1, 2026-09-24).

Scans the eight source families the stage-1 generation archive retrieved from in the Milvus
collection `medical_knowledge_v2` (the later StatPearls, DailyMed, MedlinePlus, and ICD-10
additions are excluded, as in stage1_kb_ablation.py) and counts entries, whitespace-separated
words, and characters of `text_content`, the text a retrieval returns. Sources are scanned in
parallel processes, one iterator each. Each source writes its own JSON, so an interrupted scan
resumes by skipping finished sources; `--merge` combines them.

    python stage1_kb_word_stats.py --out paper_data/stage1/kb_word_stats            # scan
    python stage1_kb_word_stats.py --out paper_data/stage1/kb_word_stats --merge    # summary
    python stage1_kb_word_stats.py --out paper_data/stage1/kb_word_stats --render paper/tables/kb_sources.tex

`--render` writes paper Table 1 from summary.json and the evidence shares of the 27B run's
generation archive (followups_2026-09-23/archive/archive_summary.json, produced by
stage1_archive_analysis.py).
"""

import argparse
import json
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

STAGE1_SOURCES = [
    "medrag_pubmed",
    "medrag_wiki",
    "medrag_textbook",
    "wikidoc",
    "pmc_vqa",
    "mirage",
    "pubmedqa",
    "climb",
]
# `id % k` slices force a full scan per page and overflow the 100 MB query output cap, so every
# source is read by one iterator with small pages.
SHARDS = {}


def scan(job):
    source, shard, shards, args = job
    out = Path(args["out"]) / f"{source}__{shard}of{shards}.json"
    if out.exists():
        return json.loads(out.read_text())
    from pymilvus import MilvusClient

    client = MilvusClient(uri=args["uri"], token=args["token"])
    expr = f'source_dataset == "{source}"'
    if shards > 1:
        expr += f" and id % {shards} == {shard}"
    it = client.query_iterator(
        args["collection"], filter=expr, output_fields=["text_content", "content_type", "modality"],
        batch_size=args["batch"],
    )
    stats = {"source": source, "shard": shard, "shards": shards, "entries": 0, "words": 0, "chars": 0,
             "content_type": Counter(), "modality": Counter(), "word_hist": Counter()}
    start = time.time()
    while True:
        batch = it.next()
        if not batch:
            break
        for row in batch:
            text = row["text_content"] or ""
            n = len(text.split())
            stats["entries"] += 1
            stats["words"] += n
            stats["chars"] += len(text)
            stats["content_type"][row["content_type"]] += 1
            stats["modality"][row["modality"]] += 1
            stats["word_hist"][min(n // 25 * 25, 1000)] += 1
    it.close()
    stats["seconds"] = round(time.time() - start, 1)
    stats = {k: dict(v) if isinstance(v, Counter) else v for k, v in stats.items()}
    out.write_text(json.dumps(stats, indent=1))
    return stats


def merge(out_dir):
    total = {}
    for path in sorted(Path(out_dir).glob("*of*.json")):
        s = json.loads(path.read_text())
        t = total.setdefault(s["source"], {"entries": 0, "words": 0, "chars": 0, "content_type": Counter(),
                                           "modality": Counter(), "word_hist": Counter(), "slices": 0})
        for k in ("entries", "words", "chars"):
            t[k] += s[k]
        for k in ("content_type", "modality", "word_hist"):
            t[k].update({kk: v for kk, v in s[k].items()})
        t["slices"] += 1
    for source, t in total.items():
        t["mean_words"] = t["words"] / t["entries"] if t["entries"] else 0
        hist = sorted((int(k), v) for k, v in t["word_hist"].items())
        acc, t["median_words_bin"] = 0, None
        for k, v in hist:
            acc += v
            if acc * 2 >= t["entries"]:
                t["median_words_bin"] = k
                break
        for k in ("content_type", "modality", "word_hist"):
            t[k] = dict(t[k])
    all_entries = sum(t["entries"] for t in total.values())
    all_words = sum(t["words"] for t in total.values())
    for t in total.values():
        t["share_entries"] = t["entries"] / all_entries
        t["share_words"] = t["words"] / all_words
    summary = {"sources": total, "entries": all_entries, "words": all_words}
    (Path(out_dir) / "summary.json").write_text(json.dumps(summary, indent=1))
    for source, t in total.items():
        print(f"{source:16s} {t['entries']:>12,d} {t['words']:>15,d} {t['mean_words']:7.1f} slices={t['slices']}")
    print(f"{'total':16s} {all_entries:>12,d} {all_words:>15,d}")


ROWS = [  # source, name, citation, one-sentence content description
    ("medrag_pubmed", "PubMed abstracts", "",
     "Abstracts of biomedical research, including clinical trials, case reports, and reviews."),
    ("medrag_wiki", "Wikipedia", "",
     "Encyclopedic articles on diseases, drugs, and procedures, along with general topics."),
    ("medrag_textbook", "Medical textbooks", "",
     "Passages from textbooks of the medical curriculum, such as anatomy, pathology, and pharmacology."),
    ("wikidoc", "WikiDoc", "",
     "A clinician-written medical encyclopedia, indexed mostly by section titles such as causes or differential diagnosis."),
    ("pmc_vqa", "PMC-VQA", "zhang2023pmc",
     "Questions and answers about figures in PubMed Central articles, indexed as text."),
    ("mirage", "MIRAGE", "",
     "Medical exam and research questions with their answers."),
    ("pubmedqa", "PubMedQA", "jin2019pubmedqa",
     "Yes-or-no research questions with the abstracts that answer them."),
    ("climb", "CLIMB", "dai2025climb",
     "Multimodal clinical questions with answers, such as imaging diagnoses, indexed as text."),
]
ARCHIVE = Path(__file__).resolve().parents[3] / "paper_data/stage1/followups_2026-09-23/archive/archive_summary.json"


def pct(x):
    if not x:
        return "0\\%"
    return "$<$0.1\\%" if x < 0.001 else f"{100 * x:.1f}\\%"


def num(x, fmt=",d"):
    return format(x, fmt).replace(",", "{,}")


def words_fmt(x):
    for scale, unit in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if x >= scale:
            return f"{x / scale:.{2 if unit == 'B' else 1}f}{unit}"
    return str(int(x))


def render(out_dir, path):
    summary = json.loads((Path(out_dir) / "summary.json").read_text())
    evidence = json.loads(ARCHIVE.read_text())["evidence"]
    citing, top = evidence["tasks_citing_source"], evidence["top_ranked_passage_source"]
    n_tasks = sum(evidence["distinct_sources_per_task"].values())
    lines = [
        "% GENERATED by scripts/self_evolving/analysis/stage1_kb_word_stats.py --render. Do not edit by hand.",
        "% Word counts: paper_data/stage1/kb_word_stats/summary.json (Milvus medical_knowledge_v2, the eight",
        "% stage-1 source families). Evidence shares: followups_2026-09-23/archive/archive_summary.json.",
        r"\begin{table*}[t]",
        r"\centering\footnotesize",
        r"\newcommand{\hd}[2]{\begin{tabular}[b]{@{}c@{}}#1\\#2\end{tabular}}",
        r"\caption{Sources of the medical knowledge base. Entries and words count the indexed text that a",
        "retrieval returns, and words per entry is their ratio. Tasks citing is the share of accepted",
        "generated tasks whose evidence includes the source, and top passage is the share whose",
        f"highest-ranked passage comes from it, over the {num(n_tasks)} analyzed tasks of the 27B run.}}",
        r"\label{tab:kb-sources}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{@{}l p{0.40\textwidth} rrrrr@{}}",
        r"\toprule",
        r"Source & Content & Entries & Words & \hd{Words per}{entry} & \hd{Tasks}{citing} & \hd{Top}{passage} \\",
        r"\midrule",
    ]
    for source, name, cite, desc in ROWS:
        t = summary["sources"][source]
        text = f"{desc[:-1]} \\citep{{{cite}}}." if cite else desc
        text = "\\raggedright " + text
        lines.append(f"{name} & {text} & {num(t['entries'])} & {words_fmt(t['words'])} & "
                     f"{t['mean_words']:.0f} & {pct(citing.get(source))} & {pct(top.get(source))} \\\\")
    words = summary["words"]
    lines += [
        r"\midrule",
        f"Total & & {num(summary['entries'])} & {words_fmt(words)} & {words / summary['entries']:.0f} & & \\\\",
        r"\bottomrule",
        r"\end{tabular}}",
        r"\end{table*}",
        "",
    ]
    text = "\n".join(lines)
    Path(path).write_text(text)
    print(text)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--uri", default="http://localhost:19531")
    p.add_argument("--token", default="root:Milvus")
    p.add_argument("--collection", default="medical_knowledge_v2")
    p.add_argument("--batch", type=int, default=2000)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--sources", nargs="*", default=STAGE1_SOURCES)
    p.add_argument("--merge", action="store_true")
    p.add_argument("--render", help="write the LaTeX table to this path from summary.json")
    a = p.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    if a.merge:
        return merge(a.out)
    if a.render:
        return render(a.out, a.render)
    args = {"out": a.out, "uri": a.uri, "token": a.token, "collection": a.collection, "batch": a.batch}
    jobs = [(s, k, SHARDS.get(s, 1), args) for s in a.sources for k in range(SHARDS.get(s, 1))]
    jobs.sort(key=lambda j: -SHARDS.get(j[0], 1))
    with ProcessPoolExecutor(a.workers) as pool:
        for s in pool.map(scan, jobs):
            print(f"{s['source']} {s['shard']}/{s['shards']}: {s['entries']:,} entries, {s['words']:,} words, "
                  f"{s.get('seconds', 0)} s", flush=True)
    merge(a.out)


if __name__ == "__main__":
    main()
