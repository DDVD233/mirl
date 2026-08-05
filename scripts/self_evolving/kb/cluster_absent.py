"""Cluster the ABSENT bucket into acquirable corpora and price each cluster."""
import json
import os
import re
import collections

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "probe_out", "cat_ABSENT.json")))

# ordered rules: first match wins
RULES = [
    ("NONACTIONABLE_HARM", r"harmful procedural|inflict vascular trauma|No appropriate clinical reference"),
    ("TRIALS", r"\bNEJM\b|\bJAMA\b|randomi[sz]ed (controlled )?trial|\btrial (publication|record|protocol|report)|ClinicalTrials\.gov|WATERFALL|IMPROVE trial|IDEAL|CHOIR|ACORN|STEMI-DTU|STOPDAPT|PAPILLON|GLOBRYTE|NP30179|JADE|KEYNOTE|PREEMPT|SVR trial|Marik|Lau JYW|Ferrari MD|Fern[aá]ndez|meta-analysis by|Cureus|PMCID|systematic review|meta-analysis|Cochrane"),
    ("SCORES", r"score|scoring|calculator|classification of prenatal|criteria for assessment of the severity|Tokyo Guidelines|PEACH|APACHE|Caprini|ELTS|VOCAL-Penn|Kaiser|Oxford Knee|Surgical Apgar|UTD|PREVENT (Online )?Calculator|risk equation"),
    ("GENETICS", r"GeneReviews|Orphanet|IC3D|SATB2|FAM20C|TGFB1|gene"),
    ("CODING", r"ICD-10|CPT|HCPCS|CMS|billing|E/M|NCHS"),
    ("DRUGREG", r"ANVISA|CIMS|MIMS|national (drug|medicines)|bula|SmPC|Summary of Product Characteristics|EMA |manufacturer (product|reimbursement|intrathecal)|import alert|recall notice|safety communication|consumer warning|Office of Dietary Supplements|NCCIH|Lexicomp|Micromedex|Drug Interaction"),
    ("FDALABEL", r"FDA (Drug )?[Ll]abel|Prescribing Information|DailyMed|package insert|FDA approval|FDA Device|FDA Digital Health|FDA drug approval database|NCI Drug Dictionary|LactMed"),
    ("WHO_INTL", r"\bWHO\b|World Health Organization|Minist[eé]rio|CONITEC|Cameroon|Expanded Programme|GPEI|Estonian|HERBA|Greek|Terminologia|NADA/WADA|WADA"),
    ("GUIDELINE", r"guideline|Guideline|Practice Parameter|Preferred Practice Pattern|position (paper|statement)|consensus|Scientific Statement|Standards of Care|Committee Opinion|Practice Bulletin|Clinical Consensus|Appropriateness Criteria|NICE|KDIGO|ACOG|AUA|EAU|ESC|AHA|ACC|IDSA|CDC|NCCN|ASTRO|ESMO|ASCO|AAP|AACAP|AAO|ASRM|ASRA|SCCM|ACG|ASCRS|ERAS|EANO|ILROG|Brain Trauma Foundation|ADA |AACE|Endocrine Society|ASH |ECTRIMS|EHF|American Headache Society|AAO-HNSF|Polyanalgesic|ASE/EACVI|FIGO|FMF|ICHD"),
    ("TEXTBOOK_SPECIALTY", r"textbook|Textbook|chapter|Facial Plastic|Baker|Clinical Methods|Nelson|Forensic Toxicology|EyeWiki|DermNet|reading-list|Neurosurgery"),
    ("REVIEW_LIT", r"[Rr]eview article|PubMed|peer-reviewed|cohort study|case report|registry|literature"),
    ("ALREADY_INGESTED", r"StatPearls|MedlinePlus|Wikidoc"),
]


def bucket(rec):
    s = (rec.get("suggested_source") or "") + " || " + (rec.get("missing_fact") or "")
    for name, pat in RULES:
        if re.search(pat, s):
            return name
    return "OTHER"


# does the suggested source name a corpus we already hold?
INGESTED = re.compile(r"StatPearls|MedlinePlus|DailyMed|FDA (Drug )?[Ll]abel|FDA label|ICD-10-CM|Wikidoc|wikidoc")


def main():
    rows = []
    for i, r in enumerate(D):
        mr = r["met_rate"]
        pts = r["points"]
        rec_pts = pts * (1 - (mr if mr is not None else 0.0)) if pts > 0 else 0.0
        rows.append({
            "i": i, "b": bucket(r), "pts": pts, "mr": mr, "rec": rec_pts,
            "ing": bool(INGESTED.search(r.get("suggested_source") or "")),
            "src": (r.get("suggested_source") or "")[:90],
            "crit": (r.get("criterion_text") or "")[:80],
        })

    agg = collections.defaultdict(lambda: {"n": 0, "rec": 0.0, "neg": 0, "negpts": 0.0, "ing": 0, "nullmr": 0})
    for x in rows:
        a = agg[x["b"]]
        a["n"] += 1
        a["rec"] += x["rec"]
        a["ing"] += int(x["ing"])
        a["nullmr"] += int(x["mr"] is None)
        if x["pts"] < 0:
            a["neg"] += 1
            a["negpts"] += x["pts"]
    print(f"{'bucket':22s} {'n':>4} {'recPts':>8} {'neg#':>5} {'negPts':>8} {'namesIngested':>13} {'mr=null':>8}")
    tot = collections.Counter()
    for k, a in sorted(agg.items(), key=lambda kv: -kv[1]["rec"]):
        print(f"{k:22s} {a['n']:4d} {a['rec']:8.1f} {a['neg']:5d} {a['negpts']:8.1f} {a['ing']:13d} {a['nullmr']:8d}")
        tot["n"] += a["n"]; tot["rec"] += a["rec"]; tot["neg"] += a["neg"]; tot["ing"] += a["ing"]
    print(f"{'TOTAL':22s} {tot['n']:4d} {tot['rec']:8.1f} {tot['neg']:5d} {'':8s} {tot['ing']:13d}")

    json.dump(rows, open("/tmp/claude-1001/-home-dvd-mirl/c735ff90-4676-45b4-914a-95e65e1316b9/scratchpad/absent_rows.json", "w"))
    # positive-only totals
    pos = [x for x in rows if x["pts"] > 0]
    print("\npositive-points criteria:", len(pos), "recoverable pts:", round(sum(x["rec"] for x in pos), 1))
    print("negative-points criteria:", len([x for x in rows if x["pts"] < 0]),
          "their points:", sum(x["pts"] for x in rows if x["pts"] < 0))
    print("names an already-ingested corpus:", sum(x["ing"] for x in rows))


if __name__ == "__main__":
    main()
