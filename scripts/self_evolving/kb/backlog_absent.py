"""Final ingestion backlog: assign each ABSENT criterion to an acquirable corpus,
price it in recoverable points, and apply the hand-audited false-ABSENT discount.

Discount: 36 ABSENT records were re-tested by hand against live Milvus (plain +
--raw, criterion-targeted queries). 16/36 were refuted (the fact IS in the KB).
The rate splits by whether the probe's suggested_source names a corpus we already
hold: 7/10 (70%) when it does, 9/26 (35%) when it does not.
"""
import collections
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "probe_out", "cat_ABSENT.json")))

INGESTED = re.compile(r"StatPearls|MedlinePlus|DailyMed|FDA (Drug )?[Ll]abel|FDA label|ICD-10-CM|Wikidoc")
FA_INGESTED, FA_EXTERNAL = 0.70, 0.35          # lenient (measured)
FA_INGESTED_S, FA_EXTERNAL_S = 0.40, 0.27      # strict (borderline refutations kept as ABSENT)

# Ordered: first match wins. Keyed to a real, nameable acquisition target.
CORPORA = [
    ("HARM_OR_RUBRIC_ERROR", r"harmful procedural|inflict vascular trauma|No appropriate clinical reference|No standard reference is likely|No authoritative clinical guideline"),
    ("SCORES_CALCULATORS", r"\bscore\b|\bScore\b|scoring (sheet|table|manual)|calculator|Calculator|point-allocation|severity assessment|Tokyo Guidelines|PEACH|APACHE|Caprini|ELTS|VOCAL-Penn|Kaiser Permanente|Oxford Knee|Surgical Apgar|Classification of Prenatal and Postnatal Urinary Tract|PREVENT (Online )?Calculator|risk equation|normative"),
    ("GENEREVIEWS_ORPHANET", r"GeneReviews|Orphanet|IC3D|SATB2|FAM20C|TGFB1-related|Camurati"),
    ("REGULATORY_ACTIONS_FDA", r"FDA (consumer|safety|Device|Digital Health|drug approval database|approval announcement|import alert|recall)|safety communication|import alert|recall notice|consumer warning|NCI Drug Dictionary|FDA-recognized listing|FDA prescribing information and FDA approval"),
    ("DRUG_LABELS_NONUS", r"ANVISA|CIMS|MIMS|bula|SmPC|Summary of Product Characteristics|EMA |European Medicines|national (drug|medicines) (registry|database)|manufacturer product information|NHS Medicines"),
    ("DRUG_LABELS_US_SPL", r"DailyMed|FDA (Drug )?[Ll]abel|FDA label|Prescribing Information|package insert|prescribing information"),
    ("DRUG_INTERACTION_COMPENDIA", r"Lexicomp|Micromedex|LactMed|Drugs and Lactation|drug-interaction (reference|compendium)|Drug Interactions monograph"),
    ("TRIALS_PMC_CTGOV", r"\bNEJM\b|\bJAMA\b|\bLancet\b|randomi[sz]ed (controlled )?(clinical )?trial|trial (publication|record|protocol|report|results|re-analysis|follow-up)|ClinicalTrials\.gov|WATERFALL|IMPROVE trial|IDEAL|CHOIR|ACORN|STEMI-DTU|STOPDAPT|PAPILLON|GLOBRYTE|NP30179|KEYNOTE|PREEMPT|SVR trial|Marik|Lau JYW|Ferrari MD|Fern[aá]ndez|Cureus|PMCID|Doyle|Knaus|Gawande|Singh et al|Cooper et al|Barbar|de-Madaria|Futier|Pfirrmann|Mahmud|Nguyen et al|Ching et al|Puopolo|derivation/validation|original .*publication"),
    ("SYSREV_PMC_OA", r"systematic review|meta-analysis|Cochrane|pooled analysis|scoping review|comparative study|cohort study|case (report|series)|registry|PubMed[- ]indexed|peer-reviewed (review|publication)|[Rr]eview article|conference report|ASH/EHA"),
    ("WHO_GLOBAL_HEALTH", r"\bWHO\b|World Health Organization|GPEI|Child Growth Standards|position paper"),
    ("NATIONAL_MOH_NONUS", r"Minist[eé]rio|CONITEC|Cameroon|Expanded Programme|SUS |Brazilian Ministry|national immunization|state cricket|NADA/WADA|Estonian|HERBA|Greek (medical|Ministry)|Terminologia|bilingual"),
    ("CODING_CMS", r"ICD-10|CPT|HCPCS|CMS|billing|E/M|NCHS|WHO ICD-10 (online )?browser"),
    ("GL_CARDIO", r"AHA|ACC/|ACC |ACCP|ACEP|CHEST|SCAI|SHM|SIR|SVM|SVN|\bESC\b|EACTS|ASE/EACVI|ASRA|ESRA|Heart Association|Brain Trauma"),
    ("GL_ONCOLOGY", r"NCCN|ASTRO|ESMO|ASCO|ILROG|European LeukemiaNet|Survivorship"),
    ("GL_OBGYN_REPRO", r"ACOG|FIGO|RCOG|ASRM|NICE .*(pregnan|menopause|miscarriage|preterm)|Preterm labour|Induction of Labor|Planned Home Birth|US MEC|Medical Eligibility"),
    ("GL_UROLOGY", r"\bAUA\b|SUFU|\bEAU\b|Urological Association|Urology"),
    ("GL_NEPHROLOGY", r"KDIGO|KDOQI|Renal Best Practice|Onco-Nephrology|ASON"),
    ("GL_ID_PUBHEALTH", r"IDSA|\bCDC\b|Yellow Book|Sexually Transmitted Infections Treatment|Foodborne|NIH HIV"),
    ("GL_OPHTH", r"\bAAO\b|EyeWiki|Preferred Practice Pattern|ophthalm"),
    ("GL_DERM", r"DermNet|American Academy of Dermatology|British Association of Dermatologists|NICE guideline NG190|dermatopathology|WHO Classification of Skin Tumours"),
    ("GL_NEURO_HEADACHE", r"American Headache Society|\bAHS\b|American College of Physicians|EANO|European Headache Federation|ICHD|ECTRIMS|Association of British Neurologists|AFTD"),
    ("GL_PEDS", r"\bAAP\b|AACAP|International Pediatric Sepsis|Neofax|HealthyChildren|Nelson Textbook"),
    ("GL_GI_SURG", r"\bACG\b|ASCRS|ERAS|Tokyo Guidelines|colorectal surgery"),
    ("GL_CRITCARE", r"Surviving Sepsis|SCCM|PADIS|ESICM|Society of Critical Care"),
    ("GL_ENDO_METAB", r"ADA |American Diabetes Association|AACE|Endocrine Society|Standards of Care in Diabetes"),
    ("GL_OTHER_SOCIETY", r"guideline|Guideline|Practice Parameter|position statement|consensus|Scientific Statement|Committee Opinion|Practice Bulletin|Clinical Consensus|NICE|\bASH\b|AAOS|joint registry|AMA |ACR |SVS|ESVS|AAO-HNSF|Polyanalgesic|society"),
    ("SPECIALTY_TEXTS", r"textbook|Textbook|chapter|Facial Plastic|Baker|Clinical Methods|Forensic Toxicology|reading-list|operative|reconstructive surgery|atlas|PROMIS|Emotion Dysregulation"),
    ("PATIENT_EDUCATION", r"patient (education|information|handout|instruction)|MedlinePlus|NHS |NCCIH|Office of Dietary Supplements|consumer"),
    ("ALREADY_INGESTED_GAP", r"StatPearls|Wikidoc"),
]


def corpus(rec):
    s = " || ".join([rec.get("suggested_source") or "", rec.get("missing_fact") or "",
                     rec.get("criterion_text") or ""])
    for name, pat in CORPORA:
        if re.search(pat, s):
            return name
    return "UNASSIGNED"


def main():
    agg = collections.defaultdict(lambda: {"n": 0, "rec": 0.0, "neg": 0, "negpts": 0.0,
                                           "dn": 0.0, "drec": 0.0, "dn_s": 0.0, "drec_s": 0.0,
                                           "idx": []})
    for i, r in enumerate(D):
        pts, mr = r["points"], r["met_rate"]
        rec = pts * (1 - (mr if mr is not None else 0.0)) if pts > 0 else 0.0
        ing = bool(INGESTED.search(r.get("suggested_source") or ""))
        keep, keep_s = (1 - (FA_INGESTED if ing else FA_EXTERNAL),
                        1 - (FA_INGESTED_S if ing else FA_EXTERNAL_S))
        a = agg[corpus(r)]
        a["n"] += 1
        a["rec"] += rec
        a["dn"] += keep
        a["drec"] += rec * keep
        a["dn_s"] += keep_s
        a["drec_s"] += rec * keep_s
        a["idx"].append(i)
        if pts < 0:
            a["neg"] += 1
            a["negpts"] += pts

    print(f"{'corpus':28s} {'n':>4} {'n*':>6} {'n*s':>6} {'recPts':>8} {'pts*':>7} {'pts*s':>7} {'neg#':>5}")
    T = collections.Counter()
    for k, a in sorted(agg.items(), key=lambda kv: -kv[1]["drec"]):
        print(f"{k:28s} {a['n']:4d} {a['dn']:6.1f} {a['dn_s']:6.1f} {a['rec']:8.1f} "
              f"{a['drec']:7.1f} {a['drec_s']:7.1f} {a['neg']:5d}")
        for f in ("n", "rec", "dn", "drec", "dn_s", "drec_s", "neg"):
            T[f] += a[f]
    print(f"{'TOTAL':28s} {T['n']:4d} {T['dn']:6.1f} {T['dn_s']:6.1f} {T['rec']:8.1f} "
          f"{T['drec']:7.1f} {T['drec_s']:7.1f} {T['neg']:5d}")
    json.dump({k: v["idx"] for k, v in agg.items()},
              open("/tmp/claude-1001/-home-dvd-mirl/c735ff90-4676-45b4-914a-95e65e1316b9/scratchpad/corpus_idx.json", "w"))


if __name__ == "__main__":
    main()
