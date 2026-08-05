"""Per-source retrievability probe for the newly ingested KB sources.

Answers a specific question: when a query is *of the kind a source exists to
answer*, does that source actually surface in the top-k under the live ranking
policy? This separates two very different failures —

  (a) the source is unreachable (bad chunking/embedding/filtering), vs
  (b) the source is reachable but the task mix rarely asks for it.

Trace-mix stats alone cannot tell these apart: DailyMed being 1% of injected
passages means nothing if no task ever asked for a dose.

    python scripts/self_evolving/kb/probe_new_sources.py
"""

import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kb.retrieval import local_search  # noqa: E402

# (expected_source, query) — each query is written the way a clinician would ask
# the thing that source was ingested to cover.
PROBES = [
    ("dailymed", "exact adult oral dose of amoxicillin for strep pharyngitis"),
    ("dailymed", "metformin maximum daily dose and renal contraindication by eGFR"),
    ("dailymed", "lisinopril starting dose in hypertension and dose adjustment"),
    ("dailymed", "warfarin drug interactions listed in the label"),
    ("dailymed", "sertraline dosage and administration in adults"),
    ("dailymed", "apixaban dose reduction criteria in atrial fibrillation"),
    ("icd10cm", "ICD-10-CM code for type 2 diabetes mellitus with diabetic polyneuropathy"),
    ("icd10cm", "ICD-10 code for acute exacerbation of chronic obstructive pulmonary disease"),
    ("icd10cm", "ICD-10-CM code for iron deficiency anemia secondary to blood loss"),
    ("icd10cm", "what is the ICD-10 code for essential hypertension"),
    ("medlineplus", "explain to a patient in plain language what an A1C test measures"),
    ("medlineplus", "patient handout about shingles vaccination and who should get it"),
    ("medlineplus", "simple explanation of high cholesterol for a patient"),
    ("medlineplus", "patient information about what causes kidney stones"),
    ("statpearls", "management of acute pancreatitis including fluid resuscitation"),
    ("statpearls", "differential diagnosis and workup of syncope in the elderly"),
    ("statpearls", "treatment of diabetic ketoacidosis insulin and electrolytes"),
    ("statpearls", "evaluation and management of community acquired pneumonia"),
]


def main():
    top_k = int(os.environ.get("TOP_K", "5"))
    hit_at_k = Counter()
    total = Counter()
    print(f"probing {len(PROBES)} queries at top-{top_k}\n")
    for expected, q in PROBES:
        passages, _ = local_search(q, top_k)
        srcs = [p["source"] for p in passages]
        total[expected] += 1
        rank = srcs.index(expected) + 1 if expected in srcs else 0
        if rank:
            hit_at_k[expected] += 1
        mark = f"rank {rank}" if rank else "MISS"
        print(f"[{mark:>6}] want={expected:12s} {q[:58]}")
        print(f"          got: {srcs}")
    print("\n=== hit rate (expected source present in top-k) ===")
    for s in sorted(total):
        print(f"  {s:12s} {hit_at_k[s]}/{total[s]}")


if __name__ == "__main__":
    main()
