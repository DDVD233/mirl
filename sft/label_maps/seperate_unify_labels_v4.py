#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unify ONLY sentiment across datasets; all other datasets are separate domains.
Meta structure/keys are preserved (dataset_domain, domains, synonyms, global_classes,
original_label_ids, unified_mapping, mental_health_notes, binary_domain_notes, ...).
"""

import json
from copy import deepcopy
from pathlib import Path
from datetime import datetime

# ==========
# 1) PATHS
# ==========
INPUT_JSON  = "/Users/keane/Desktop/research/human-behavior/data/new/final_v2/final/label_map_v6.json"
OUTPUT_JSON = "/Users/keane/Desktop/research/human-behavior/data/new/final_v2/final/seperate_unified_label_map_v6.json"

# ==========
# Canonicals & Synonyms
# ==========

# 7-point sentiment canonical (shared across sentiment datasets)
SENTIMENT_CANONICAL = [
    "highly negative",
    "negative",
    "weakly negative",
    "neutral",
    "weakly positive",
    "positive",
    "highly positive",
]

# Keep emotion synonyms for backwards compatibility in meta.synonyms
EMOTION_SYNONYMS = {
    "angry": "anger",
    "happiness": "happy",
    "joy": "happy",
    "sadness": "sad",
    "fearful": "fear",
    "surprised": "surprise",
    "pleasant surprise": "surprise",
}

# Sentiment synonyms -> 7-point canonical
SENTIMENT_SYNONYMS = {
    "strongly negative": "highly negative",
    "negative": "negative",
    "weakly negative": "weakly negative",
    "neutral": "neutral",
    "weakly positive": "weakly positive",
    "positive": "positive",
    "strongly positive": "highly positive",
}

# Boolean-ish synonyms
BOOL_SYNONYMS_TRUE  = {"true", "yes", "1", "y", "t"}
BOOL_SYNONYMS_FALSE = {"false", "no", "0", "n", "f"}

def normalize_sentiment_label(raw: str) -> str:
    low = raw.strip().lower()
    return SENTIMENT_SYNONYMS.get(low, low)

def normalize_booleanish(raw: str):
    low = raw.strip().lower()
    if low in BOOL_SYNONYMS_TRUE:  return True
    if low in BOOL_SYNONYMS_FALSE: return False
    if low in {"true", "false"}:   return (low == "true")
    return None

# ==========
# Dataset → domain (routing)
# ==========
# Sentiment datasets share the unified sentiment domain
SENTIMENT_DATASETS = {"chsimsv2", "mosei_senti", "meld_senti"}

# All other datasets become their own domains (use dataset name or renamed MH domains)
DATASET_DOMAIN = {
    # sentiment intensity (shared)
    "chsimsv2": "sentiment_intensity",
    "mosei_senti": "sentiment_intensity",
    "meld_senti": "sentiment_intensity",

    # emotion recognition → separate domains per dataset
    "cremad": "cremad_emotion",
    "einterface": "einterface_emotion",
    "expw": "expw_emotion",
    "meld_emotion": "meld_emotion",
    "mosei_emotion": "mosei_emotion",
    "ravdess": "ravdess_emotion",
    "tess": "tess_emotion",

    # mental health → separate domains per dataset
    "daicwoz": "mental_health_daicwoz",           # was merged before; now its own domain
    "mmpsy_depression": "mental_health_mmpsy_depression",
    "mmpsy_anxiety": "mental_health_mmpsy_anxiety",
    "ptsd_in_the_wild": "mental_health_ptsd_in_the_wild",

    # binary others → separate domains per dataset
    "mmsd": "mmsd",
    "urfunny": "urfunny",
}

# Recognized dataset prefixes (keys in label_mapping start with "<dataset>_<label>")
KNOWN_DATASETS = set(DATASET_DOMAIN.keys())

# ==========
# Helpers
# ==========

def longest_prefix_parse(key: str, known_datasets):
    """
    Parse 'dataset_...' using the LONGEST matching dataset name as prefix.
    Returns (dataset, tail) or (None, original_key) if no match.
    """
    for ds in sorted(known_datasets, key=len, reverse=True):
        prefix = ds + "_"
        if key.startswith(prefix):
            return ds, key[len(prefix):]
        if key == ds:
            return ds, ""
    return None, key

# ==========
# Core
# ==========

def build_unified_and_renumber(data: dict):
    """
    - Unify sentiment only (shared 7-point).
    - For all other datasets, keep dataset-specific domains and labels (lowercased).
    - Preserve meta structure/keys downstream expects.
    """
    lm = data["label_mapping"]
    original_label_ids = deepcopy(lm)  # for meta traceability

    report = {
        "unknown_dataset": [],
        "empty_tail_label": [],
        "noncanonical_sentiment": [],
        "noncanonical_emotion": [],   # kept for compatibility (unused in routing)
        "noncanonical_binary": [],
        "unmappable_to_global_class": [],
    }

    unified_mapping = {}   # key -> {domain, unified_label, notes, global_index}
    mh_bins = {
        "ptsd_binary": {},
        "depression_binary": {},
        "anxiety_binary": {},
    }
    new_bin_notes = {
        "sarcasm_binary": {},
        "humour_binary": {},
    }

    new_label_mapping = {}
    per_domain_labels = {}  # domain -> set(labels)

    for key in lm.keys():
        dataset, tail = longest_prefix_parse(key, KNOWN_DATASETS)
        tail = tail.strip()

        if not tail:
            report["empty_tail_label"].append(key)

        if dataset is None:
            # Unknown dataset → pass-through; likely unmapped later
            domain = "unknown"
            uni_label = tail.lower()
            notes = "Dataset not recognized by longest-prefix match; pass-through."
            report["unknown_dataset"].append(key)

        else:
            domain = DATASET_DOMAIN[dataset]
            low = tail.lower()

            if domain == "sentiment_intensity":
                uni_label = normalize_sentiment_label(tail)
                notes = ""
                if uni_label not in SENTIMENT_CANONICAL:
                    notes = f"Normalized '{tail}' -> '{uni_label}', NOT in 7-point canonical."
                    report["noncanonical_sentiment"].append(key)

            elif domain == "mental_health_daicwoz":
                # Keep DAICWOZ binary inside its own domain
                b = normalize_booleanish(low)
                if b is True:
                    uni_label = "depression"
                    notes = "DAICWOZ boolean true -> depression"
                    mh_bins["depression_binary"][key] = uni_label
                elif b is False:
                    uni_label = "no depression"
                    notes = "DAICWOZ boolean false -> no depression"
                    mh_bins["depression_binary"][key] = uni_label
                else:
                    # If DAICWOZ ever provides text labels, pass them through
                    uni_label = low
                    notes = "DAICWOZ label pass-through"

            elif domain == "mental_health_mmpsy_depression":
                b = normalize_booleanish(low)
                if b is True:
                    uni_label = "depression"
                    notes = "MMPSY depression boolean true -> depression"
                    mh_bins["depression_binary"][key] = uni_label
                elif b is False:
                    uni_label = "no depression"
                    notes = "MMPSY depression boolean false -> no depression"
                    mh_bins["depression_binary"][key] = uni_label
                else:
                    uni_label = low
                    notes = "MMPSY depression label pass-through"

            elif domain == "mental_health_mmpsy_anxiety":
                if low in {"anxiety", "no anxiety"}:
                    uni_label = low
                    notes = "MMPSY anxiety binary"
                    mh_bins["anxiety_binary"][key] = uni_label
                else:
                    uni_label = low
                    notes = "MMPSY anxiety label pass-through"

            elif domain == "mental_health_ptsd_in_the_wild":
                if low in {"ptsd", "no ptsd"}:
                    uni_label = low
                    notes = "PTSD in the wild binary"
                    mh_bins["ptsd_binary"][key] = uni_label
                else:
                    uni_label = low
                    notes = "PTSD in the wild label pass-through"

            elif domain == "mmsd":
                # Sarcasm binary but kept inside mmsd domain
                b = normalize_booleanish(low)
                if b is True:
                    uni_label = "sarcasm"
                    notes = "MMSD boolean true -> sarcasm"
                    new_bin_notes["sarcasm_binary"][key] = uni_label
                elif b is False:
                    uni_label = "not sarcasm"
                    notes = "MMSD boolean false -> not sarcasm"
                    new_bin_notes["sarcasm_binary"][key] = uni_label
                else:
                    uni_label = low
                    notes = "MMSD label pass-through"
                    report["noncanonical_binary"].append(key)

            elif domain == "urfunny":
                # Humour binary but kept inside urfunny domain
                b = normalize_booleanish(low)
                if b is True:
                    uni_label = "humour"
                    notes = "URFunny boolean true -> humour"
                    new_bin_notes["humour_binary"][key] = uni_label
                elif b is False:
                    uni_label = "not humour"
                    notes = "URFunny boolean false -> not humour"
                    new_bin_notes["humour_binary"][key] = uni_label
                else:
                    uni_label = low
                    notes = "URFunny label pass-through"
                    report["noncanonical_binary"].append(key)

            else:
                # All other per-dataset domains (e.g., meld_emotion, ravdess, tess, ...)
                # Keep the dataset's own annotation scheme (lowercased, trimmed)
                uni_label = low
                notes = "Dataset-specific label pass-through"

        # collect labels per domain for later global class construction
        per_domain_labels.setdefault(domain, set()).add(uni_label)

        unified_mapping[key] = {
            "domain": domain,
            "unified_label": uni_label,
            "notes": notes,
        }

    # ==========
    # Build GLOBAL_CLASSES deterministically:
    #   1) sentiment_intensity in canonical order
    #   2) all other domains A→Z; labels within each domain A→Z
    # ==========
    GLOBAL_CLASSES = []
    if "sentiment_intensity" in per_domain_labels:
        for lab in SENTIMENT_CANONICAL:
            # include only those that appear (usually all 7)
            if lab in per_domain_labels["sentiment_intensity"]:
                GLOBAL_CLASSES.append(("sentiment_intensity", lab))

    for domain in sorted(d for d in per_domain_labels if d != "sentiment_intensity"):
        for lab in sorted(per_domain_labels[domain]):
            GLOBAL_CLASSES.append((domain, lab))

    GLOBAL_CLASS_TO_INDEX = {dl: i for i, dl in enumerate(GLOBAL_CLASSES)}

    # attach indices
    new_label_mapping = {}
    for key, entry in unified_mapping.items():
        tup = (entry["domain"], entry["unified_label"])
        idx = GLOBAL_CLASS_TO_INDEX.get(tup)
        if idx is None:
            report["unmappable_to_global_class"].append(f"{key} -> {tup}")
            raise ValueError(f"Key '{key}' normalized to {tup} not in GLOBAL_CLASSES.")
        entry["global_index"] = idx
        new_label_mapping[key] = idx

    # Build nested global_classes by domain (for meta)
    global_classes_nested = {}
    for i, (d, l) in enumerate(GLOBAL_CLASSES):
        global_classes_nested.setdefault(d, []).append({"index": i, "label": l})

    # === meta (preserve SAME KEYS as before) ===
    # For 'domains', we now publish the discovered label sets per domain,
    # while keeping sentiment canonical. This preserves the key and shape.
    meta_domains = {}
    for d, labels in per_domain_labels.items():
        if d == "sentiment_intensity":
            meta_domains[d] = SENTIMENT_CANONICAL  # fixed canonical
        else:
            meta_domains[d] = sorted(labels)

    meta = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "domains": meta_domains,                    # SAME KEY
        "synonyms": {                               # SAME KEY and inner keys
            "emotion": EMOTION_SYNONYMS,            # retained for compatibility
            "sentiment_intensity": SENTIMENT_SYNONYMS,
        },
        "dataset_domain": DATASET_DOMAIN,           # SAME KEY
        "global_classes": global_classes_nested,    # SAME KEY
        "original_label_ids": original_label_ids,   # SAME KEY
        "unified_mapping": unified_mapping,         # SAME KEY
        "mental_health_notes": [                    # SAME KEY
            "Mental health domains are dataset-specific (no cross-dataset merge).",
            "DAICWOZ is its own domain with boolean mapping to depression/no depression."
        ],
        "binary_domain_notes": new_bin_notes,       # SAME KEY
    }

    out = deepcopy(data)
    out["label_mapping"] = new_label_mapping
    out["num_classes"] = len(GLOBAL_CLASSES)
    out["meta"] = meta

    return out, report

# ==========
# I/O & main
# ==========

def main():
    in_path = Path(INPUT_JSON)
    if not in_path.exists():
        raise FileNotFoundError(f"INPUT_JSON not found: {in_path}")

    with in_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "label_mapping" not in data or not isinstance(data["label_mapping"], dict):
        raise ValueError("JSON must contain a 'label_mapping' dict.")

    out, report = build_unified_and_renumber(data)

    Path(OUTPUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_JSON).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote updated JSON with unified mapping to: {OUTPUT_JSON}")

    print("\n=== Unification Report ===")
    for k in [
        "unknown_dataset",
        "empty_tail_label",
        "noncanonical_sentiment",
        "noncanonical_emotion",
        "noncanonical_binary",
        "unmappable_to_global_class",
    ]:
        vals = report.get(k, [])
        print(f"- {k} ({len(vals)}):")
        for item in vals:
            print(f"    • {item}")

if __name__ == "__main__":
    main()