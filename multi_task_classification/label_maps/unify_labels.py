#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unify dataset labels into a single global class space and rewrite label_mapping VALUES
to the unified class indices, while keeping KEYS exactly the same.

Updates in this version:
- Longest-prefix dataset parsing (handles mosei_emotion, mosei_senti, ptsd_in_the_wild).
- Mental health is strictly binary (no collapse from severities; severities = unmapped).
- Global class space (21 classes) with stable indices; label_mapping values are renumbered from 0.
- Nested `global_classes` by domain in `meta`.
- End-of-run report lists anything unmapped/uncaught.
"""

import json
from copy import deepcopy
from pathlib import Path
from datetime import datetime

# ==========
# 1) HARD-CODED PATHS (edit these)
# ==========
INPUT_JSON  = "/Users/keane/Desktop/research/human-behavior/data/unified_scheme/label_maps/unified_scheme_binarymmpsy_no_vptd_chalearn_lmvd_esconv_full_label_map.json"
OUTPUT_JSON = "/Users/keane/Desktop/research/human-behavior/data/unified_scheme/label_maps/final_unified_scheme_binarymmpsy_no_vptd_chalearn_lmvd_esconv_full_label_map.json"

# ==========
# Canonicals & Synonyms
# ==========

# 7-point sentiment canonical
SENTIMENT_CANONICAL = [
    "highly negative",
    "negative",
    "weakly negative",
    "neutral",
    "weakly positive",
    "positive",
    "highly positive",
]

# Emotion canonical (keep "calm" distinct from "neutral")
EMOTION_CANONICAL = [
    "anger",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "calm",
    "sad",
    "surprise",
]

# Mental health canonical (strictly binaries; no severity handling)
MENTAL_HEALTH_CANONICAL = [
    "no_ptsd", "ptsd",
    "no_depression", "depression",
    "no_anxiety", "anxiety",
]

# Emotion synonyms (case-insensitive)
EMOTION_SYNONYMS = {
    "angry": "anger",
    "happiness": "happy",
    "joy": "happy",
    "sadness": "sad",
    "fearful": "fear",
    "surprised": "surprise",
    "pleasant surprise": "surprise",
    # "calm" and "neutral" are intentionally distinct
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

# Dataset → unified domain (single mental_health bucket)
DATASET_DOMAIN = {
    # sentiment intensity
    "chsimsv2": "sentiment_intensity",
    "mosei_senti": "sentiment_intensity",

    # emotion recognition
    "cremad": "emotion",
    "einterface": "emotion",
    "expw": "emotion",
    "meld": "emotion",
    "mosei_emotion": "emotion",
    "ravdess": "emotion",
    "tess": "emotion",

    # mental health (unified)
    "ptsd_in_the_wild": "mental_health",
    "mmpsy": "mental_health",
}

# Build a **global** unified classes list (domain/label tuples) with stable order
GLOBAL_CLASSES = []
# Sentiment
GLOBAL_CLASSES += [("sentiment_intensity", lab) for lab in SENTIMENT_CANONICAL]
# Emotion
GLOBAL_CLASSES += [("emotion", lab) for lab in EMOTION_CANONICAL]
# Mental health
GLOBAL_CLASSES += [("mental_health", lab) for lab in MENTAL_HEALTH_CANONICAL]

GLOBAL_CLASS_TO_INDEX = {dl: i for i, dl in enumerate(GLOBAL_CLASSES)}

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
        if key == ds:  # degenerate case
            return ds, ""
    return None, key

def normalize_emotion_label(raw: str) -> str:
    low = raw.strip().lower()
    return EMOTION_SYNONYMS.get(low, low)

def normalize_sentiment_label(raw: str) -> str:
    low = raw.strip().lower()
    return SENTIMENT_SYNONYMS.get(low, low)

def mh_direct_binary(label_low: str):
    """
    Map direct mental health binaries to canonical tokens.
    Returns (canonical_label or None, note)
    """
    l = " ".join(label_low.split())  # normalize spacing
    if l in {"ptsd", "no ptsd"}:
        return ("ptsd" if l == "ptsd" else "no_ptsd"), "PTSD direct binary"
    if l in {"depression", "no depression"}:
        return ("depression" if l == "depression" else "no_depression"), "Depression direct binary"
    if l in {"anxiety", "no anxiety"}:
        return ("anxiety" if l == "anxiety" else "no_anxiety"), "Anxiety direct binary"
    return None, ""

# ==========
# Core
# ==========

def build_unified_and_renumber(data: dict):
    """
    - Build unified_mapping and mental-health binary buckets.
    - Rewrite label_mapping values to unified indices (GLOBAL_CLASSES).
    - Return updated data + report.
    """
    lm = data["label_mapping"]
    original_label_ids = deepcopy(lm)  # for meta traceability

    report = {
        "unknown_dataset": [],
        "empty_tail_label": [],
        "noncanonical_sentiment": [],
        "noncanonical_emotion": [],
        "mental_health_unrecognized": [],
        "unmappable_to_global_class": [],
    }

    unified_mapping = {}   # key -> {domain, unified_label, notes, global_index}
    mh_bins = {
        "ptsd_binary": {},        # key -> no_ptsd/ptsd
        "depression_binary": {},  # key -> no_depression/depression
        "anxiety_binary": {},     # key -> no_anxiety/anxiety
    }

    new_label_mapping = {}
    known_datasets = set(DATASET_DOMAIN.keys())

    for key in lm.keys():
        dataset, tail = longest_prefix_parse(key, known_datasets)
        tail = tail.strip()

        if not tail:
            report["empty_tail_label"].append(key)

        if dataset is None:
            domain = "unknown"
            uni_label = tail.lower()
            notes = "Dataset not recognized by longest-prefix match; pass-through."
        else:
            domain = DATASET_DOMAIN.get(dataset, "unknown")

            if domain == "sentiment_intensity":
                uni_label = normalize_sentiment_label(tail)
                notes = ""
                if uni_label not in SENTIMENT_CANONICAL:
                    notes = f"Normalized '{tail}' -> '{uni_label}', NOT in canonical 7-point set."
                    report["noncanonical_sentiment"].append(key)

            elif domain == "emotion":
                uni_label = normalize_emotion_label(tail)
                notes = ""
                if uni_label not in EMOTION_CANONICAL:
                    notes = f"Normalized '{tail}' -> '{uni_label}', NOT in canonical emotion set."
                    report["noncanonical_emotion"].append(key)

            elif domain == "mental_health":
                low = tail.lower()
                direct, why = mh_direct_binary(low)
                if direct is not None:
                    uni_label = direct
                    notes = why
                    if direct in {"ptsd", "no_ptsd"}:
                        mh_bins["ptsd_binary"][key] = direct
                    elif direct in {"depression", "no_depression"}:
                        mh_bins["depression_binary"][key] = direct
                    elif direct in {"anxiety", "no_anxiety"}:
                        mh_bins["anxiety_binary"][key] = direct
                else:
                    # Any non-binary MH labels (e.g., severities) are now *unrecognized* by design.
                    uni_label = low
                    notes = "Unrecognized mental health label (not binary); pass-through & flagged."
                    report["mental_health_unrecognized"].append(key)

            else:
                uni_label = tail.lower()
                notes = "Unknown domain; pass-through."

        # Place into GLOBAL_CLASSES
        global_idx = GLOBAL_CLASS_TO_INDEX.get((domain, uni_label))
        if global_idx is None:
            report["unmappable_to_global_class"].append(f"{key} -> ({domain}, {uni_label})")

        unified_mapping[key] = {
            "domain": domain,
            "unified_label": uni_label,
            "global_index": global_idx,
            "notes": notes,
        }

        if global_idx is None:
            raise ValueError(
                f"Key '{key}' normalized to ({domain}, '{uni_label}') "
                f"is NOT in GLOBAL_CLASSES. Extend canonical lists or add synonyms."
            )

        new_label_mapping[key] = global_idx

    # Build nested global_classes by domain
    global_classes_nested = {}
    for i, (d, l) in enumerate(GLOBAL_CLASSES):
        global_classes_nested.setdefault(d, []).append({"index": i, "label": l})

    meta = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "domains": {
            "sentiment_intensity": SENTIMENT_CANONICAL,
            "emotion": EMOTION_CANONICAL,
            "mental_health": MENTAL_HEALTH_CANONICAL,
        },
        "synonyms": {
            "emotion": EMOTION_SYNONYMS,
            "sentiment_intensity": SENTIMENT_SYNONYMS,
        },
        "dataset_domain": DATASET_DOMAIN,
        "global_classes": global_classes_nested,  # nested by domain
        "original_label_ids": original_label_ids,
        "unified_mapping": unified_mapping,
        "mental_health_notes": [
            "Mental health labels are STRICT binaries only.",
            "Non-binary MH labels (e.g., severities) are flagged as unrecognized in the report."
        ],
        "mental_health_binaries": mh_bins,
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

    original = deepcopy(data)

    out, report = build_unified_and_renumber(data)

    # Write output
    if OUTPUT_JSON:
        Path(OUTPUT_JSON).parent.mkdir(parents=True, exist_ok=True)
        Path(OUTPUT_JSON).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote updated JSON with unified mapping to: {OUTPUT_JSON}")
    else:
        backup = in_path.with_suffix(in_path.suffix + ".bak")
        backup.write_text(json.dumps(original, indent=2, ensure_ascii=False), encoding="utf-8")
        in_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Backed up original to: {backup}")
        print(f"Overwrote input with unified mapping: {in_path}")

    # Report
    print("\n=== Unification Report ===")
    for k in ["unknown_dataset", "empty_tail_label", "noncanonical_sentiment",
              "noncanonical_emotion", "mental_health_unrecognized", "unmappable_to_global_class"]:
        vals = report.get(k, [])
        print(f"- {k} ({len(vals)}):")
        for item in vals:
            print(f"    • {item}")

if __name__ == "__main__":
    main()