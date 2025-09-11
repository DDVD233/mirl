#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unify dataset labels into a single global class space and rewrite label_mapping VALUES
to the unified class indices, while keeping KEYS exactly the same.

This version:
- Splits mental health into THREE domains: mental_health_ptsd, mental_health_depression, mental_health_anxiety.
- Splits MELD into TWO datasets: meld_senti (sentiment_intensity) and meld_emotion (emotion).
- Adds TWO new binary domains:
    • sarcasm  ← dataset "mmsd" (labels: true/false)
    • humour   ← dataset "urfunny" (labels: true/false)
- Longest-prefix dataset parsing (mosei_emotion, mosei_senti, ptsd_in_the_wild, mmpsy_*, meld_senti, meld_emotion).
- End-of-run report lists anything unmapped/uncaught.

NOTE: Global class space expands from 21 → 25 with the new binary domains.
"""

import json
from copy import deepcopy
from pathlib import Path
from datetime import datetime

# ==========
# 1) PATHS
# ==========
INPUT_JSON  = "/Users/keane/Desktop/research/human-behavior/data/new/final_v2/final/label_map_v6.json"
OUTPUT_JSON = "/Users/keane/Desktop/research/human-behavior/data/new/final_v2/final/unified_label_map_v6.json"

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

# Mental health canonical labels (domain-specific pairs)
MH_PTSD_CANONICAL = ["no ptsd", "ptsd"]                  # → global indices 15,16
MH_DEPR_CANONICAL = ["no depression", "depression"]      # → global indices 17,18
MH_ANX_CANONICAL  = ["no anxiety", "anxiety"]            # → global indices 19,20

# New binary domains
SARCASM_CANONICAL = ["not sarcasm", "sarcasm"]           # → global indices 21,22
HUMOUR_CANONICAL  = ["not humour", "humour"]             # → global indices 23,24

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

# Boolean-ish synonyms for sarcasm/humour binaries
BOOL_SYNONYMS_TRUE  = {"true", "yes", "1", "y", "t"}
BOOL_SYNONYMS_FALSE = {"false", "no", "0", "n", "f"}

# ==========
# Dataset → domain (task-specific datasets)
# ==========
DATASET_DOMAIN = {
    # sentiment intensity
    "chsimsv2": "sentiment_intensity",
    "mosei_senti": "sentiment_intensity",
    "meld_senti": "sentiment_intensity",     # <— NEW: MELD sentiment split

    # emotion recognition
    "cremad": "emotion",
    "einterface": "emotion",
    "expw": "emotion",
    "meld_emotion": "emotion",               # <— NEW: MELD emotion split
    "mosei_emotion": "emotion",
    "ravdess": "emotion",
    "tess": "emotion",

    # mental health (split)
    "ptsd_in_the_wild": "mental_health_ptsd",
    "mmpsy_depression": "mental_health_depression",
    "mmpsy_anxiety": "mental_health_anxiety",
    "daicwoz": "mental_health_depression",   # ← NEW: map DAICWOZ to depression


    # new binary domains
    "mmsd": "sarcasm",                       # <— NEW
    "urfunny": "humour",                     # <— NEW
}

# ==========
# Build GLOBAL unified classes (domain,label) tuples with STABLE order/indices
# ==========
GLOBAL_CLASSES = []
# Sentiment → indices 0..6
GLOBAL_CLASSES += [("sentiment_intensity", lab) for lab in SENTIMENT_CANONICAL]
# Emotion → indices 7..14
GLOBAL_CLASSES += [("emotion", lab) for lab in EMOTION_CANONICAL]
# Mental health (split) → indices 15..20
GLOBAL_CLASSES += [("mental_health_ptsd", lab) for lab in MH_PTSD_CANONICAL]       # 15,16
GLOBAL_CLASSES += [("mental_health_depression", lab) for lab in MH_DEPR_CANONICAL] # 17,18
GLOBAL_CLASSES += [("mental_health_anxiety", lab) for lab in MH_ANX_CANONICAL]     # 19,20
# New binary domains → indices 21..24
GLOBAL_CLASSES += [("sarcasm", lab) for lab in SARCASM_CANONICAL]                  # 21,22
GLOBAL_CLASSES += [("humour", lab) for lab in HUMOUR_CANONICAL]                    # 23,24

GLOBAL_CLASS_TO_INDEX = {dl: i for i, dl in enumerate(GLOBAL_CLASSES)}
NUM_CLASSES = len(GLOBAL_CLASSES)  # == 25

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

def normalize_booleanish(raw: str):
    """
    Map common boolean-ish tokens to True/False or return None if unknown.
    """
    low = raw.strip().lower()
    if low in BOOL_SYNONYMS_TRUE:
        return True
    if low in BOOL_SYNONYMS_FALSE:
        return False
    if low in {"true", "false"}:
        return low == "true"
    return None

# ==========
# Core
# ==========

def build_unified_and_renumber(data: dict):
    """
    - Build unified_mapping with split MH domains + new sarcasm/humour domains.
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
        "noncanonical_binary": [],           # <— NEW for sarcasm/humour weird labels
        "unmappable_to_global_class": [],
    }

    unified_mapping = {}   # key -> {domain, unified_label, notes, global_index}
    # Optional convenience binning (kept for parity with your older meta)
    mh_bins = {
        "ptsd_binary": {},        # key -> no ptsd/ptsd
        "depression_binary": {},  # key -> no depression/depression
        "anxiety_binary": {},     # key -> no anxiety/anxiety
    }
    new_bin_notes = {
        "sarcasm_binary": {},     # key -> not sarcasm/sarcasm
        "humour_binary": {},      # key -> not humour/humour
    }

    new_label_mapping = {}
    known_datasets = set(DATASET_DOMAIN.keys())

    for key in lm.keys():
        dataset, tail = longest_prefix_parse(key, known_datasets)
        tail = tail.strip()

        if dataset is None:
            # Unknown dataset → pass-through; likely unmappable
            domain = "unknown"
            uni_label = tail.lower()
            notes = "Dataset not recognized by longest-prefix match; pass-through."
            report["unknown_dataset"].append(key)

        else:
            domain = DATASET_DOMAIN[dataset]

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

            elif domain == "mental_health_ptsd":
                low = " ".join(tail.lower().split())
                if low not in {"ptsd", "no ptsd"}:
                    notes = "Unrecognized PTSD label"
                    uni_label = low
                else:
                    uni_label = "ptsd" if low == "ptsd" else "no ptsd"
                    notes = "PTSD direct binary"
                    mh_bins["ptsd_binary"][key] = uni_label

            elif domain == "mental_health_depression":
                low = " ".join(tail.lower().split())
                b = normalize_booleanish(low)  # accepts true/false/yes/no/1/0/y/n/t/f
                if b is True:
                    uni_label = "depression"
                    notes = "Depression direct binary (boolean true)"
                    mh_bins["depression_binary"][key] = uni_label
                elif b is False:
                    uni_label = "no depression"
                    notes = "Depression direct binary (boolean false)"
                    mh_bins["depression_binary"][key] = uni_label
                elif low not in {"depression", "no depression"}:
                    notes = "Unrecognized depression label"
                    uni_label = low
                else:
                    uni_label = "depression" if low == "depression" else "no depression"
                    notes = "Depression direct binary"
                    mh_bins["depression_binary"][key] = uni_label

            elif domain == "mental_health_anxiety":
                low = " ".join(tail.lower().split())
                if low not in {"anxiety", "no anxiety"}:
                    notes = "Unrecognized anxiety label"
                    uni_label = low
                else:
                    uni_label = "anxiety" if low == "anxiety" else "no anxiety"
                    notes = "Anxiety direct binary"
                    mh_bins["anxiety_binary"][key] = uni_label

            elif domain == "sarcasm":
                b = normalize_booleanish(tail)
                if b is None:
                    uni_label = tail.lower()
                    notes = "Unrecognized boolean label for sarcasm (expected true/false)."
                    report["noncanonical_binary"].append(key)
                else:
                    uni_label = "sarcasm" if b else "not sarcasm"
                    notes = "Sarcasm direct binary (true/false)"
                    new_bin_notes["sarcasm_binary"][key] = uni_label

            elif domain == "humour":
                b = normalize_booleanish(tail)
                if b is None:
                    uni_label = tail.lower()
                    notes = "Unrecognized boolean label for humour (expected true/false)."
                    report["noncanonical_binary"].append(key)
                else:
                    uni_label = "humour" if b else "not humour"
                    notes = "Humour direct binary (true/false)"
                    new_bin_notes["humour_binary"][key] = uni_label

            else:
                # Should not happen
                uni_label = tail.lower()
                notes = "Unknown domain; pass-through."

        # Map (domain, unified_label) → global index
        global_idx = GLOBAL_CLASS_TO_INDEX.get((domain, uni_label))
        if global_idx is None:
            report["unmappable_to_global_class"].append(f"{key} -> ({domain}, {uni_label})")
            raise ValueError(
                f"Key '{key}' normalized to ({domain}, '{uni_label}') "
                f"is NOT in GLOBAL_CLASSES. Extend canonicals/synonyms or fix labels."
            )

        unified_mapping[key] = {
            "domain": domain,
            "unified_label": uni_label,
            "global_index": global_idx,
            "notes": notes,
        }
        new_label_mapping[key] = global_idx

    # Build nested global_classes by domain (reflecting split MH + new binaries)
    global_classes_nested = {}
    for i, (d, l) in enumerate(GLOBAL_CLASSES):
        global_classes_nested.setdefault(d, []).append({"index": i, "label": l})

    meta = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "domains": {
            "sentiment_intensity": SENTIMENT_CANONICAL,
            "emotion": EMOTION_CANONICAL,
            "mental_health_ptsd": MH_PTSD_CANONICAL,
            "mental_health_depression": MH_DEPR_CANONICAL,
            "mental_health_anxiety": MH_ANX_CANONICAL,
            "sarcasm": SARCASM_CANONICAL,
            "humour": HUMOUR_CANONICAL,
        },
        "synonyms": {
            "emotion": EMOTION_SYNONYMS,
            "sentiment_intensity": SENTIMENT_SYNONYMS,
        },
        "dataset_domain": DATASET_DOMAIN,       # routing is dataset-based only
        "global_classes": global_classes_nested, # nested by domain
        "original_label_ids": original_label_ids,
        "unified_mapping": unified_mapping,
        "mental_health_notes": [
            "Mental health split into PTSD/Depression/Anxiety sub-domains with binary labels only.",
            "Routing is purely dataset-based (mmpsy_* and ptsd_in_the_wild)."
        ],
        "binary_domain_notes": new_bin_notes,
    }

    out = deepcopy(data)
    out["label_mapping"] = new_label_mapping
    out["num_classes"] = NUM_CLASSES  # now 25
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