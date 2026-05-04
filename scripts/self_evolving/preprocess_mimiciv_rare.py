"""
Preprocess MIMIC-IV admissions into a rare-disease primary-diagnosis QA dataset
with chest X-ray and ECG modalities linked.

Pipeline:
  1. Filter admissions whose primary diagnosis (first index in `outputs`) is an
     ICD-10 code that has an Orphanet "E" (Exact) cross-reference.
  2. Mask leakage from each admission: drop dischtime, deathtime,
     discharge_location, hospital_expire_flag, survived, 48_ihm. Lab/chart/
     prescription events are kept as-is.
  3. Render up to one 12-lead ECG per admission to a PNG (cached on disk).
  4. Attach up to N_XRAYS chest X-rays (latest ones) and 1 ECG plot per
     admission via the verl `images` field, with `<image>` placeholders in
     the user prompt.
  5. Format remaining content into a compact clinical-note prompt with
     min/max/last summaries for labs/charts to fit the token budget.
  6. Split temporally with patient-level disjointness (70:30).

Inputs (on mib):
  /scratch/high_modality/multimodal/mimiciv/admissions/<hadm_id>.json
  /scratch/high_modality/multimodal/mimiciv/icd_code_to_description.json
  /scratch/high_modality/multimodal/mimiciv/hosp/admissions.csv.gz
  /scratch/high_modality/multimodal/mimiciv/chest_xray/<image_path>
  /scratch/high_modality/multimodal/mimiciv/<ecg_file_path> (.mat)
  /scratch/self_evolving_datasets/orphanet/en_product1.xml
"""

import argparse
import csv
import gzip
import json
import os
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from glob import glob
from multiprocessing import Pool

ICD10_RE = re.compile(r"^[A-Z][0-9][A-Z0-9]")

SYSTEM_PROMPT = (
    "You are a senior physician reviewing a hospital admission for a patient "
    "who has a rare disease. Examine the demographics, vital signs, lab "
    "results, medications, procedures, and any chest X-ray or 12-lead ECG "
    "images provided. Identify the single most likely primary diagnosis.\n\n"
    "First reason inside <think>...</think> tags, then output the final answer "
    "in the form \\boxed{ICD-10 CODE: Diagnosis name} (e.g. "
    "\\boxed{E70.0: Classical phenylketonuria})."
)


def load_orphanet_e_codes(xml_path: str) -> set:
    tree = ET.parse(xml_path)
    e_codes = set()
    for d in tree.getroot().iter("Disorder"):
        erl = d.find("ExternalReferenceList")
        if erl is None:
            continue
        for r in erl.findall("ExternalReference"):
            src = r.find("Source")
            if src is None or src.text != "ICD-10":
                continue
            ref = (r.find("Reference").text or "").strip().rstrip(",").replace(".", "").upper()
            rel_node = r.find("DisorderMappingRelation/Name")
            rel = (rel_node.text if rel_node is not None and rel_node.text else "").split(" ", 1)[0]
            if rel == "E" and ref:
                e_codes.add(ref)
    return e_codes


def match_e(code: str, e_set: set) -> str | None:
    for n in range(len(code), 2, -1):
        if code[:n] in e_set:
            return code[:n]
    return None


def load_subject_admit_map(csv_gz_path: str) -> dict:
    out = {}
    with gzip.open(csv_gz_path, "rt") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out[int(row["hadm_id"])] = {
                "subject_id": int(row["subject_id"]),
                "admittime": row["admittime"],
            }
    return out


def load_microbiology_map(
    csv_gz_path: str, keep_hadms: set | None = None
) -> dict:
    """Group microbiologyevents.csv.gz rows by hadm_id, aggregating antibiotic
    susceptibilities back into one dict per (specimen, isolate).

    Returns dict[hadm_id, list[culture]]. Each culture has:
      date, spec (specimen type), test (test name), organism (or None),
      susc (list of (antibiotic, S/I/R)), comments.

    `keep_hadms` (optional) — restrict to a known set of admissions to keep
    memory low; rows for other admissions are dropped on read.
    """
    by_admit: dict[int, dict] = defaultdict(dict)
    with gzip.open(csv_gz_path, "rt", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            hadm_str = (row.get("hadm_id") or "").strip()
            if not hadm_str:
                continue  # outpatient / pre-admit
            try:
                hadm_id = int(hadm_str)
            except ValueError:
                continue
            if keep_hadms is not None and hadm_id not in keep_hadms:
                continue
            spec_id = row.get("micro_specimen_id") or ""
            isolate = (row.get("isolate_num") or "0").strip() or "0"
            test_name = (row.get("test_name") or "").strip()
            key = (spec_id, test_name, isolate)
            d = by_admit[hadm_id].get(key)
            if d is None:
                date = (row.get("charttime") or row.get("chartdate") or "")[:10]
                d = {
                    "date": date,
                    "spec": (row.get("spec_type_desc") or "").strip(),
                    "test": test_name,
                    "organism": (row.get("org_name") or "").strip() or None,
                    "susc": [],
                    "comments": (row.get("comments") or "").strip(),
                }
                by_admit[hadm_id][key] = d
            ab = (row.get("ab_name") or "").strip()
            interp = (row.get("interpretation") or "").strip()
            if ab and interp:
                d["susc"].append((ab, interp))
    return {h: list(v.values()) for h, v in by_admit.items()}


def format_microbiology(cultures: list) -> list:
    """Render cultures as compact lines. Skip rows that are entirely empty
    or only contain redacted-PHI placeholders ('___')."""
    if not cultures:
        return []
    cultures = sorted(cultures, key=lambda c: c.get("date", ""))
    body: list = []
    for c in cultures:
        date = c.get("date", "")
        spec = (c.get("spec") or "").strip()
        test = (c.get("test") or "").strip()
        org = c.get("organism")
        comments = (c.get("comments") or "").strip()
        susc = c.get("susc") or []

        if test and spec and test.lower() != spec.lower():
            label = f"{spec} / {test}"
        else:
            label = spec or test
        if not label:
            continue

        if org:
            extras = ""
            if susc:
                # group antibiotics by S/I/R (rounded), cap to 12 to keep tight
                by_interp: dict[str, list] = defaultdict(list)
                for ab, ip in susc[:12]:
                    by_interp[ip].append(ab)
                extras = " — susc: " + "; ".join(
                    f"{ip}: " + ", ".join(abs_) for ip, abs_ in by_interp.items()
                )
            body.append(f"  - {date} {label}: {org}{extras}")
        else:
            clean = " ".join(comments.split())
            # strip mostly-redacted noise
            if clean and clean.replace("_", "").strip():
                if len(clean) > 200:
                    clean = clean[:200] + "…"
                body.append(f"  - {date} {label}: {clean}")
            elif "CULTURE" in label.upper():
                body.append(f"  - {date} {label}: no growth")
    if not body:
        return []
    return ["# Microbiology"] + body + [""]


def render_ecg_png(mat_path: str, out_png: str) -> bool:
    """Render a 12-lead ECG .mat file to a PNG plot. Returns True on success."""
    if os.path.exists(out_png):
        return True
    try:
        from scipy.io import loadmat
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        m = loadmat(mat_path)
        val = m.get("val")
        if val is None or val.ndim != 2 or val.shape[0] < 12:
            return False
        fs_arr = m.get("fs")
        fs = int(fs_arr.flatten()[0]) if fs_arr is not None else 500
        leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        n_samples = val.shape[1]
        t = np.arange(n_samples) / fs

        fig, axes = plt.subplots(6, 2, figsize=(12, 8), sharex=True)
        for i, ax in enumerate(axes.flatten()):
            ax.plot(t, val[i], linewidth=0.6, color="black")
            ax.set_ylabel(leads[i], rotation=0, labelpad=20, fontsize=9)
            ax.grid(True, which="both", color="#ffcccc", linewidth=0.4)
            ax.set_xticks(np.arange(0, t[-1] + 0.1, 0.2), minor=True)
            ax.tick_params(axis="both", labelsize=7)
        axes[-1, 0].set_xlabel("Time (s)")
        axes[-1, 1].set_xlabel("Time (s)")
        plt.suptitle("12-lead ECG (10 s)", fontsize=10)
        plt.tight_layout()
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        fig.savefig(out_png, dpi=80, bbox_inches="tight")
        plt.close(fig)
        return True
    except Exception:
        return False


def summarize_series(name: str, events: list) -> str | None:
    """Compact summary: name: latest unit (range min-max, n=N)."""
    if not events:
        return None
    nums = []
    unit = ""
    latest = None
    latest_time = ""
    for e in events:
        v = e.get("value")
        if v is None:
            continue
        try:
            f = float(v)
            nums.append(f)
            if not unit and e.get("unit"):
                unit = e["unit"]
            t = e.get("time", "")
            if t > latest_time:
                latest_time = t
                latest = f
        except (TypeError, ValueError):
            if latest is None:
                latest = v
    if not nums and latest is None:
        return None
    if nums:
        vmin, vmax = min(nums), max(nums)
        n = len(nums)
        if vmin == vmax:
            return f"  - {name}: {vmin:g}{(' ' + unit) if unit else ''}"
        return (
            f"  - {name}: latest {latest:g}{(' ' + unit) if unit else ''} "
            f"(range {vmin:g}-{vmax:g}, n={n})"
        )
    return f"  - {name}: {latest}{(' ' + unit) if unit else ''}"


def build_user_prompt(
    admission: dict, n_xrays: int, n_ecgs: int,
    microbiology: list | None = None,
) -> str:
    """Build the masked clinical-note prompt with <image> placeholders.

    Section order is also the priority order for truncation: the trailing
    sections (labs, vitals, meds) get trimmed first; high-value sections
    (microbiology, procedures) are placed early so they survive when content
    overflows the char budget.
    """
    inp = admission["input"]
    lines = []

    demo = inp.get("demographics", {})
    admit = (inp.get("admissions") or [{}])[0]
    admittime = admit.get("admittime", "")
    age_at_admit = None
    if demo.get("anchor_age") is not None and demo.get("anchor_year") is not None and admittime:
        try:
            admit_year = int(admittime[:4])
            age_at_admit = int(demo["anchor_age"]) + (admit_year - int(demo["anchor_year"]))
        except Exception:
            age_at_admit = int(demo.get("anchor_age", 0))

    lines.append("# Demographics")
    lines.append(f"  - Sex: {demo.get('gender', 'unknown')}")
    if age_at_admit is not None:
        lines.append(f"  - Age at admission: {age_at_admit}")
    lines.append("")

    lines.append("# Admission")
    lines.append(f"  - Type: {admit.get('admission_type', 'unknown')}")
    lines.append(f"  - Source: {admit.get('admission_location', 'unknown')}")
    if admit.get("insurance"):
        lines.append(f"  - Insurance: {admit['insurance']}")
    lines.append("")

    if n_xrays > 0:
        lines.append("# Chest X-ray imaging")
        for _ in range(n_xrays):
            lines.append("<image>")
        lines.append("")

    if n_ecgs > 0:
        lines.append("# 12-lead ECG")
        for _ in range(n_ecgs):
            lines.append("<image>")
        lines.append("")

    if microbiology:
        lines.extend(format_microbiology(microbiology))

    procs = inp.get("procedures") or []
    if procs:
        lines.append("# Procedures")
        for p in procs[:30]:
            if isinstance(p, dict):
                name = p.get("long_title") or p.get("title") or p.get("icd_code") or str(p)
                lines.append(f"  - {name}")
            else:
                lines.append(f"  - {p}")
        lines.append("")

    labs = inp.get("lab_values") or {}
    lab_lines = [s for s in (summarize_series(n, ev) for n, ev in labs.items()) if s]
    if lab_lines:
        lines.append("# Laboratory results")
        lines.extend(lab_lines)
        lines.append("")

    charts = inp.get("chart_values") or {}
    chart_lines = [s for s in (summarize_series(n, ev) for n, ev in charts.items()) if s]
    if chart_lines:
        lines.append("# Vital signs and clinical observations")
        lines.extend(chart_lines)
        lines.append("")

    short = inp.get("prescription_short")
    if short:
        lines.append("# Medications administered")
        for d in short:
            lines.append(f"  - {d}")
        lines.append("")
    else:
        scripts = inp.get("prescriptions") or []
        if scripts:
            seen = set()
            uniq_drugs = []
            for p in scripts:
                d = p.get("drug")
                if d and d not in seen:
                    seen.add(d)
                    uniq_drugs.append(d)
            if uniq_drugs:
                lines.append("# Medications administered")
                for d in uniq_drugs[:60]:
                    lines.append(f"  - {d}")
                lines.append("")

    lines.append(
        "Based on this admission, what is the most likely primary diagnosis? "
        "Provide both the ICD-10 code and the diagnosis name."
    )
    return "\n".join(lines)


def truncate_to_chars(text: str, max_chars: int) -> str:
    """Cap user prompt at `max_chars` chars by trimming whole lines from the end
    of the laboratory/chart sections (the longest typical sections), but always
    keep the closing question."""
    if len(text) <= max_chars:
        return text
    closing = "Based on this admission, what is the most likely primary diagnosis? Provide both the ICD-10 code and the diagnosis name."
    body = text.rsplit(closing, 1)[0]
    while len(body) + len(closing) + 2 > max_chars and "\n" in body:
        body = body.rsplit("\n", 1)[0]
    return body.rstrip() + "\n\n" + closing


def mask_admission(admission: dict) -> dict:
    a = json.loads(json.dumps(admission))
    a.pop("survived", None)
    a.pop("48_ihm", None)
    a.pop("outputs", None)
    if a.get("input", {}).get("admissions"):
        for adm in a["input"]["admissions"]:
            for k in ("dischtime", "deathtime", "discharge_location", "hospital_expire_flag"):
                adm.pop(k, None)
    return a


def process_one(args: tuple) -> dict | None:
    (
        fp,
        e_set,
        idx2code,
        code2desc,
        hadm_to_subject,
        hadm_to_micro,
        chest_xray_root,
        ecg_root,
        ecg_png_dir,
        max_xrays,
        max_ecgs,
        max_chars,
        max_pixels_per_image,
    ) = args
    try:
        with open(fp) as f:
            d = json.load(f)
    except Exception:
        return None
    outs = d.get("outputs") or []
    if not outs:
        return None
    code = idx2code.get(outs[0])
    if not code or not ICD10_RE.match(code):
        return None
    matched = match_e(code, e_set)
    if not matched:
        return None
    desc = code2desc.get(code, "")

    hadm_id = int(os.path.basename(fp).split(".")[0])
    meta = hadm_to_subject.get(hadm_id)
    if not meta:
        return None

    images: list[dict] = []
    image_max_pixels = max_pixels_per_image
    raw_xrays = d.get("image_paths") or []
    chosen_xrays = raw_xrays[-max_xrays:] if max_xrays > 0 else []
    n_xrays_actual = 0
    for im in chosen_xrays:
        rel = im.get("path") if isinstance(im, dict) else im
        if not rel:
            continue
        full = os.path.join(chest_xray_root, rel)
        if os.path.exists(full):
            images.append({"image": full, "max_pixels": image_max_pixels})
            n_xrays_actual += 1

    n_ecgs_actual = 0
    raw_ecgs = d.get("ecg_data") or []
    for ecg in raw_ecgs[:max_ecgs]:
        if not isinstance(ecg, dict):
            continue
        rel = ecg.get("file_path")
        if not rel:
            continue
        mat_full = os.path.join(ecg_root, rel)
        if not os.path.exists(mat_full):
            continue
        study_id = ecg.get("study_id") or os.path.splitext(os.path.basename(rel))[0]
        png_path = os.path.join(ecg_png_dir, f"{study_id}.png")
        if render_ecg_png(mat_full, png_path):
            images.append({"image": png_path, "max_pixels": image_max_pixels})
            n_ecgs_actual += 1

    masked = mask_admission(d)
    micro = hadm_to_micro.get(hadm_id) if hadm_to_micro else None
    user_content = build_user_prompt(
        masked, n_xrays_actual, n_ecgs_actual, microbiology=micro,
    )
    user_content = truncate_to_chars(user_content, max_chars)

    pretty_code = code if len(code) <= 3 else f"{code[:3]}.{code[3:]}"
    answer = f"{pretty_code}: {desc}"

    return {
        "data_source": "mimiciv_rare_dx",
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "images": images,
        "reward_model": {"style": "rule", "ground_truth": answer},
        "extra_info": {
            "hadm_id": hadm_id,
            "subject_id": meta["subject_id"],
            "admittime": meta["admittime"],
            "primary_icd10": code,
            "primary_icd10_pretty": pretty_code,
            "primary_description": desc,
            "orphanet_e_match": matched,
            "n_xrays": n_xrays_actual,
            "n_ecgs": n_ecgs_actual,
        },
    }


def temporal_patient_split(entries: list, train_ratio: float):
    by_patient = defaultdict(list)
    for e in entries:
        by_patient[e["extra_info"]["subject_id"]].append(e)
    patients = []
    for sid, items in by_patient.items():
        earliest = min(it["extra_info"]["admittime"] for it in items)
        patients.append((earliest, sid, items))
    patients.sort(key=lambda x: x[0])

    total = len(entries)
    target = int(total * train_ratio)
    train, test = [], []
    cutoff = None
    cum = 0
    for earliest, sid, items in patients:
        if cum < target:
            train.extend(items)
            cum += len(items)
            cutoff = earliest
        else:
            test.extend(items)
    return train, test, cutoff


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--admissions_dir",
        default="/scratch/high_modality/multimodal/mimiciv/admissions",
    )
    parser.add_argument(
        "--icd_map",
        default="/scratch/high_modality/multimodal/mimiciv/icd_code_to_description.json",
    )
    parser.add_argument(
        "--admissions_csv",
        default="/scratch/high_modality/multimodal/mimiciv/hosp/admissions.csv.gz",
    )
    parser.add_argument(
        "--microbiology_csv",
        default="/scratch/high_modality/multimodal/mimiciv/hosp/microbiologyevents.csv.gz",
        help="MIMIC-IV microbiologyevents.csv.gz; cultures are joined to each "
             "admission by hadm_id and rendered into a # Microbiology section "
             "placed early in the prompt so it survives truncation.",
    )
    parser.add_argument(
        "--orphanet_xml",
        default="/scratch/self_evolving_datasets/orphanet/en_product1.xml",
    )
    parser.add_argument(
        "--chest_xray_root",
        default="/scratch/high_modality/multimodal/mimiciv/chest_xray",
    )
    parser.add_argument(
        "--ecg_root",
        default="/scratch/high_modality/multimodal/mimiciv",
    )
    parser.add_argument(
        "--ecg_png_dir",
        default="/scratch/self_evolving_datasets/mimiciv_rare/ecg_images",
    )
    parser.add_argument(
        "--output_dir",
        default="/scratch/self_evolving_datasets/mimiciv_rare",
    )
    parser.add_argument("--max_xrays", type=int, default=2)
    parser.add_argument("--max_ecgs", type=int, default=1)
    # Each image is downsized to <= max_pixels (Qwen3-VL respects this via
    # qwen_vl_utils.fetch_image). 256*256=65536 px -> ~84 vision tokens/image.
    # 3 images -> ~250 vision tokens. Leaves ~5K for text inside the 6K budget.
    parser.add_argument("--max_pixels_per_image", type=int, default=256 * 256)
    # Token budget: 6K total prompt. 3 images @ 256x256 ~= 750 vision tokens,
    # system+chat-template+closing ~= 450 tokens. Free for narrative text:
    # ~4800 tokens. At 2 chars/token (conservative for medical text), that's
    # ~9600 chars.
    parser.add_argument("--max_text_chars", type=int, default=9000)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--num_workers", type=int, default=32)
    args = parser.parse_args()

    print(f"Loading Orphanet E set ...")
    e_set = load_orphanet_e_codes(args.orphanet_xml)
    print(f"  E codes: {len(e_set)}")

    with open(args.icd_map) as f:
        icd_map = json.load(f)
    idx2code = {info["index"]: code.strip().upper() for code, info in icd_map.items()}
    code2desc = {code.strip().upper(): info["description"] for code, info in icd_map.items()}

    hadm_to_subject = load_subject_admit_map(args.admissions_csv)
    print(f"  hadm_ids in admissions.csv: {len(hadm_to_subject)}")

    files = sorted(glob(os.path.join(args.admissions_dir, "*.json")))
    print(f"Processing {len(files)} files with {args.num_workers} workers ...")

    keep_hadms = set()
    for fp in files:
        try:
            keep_hadms.add(int(os.path.basename(fp).split(".")[0]))
        except ValueError:
            continue
    hadm_to_micro: dict = {}
    if args.microbiology_csv and os.path.exists(args.microbiology_csv):
        print(f"Loading microbiology events from {args.microbiology_csv} ...")
        hadm_to_micro = load_microbiology_map(args.microbiology_csv, keep_hadms=keep_hadms)
        print(f"  admissions with microbiology: {len(hadm_to_micro)}")
    else:
        print("  microbiology CSV not found — skipping # Microbiology section")

    work = [
        (
            fp,
            e_set,
            idx2code,
            code2desc,
            hadm_to_subject,
            hadm_to_micro,
            args.chest_xray_root,
            args.ecg_root,
            args.ecg_png_dir,
            args.max_xrays,
            args.max_ecgs,
            args.max_text_chars,
            args.max_pixels_per_image,
        )
        for fp in files
    ]
    with Pool(args.num_workers) as pool:
        results = pool.map(process_one, work, chunksize=128)
    entries = [r for r in results if r is not None]
    print(f"  Matched (Orphanet E primary, with subject_id): {len(entries)}")

    n_with_xray = sum(1 for e in entries if e["extra_info"]["n_xrays"] > 0)
    n_with_ecg = sum(1 for e in entries if e["extra_info"]["n_ecgs"] > 0)
    print(f"  Admissions with X-ray: {n_with_xray}, with ECG: {n_with_ecg}")

    train, test, cutoff = temporal_patient_split(entries, args.train_ratio)
    print(f"Cutoff (last train earliest-admit): {cutoff}")
    print(f"  Train: {len(train)}  Test: {len(test)}")

    train_subjects = {e["extra_info"]["subject_id"] for e in train}
    test_subjects = {e["extra_info"]["subject_id"] for e in test}
    overlap = train_subjects & test_subjects
    assert not overlap, f"Patient leakage: {len(overlap)} shared subject_ids"
    print(f"  Patient-level disjoint: train={len(train_subjects)}  test={len(test_subjects)}")

    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, "train.jsonl")
    test_path = os.path.join(args.output_dir, "test.jsonl")
    for path, items in [(train_path, train), (test_path, test)]:
        with open(path, "w") as f:
            for e in items:
                f.write(json.dumps(e) + "\n")
        print(f"  Wrote {len(items)} entries to {path}")


if __name__ == "__main__":
    main()
