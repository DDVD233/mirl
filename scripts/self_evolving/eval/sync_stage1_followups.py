"""Publish only complete component summaries and fixed-judge control checkpoints."""

import argparse
import fcntl
import json
import shlex
import subprocess
import time
from pathlib import Path

from compile_mimic_methods import CHAPTERS
from mimic_method_baselines import write_json
from publish_stage1_paper import check_publish_workspace, publish
from stage1_component_audit import ARMS
from watch_mimic_main_table import BEGIN, update_table

ROOT = Path(__file__).resolve().parents[3]
ARCHIVE = ROOT / "paper_data/stage1/component_audit"
CONTROL_BEGIN = "% BEGIN AUTO 9B CURATED CONTROL"
CONTROL_END = "% END AUTO 9B CURATED CONTROL"
ARM_NAMES = [
    r"16 passages, 50\% feedback",
    r"3 passages, 50\% feedback",
    r"No passages, 50\% feedback",
    "16 passages, no feedback",
    r"16 passages, 20\% feedback",
    r"16 passages, 80\% feedback",
]


def fetch():
    code = """import json
from pathlib import Path
s = Path('/scratch/sheng/self_evolving')
paths = {
    'components': s/'stage1_component_audit/results/components.summary.json',
    'manifest': s/'stage1_component_audit/results/manifest.json',
    'control': s/'logs_stage1_9b_control/fixed_judge/curve.json',
}
print(json.dumps({k: json.loads(p.read_text()) for k,p in paths.items() if p.exists()}))
"""
    failures = []
    for port in (2333, 2335):
        try:
            output = subprocess.check_output(
                [
                    "ssh",
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    "ConnectTimeout=8",
                    "-p",
                    str(port),
                    "root@point.dd.works",
                    shlex.join(["python", "-c", code]),
                ],
                text=True,
                stderr=subprocess.PIPE,
                timeout=90,
            )
            result = json.loads(output)
            if not isinstance(result, dict) or not {"components", "control"}.intersection(result):
                raise ValueError("No stage-1 summaries on this endpoint")
            return result
        except (subprocess.SubprocessError, ValueError) as exc:
            detail = exc.stderr.strip() if isinstance(exc, subprocess.CalledProcessError) else str(exc)
            failures.append(f"SSH {port}: {detail}")
    raise RuntimeError("; ".join(failures))


def finish_pending(push):
    pending = ARCHIVE / "pdf_pending"
    publication = ARCHIVE / "publish_pending"
    if pending.exists():
        with (ARCHIVE / "latexmk.log").open("w") as log:
            subprocess.run(
                ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "iclr2026_conference.tex"],
                cwd=ROOT / "paper",
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=180,
            )
        pending.unlink()
    if push and publication.exists():
        revision = publish(ROOT / "paper")
        write_json(ARCHIVE / "published.json", {"commit": revision, "published_at": time.time()})
        publication.unlink()


def component_table(summary):
    arms = summary["arms"]
    if set(arms) != set(ARMS) or any(row["n"] != 64 for row in arms.values()):
        raise ValueError("Incomplete component matrix")
    lines = [
        r"\subsection{Generation-Level Component Checks}",
        r"\label{app:components}",
        "We pair six conditions on 64 text-only training admissions (32 MCQ and 32 free response).",
        "The same frozen 27B model proposes tasks and supplies eight solver samples per formatted task.",
        "Grounding varies the number of passages from the shared medical index; difficulty feedback",
        "varies the reported recent solver accuracy or removes the calibration instruction.",
        "Validity is assessed by the independent model assessor from Appendix~\\ref{app:judging}.",
        "These checks measure generated tasks, not downstream RL outcomes.",
        r"\begin{table}[h]\centering\small",
        r"\caption{Paired component checks. Validity and acceptance use all 64 candidates as denominator.",
        "Solver accuracy and mixed-reward group frequency are conditional on reference-valid tasks.",
        "Uncertain assessments are not counted as valid. Single-seed-generation estimates are descriptive.}",
        r"\label{tab:components}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Condition & Valid & Accepted & Solver acc. & Mixed groups \\",
        r"\midrule",
    ]
    for arm, label in zip(ARMS, ARM_NAMES, strict=True):
        row = arms[arm]
        valid = row["reference_valid_tasks"]
        values = [
            row["reference_valid"] / 64,
            row["validator_accepted"] / 64,
            valid["accuracy"] if valid else None,
            valid["mixed_reward_groups"] if valid else None,
        ]
        lines.append(label + " & " + " & ".join("--" if v is None else f"{v:.3f}" for v in values) + r" \\")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}\end{table}", ""])


def control_table(before, curve):
    points = [p for p in curve["points"] if p["step"] > 0]
    if not points:
        return before
    for row in points:
        if row["n"] != 2452 or row["judge_model"] != "gpt-chat-latest_2026-05-28":
            raise ValueError("Incomplete or wrong-judge control result")
        if set(row["per_cat"]) != set(CHAPTERS) or sum(row["cat_n"].values()) != 2452:
            raise ValueError("Invalid control chapter coverage")
        overall = sum(row["per_cat"][c] * row["cat_n"][c] for c in CHAPTERS) / 2452
        if abs(overall - row["overall"]) > 1e-10:
            raise ValueError("Control summary mismatch")
    best = max(points, key=lambda p: p["overall"])
    cells = [best["per_cat"][c] for c in CHAPTERS] + [best["overall"]]
    block = "\n".join(
        [
            CONTROL_BEGIN,
            r"\midrule",
            r"\multicolumn{10}{l}{\emph{9B curated-data control (SFT + RL, self-judge)}} \\",
            f"% Best completed fixed-judge checkpoint: step {best['step']}; trace {best['trace_sha256']}",
            "Qwen3.5-9B & " + " & ".join(f"{v:.3f}" for v in cells) + r" \\",
            CONTROL_END,
        ]
    )
    if CONTROL_BEGIN in before:
        head, rest = before.split(CONTROL_BEGIN)
        _, tail = rest.split(CONTROL_END)
        after = head + block + tail
    else:
        after = before.replace(BEGIN, block + "\n" + BEGIN)
    ledger = json.loads((ROOT / "paper_data/stage1/method_baselines/status.json").read_text())
    return update_table(after, ledger)


def refresh(push=False):
    publication = ARCHIVE / "publish_pending"
    if push:
        check_publish_workspace(ROOT / "paper", pending=publication.exists())
    # Previously collected results must not wait for the next successful SSH fetch.
    finish_pending(push)
    results = fetch()
    changed = False
    if "components" in results:
        pinned = json.loads((ARCHIVE / "manifest.json").read_text())
        if pinned != results["manifest"]:
            raise ValueError("Component run provenance changed")
        table = ROOT / "paper/tables/component_audit.tex"
        rendered = component_table(results["components"])
        if not table.exists() or table.read_text() != rendered:
            table.write_text(rendered)
            changed = True
        write_json(ARCHIVE / "components.summary.json", results["components"])
    if "control" in results:
        write_json(ARCHIVE / "control_curve.json", results["control"])
        table = ROOT / "paper/tables/main_results.tex"
        before = table.read_text()
        after = control_table(before, results["control"])
        if before != after:
            if table.read_text() != before:
                raise RuntimeError("Concurrent main-table edit; retrying")
            write_json(ARCHIVE / f"main_table_before_{int(time.time())}.json", {"text": before})
            table.write_text(after)
            changed = True
    pending = ARCHIVE / "pdf_pending"
    if changed:
        pending.touch()
        if push:
            publication.touch()
    finish_pending(push)
    status = {
        "checked_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "paper_changed": changed,
        "components_complete": "components" in results,
        "control_steps": [p["step"] for p in results.get("control", {}).get("points", [])],
    }
    write_json(ARCHIVE / "sync_status.json", status)
    print(json.dumps(status), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--push", action="store_true", help="Commit and push completed paper updates to origin/main")
    args = parser.parse_args()
    with (ARCHIVE / "sync.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            try:
                refresh(push=args.push)
            except Exception as exc:
                print(f"ERROR: {exc}", flush=True)
                if args.once:
                    raise
            if args.once:
                break
            time.sleep(300)


if __name__ == "__main__":
    main()
