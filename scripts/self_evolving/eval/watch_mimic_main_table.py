"""Collect independently completed MIMIC baselines into the main paper table.

Runs locally in tmux; uses only SSH/SCP to existing nodes. Remote per-case
validation produces a compact ledger, avoiding large local copies of traces.
"""

import argparse
import fcntl
import json
import re
import shlex
import subprocess
import time
from pathlib import Path

from compile_mimic_methods import CHAPTERS, LABELS, validate_protocol
from mimic_method_baselines import METHODS, write_json

BEGIN = "% BEGIN AUTO MIMIC METHOD ROWS"
END = "% END AUTO MIMIC METHOD ROWS"
MODELS = {"qwen35_9b": "Qwen3.5-9B", "qwen36_27b": "Qwen3.6-27B"}
CELL = re.compile(r"\s*(?:\\textbf\{)?(0\.\d{3}|1\.000)\}?\s*")


def render_rows(ledger):
    validate_protocol(ledger["rows"])
    lines = []
    for tag, name in MODELS.items():
        rows = [row for row in ledger["rows"][tag] if row["status"] == "complete"]
        if not rows:
            continue
        if len({row["method"] for row in rows}) != len(rows):
            raise ValueError("Duplicate completed methods")
        lines.extend([r"\midrule", rf"\multicolumn{{10}}{{l}}{{\emph{{Frozen {name} inference methods}}}} \\"])
        for row in sorted(rows, key=lambda row: METHODS.index(row["method"])):
            if not row.get("verified") or row["n"] != 2452:
                raise ValueError("Unverified or incomplete result")
            if row["model"] != "Qwen/" + name or row["judge_model"] != "gpt-chat-latest_2026-05-28":
                raise ValueError("Unexpected model or judge")
            values = [row["per_cat"][chapter] for chapter in CHAPTERS] + [row["overall"]]
            if any(not 0 <= value <= 1 for value in values):
                raise ValueError("Invalid accuracy")
            label = "Direct (matched run)" if row["method"] == "direct" else LABELS[row["method"]]
            lines.append(f"% Verified {tag}/{row['method']}; config {row['config_sha256']}")
            lines.append(r"\quad " + label + " & " + " & ".join(f"{value:.3f}" for value in values) + r" \\")
    return "\n".join(lines)


def update_table(text, ledger):
    if text.count(BEGIN) != 1 or text.count(END) != 1:
        raise ValueError("Main table must have exactly one pair of generated-row markers")
    head, rest = text.split(BEGIN)
    _, tail = rest.split(END)
    block = render_rows(ledger)
    updated = head + BEGIN + "\n" + (block + "\n" if block else "") + END + tail
    lines = updated.splitlines(keepends=True)
    cells_by_line = {}
    for index, line in enumerate(lines):
        if line.lstrip().startswith("%") or " & " not in line or r"\\" not in line:
            continue
        body, suffix = line.rsplit(r"\\", 1)
        cells = body.split(" & ")
        if len(cells) != 10:
            continue
        matches = [CELL.fullmatch(cell) for cell in cells[1:]]
        if all(matches):
            cells_by_line[index] = (cells, suffix, [float(match[1]) for match in matches])
    maxima = [max(row[2][column] for row in cells_by_line.values()) for column in range(9)]
    for index, (cells, suffix, values) in cells_by_line.items():
        scores = [
            rf"\textbf{{{value:.3f}}}" if value == maxima[i] else f"{value:.3f}" for i, value in enumerate(values)
        ]
        lines[index] = cells[0] + " & " + " & ".join(scores) + r" \\" + suffix
    return "".join(lines)


def fetch(args):
    remote = args.remote_root
    command = shlex.join(
        [
            "python",
            remote + "/code/compile_mimic_methods.py",
            "--root",
            remote,
            "--output",
            remote + "/main_table_status.json",
            "--allow-partial",
        ]
    )
    errors = []
    for port in args.ports:
        try:
            options = [
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=8",
                "-o",
                "ServerAliveInterval=5",
                "-o",
                "ServerAliveCountMax=2",
            ]
            subprocess.run(
                ["ssh", *options, "-p", str(port), args.host, command],
                check=True,
                capture_output=True,
                text=True,
                timeout=180,
            )
            temporary = args.archive / "incoming.json"
            subprocess.run(
                [
                    "scp",
                    *options,
                    "-P",
                    str(port),
                    args.host + ":" + remote + "/main_table_status.json",
                    str(temporary),
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )
            ledger = json.loads(temporary.read_text())
            render_rows(ledger)
            previous_path = args.archive / "status.json"
            if previous_path.exists():
                previous = json.loads(previous_path.read_text())
                current = {(tag, row["method"]): row for tag, rows in ledger["rows"].items() for row in rows}
                for tag, rows in previous.get("rows", {}).items():
                    for row in rows:
                        if row["status"] != "complete":
                            continue
                        replacement = current[(tag, row["method"])]
                        for key in ("status", "config_sha256", "overall", "per_cat"):
                            if replacement.get(key) != row[key]:
                                raise ValueError(f"Previously published result changed: {tag}/{row['method']}")
            temporary.replace(args.archive / "status.json")
            return ledger
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            detail = getattr(exc, "stderr", "") or ""
            errors.append(f"port {port}: {exc}; {detail[-1000:]}")
    raise RuntimeError("; ".join(errors))


def refresh(args):
    ledger = fetch(args)
    table = args.paper / "tables/main_results.tex"
    before = table.read_text()
    after = update_table(before, ledger)
    changed = after != before
    if changed:
        # Preserve an audit copy and avoid overwriting a concurrent user edit.
        if table.read_text() != before:
            raise RuntimeError("Main table changed while preparing the update; retrying next poll")
        stamp = time.strftime("%Y%m%d_%H%M%S")
        (args.archive / f"main_results_before_{stamp}.tex").write_text(before)
        temporary = table.with_suffix(".tex.tmp")
        temporary.write_text(after)
        temporary.replace(table)
        (args.archive / "pdf_pending").touch()
    if (args.archive / "pdf_pending").exists():
        with (args.archive / "latexmk.log").open("w") as log:
            subprocess.run(
                ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "iclr2026_conference.tex"],
                cwd=args.paper,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=180,
            )
        (args.archive / "pdf_pending").unlink()
    count = sum(row["status"] == "complete" for rows in ledger["rows"].values() for row in rows)
    status = {
        "checked_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "completed": count,
        "total": len(METHODS) * len(MODELS),
        "table_changed": changed,
        "main_table": str(table),
        "incomplete": ledger["incomplete"],
    }
    write_json(args.archive / "watcher_status.json", status)
    print(json.dumps(status), flush=True)
    return count == status["total"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[3]
    parser.add_argument("--paper", type=Path, default=root / "paper")
    parser.add_argument("--archive", type=Path, default=root / "paper_data/stage1/method_baselines")
    parser.add_argument("--remote-root", default="/scratch/sheng/self_evolving/stage1_method_baselines_v2")
    parser.add_argument("--host", default="root@point.dd.works")
    parser.add_argument("--ports", type=int, nargs="+", default=[2333, 2335])
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval < 30:
        parser.error("--interval must be at least 30 seconds")
    args.archive.mkdir(parents=True, exist_ok=True)
    with (args.archive / "watcher.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            try:
                complete = refresh(args)
            except Exception as exc:
                print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} ERROR {exc}", flush=True)
                if args.once:
                    raise
            else:
                if complete or args.once:
                    return
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
