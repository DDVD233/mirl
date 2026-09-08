"""Incremental publication must preserve the paper and reject invalid grades."""

import copy
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import compile_mimic_methods as compiler
import watch_mimic_main_table as watcher
from mimic_method_baselines import digest, file_digest


def ledger_fixture():
    row = {
        "status": "complete",
        "verified": True,
        "n": 2452,
        "method": "medrag",
        "model": "Qwen/Qwen3.5-9B",
        "judge_model": "gpt-chat-latest_2026-05-28",
        "per_cat": dict.fromkeys(compiler.CHAPTERS, 0.4),
        "overall": 0.4,
        "config_sha256": "fixture",
    }
    return {"rows": {"qwen35_9b": [row, {"method": "imedrag", "status": "pending"}], "qwen36_27b": []}}


class TableTests(unittest.TestCase):
    def test_incremental_rows_and_global_highlighting_are_idempotent(self):
        text = (
            "Preserve this caption.\n"
            + r"Base & "
            + " & ".join([r"\textbf{0.300}"] * 9)
            + " "
            + r"\\"
            + "\n"
            + watcher.BEGIN
            + "\n"
            + watcher.END
            + "\n% User note\n"
        )
        with patch.object(watcher, "validate_protocol"):
            updated = watcher.update_table(text, ledger_fixture())
            self.assertEqual(watcher.update_table(updated, ledger_fixture()), updated)
        self.assertIn("Preserve this caption.", updated)
        self.assertIn("% User note", updated)
        self.assertIn("Medical RAG", updated)
        self.assertNotIn("i-MedRAG", updated)
        self.assertNotIn(r"\textbf{0.300}", updated)
        self.assertEqual(updated.count(r"\textbf{0.400}"), 9)
        self.assertEqual(updated.count(r"\\"), 3)

    def test_unverified_result_cannot_enter_table(self):
        ledger = ledger_fixture()
        ledger["rows"]["qwen35_9b"][0]["verified"] = False
        with patch.object(watcher, "validate_protocol"):
            with self.assertRaisesRegex(ValueError, "Unverified"):
                watcher.render_rows(ledger)

    def test_pending_partial_jsonl_is_never_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "medrag.jsonl").write_text('{"incomplete":')
            self.assertEqual(compiler.collect(root, "medrag")["status"], "pending")

    def test_full_result_is_verified_and_wrong_chapter_score_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entries = [
                {
                    "extra_info": {"hadm_id": i},
                    "reward_model": {"ground_truth": "synthetic"},
                    "data_source": "mimic_rare/" + compiler.CHAPTERS[i % 8],
                }
                for i in range(2452)
            ]
            val_file = root / "test.jsonl"
            val_file.write_text("".join(json.dumps(row) + "\n" for row in entries))
            config = {
                "val_file": str(val_file),
                "dataset_sha256": file_digest(val_file),
                "model": "Qwen/Qwen3.5-9B",
                "judge_model": "fixed",
                "code_sha256": "test-implementation",
            }
            config_hash = digest(config)
            (root / "manifest.json").write_text(json.dumps({"config": config, "config_sha256": config_hash}))
            generations = [
                {
                    "hadm_id": row["extra_info"]["hadm_id"],
                    "data_source": row["data_source"],
                    "ground_truth": "synthetic",
                    "response": "synthetic",
                    "extracted_answer": "synthetic",
                    "method": "medrag",
                    "model": config["model"],
                    "config_sha256": config_hash,
                    "trace": [],
                }
                for row in entries
            ]
            grades = [dict(row, judge_acc_lenient=1.0) for row in generations]
            for name, rows in (("medrag.jsonl", generations), ("medrag.graded.jsonl", grades)):
                (root / name).write_text("".join(json.dumps(row) + "\n" for row in rows))
            counts = Counter(row["data_source"].split("/")[-1] for row in entries)
            summary = {
                "method": "medrag",
                "model": config["model"],
                "judge_model": "fixed",
                "config_sha256": config_hash,
                "n": 2452,
                "overall": 1.0,
                "cat_n": dict(counts),
                "per_cat": dict.fromkeys(counts, 1.0),
            }
            path = root / "medrag.summary.json"
            path.write_text(json.dumps(summary))
            self.assertTrue(compiler.collect(root, "medrag")["verified"])
            generations[0]["trace"] = [
                {"kind": "retrieval", "documents": [], "searches": [{"depth": 32}, {"depth": 128}]},
                {"kind": "retrieval", "documents": [{"text": "synthetic evidence"}]},
            ]
            generations[0]["implementation_sha256"] = "failure-repair"
            (root / "medrag.jsonl").write_text("".join(json.dumps(row) + "\n" for row in generations))
            with self.assertRaisesRegex(ValueError, "Unregistered generation implementation"):
                compiler.collect(root, "medrag")
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "config": config,
                        "config_sha256": config_hash,
                        "implementation_history": {"failure-repair": {"reason": "synthetic recovery"}},
                    }
                )
            )
            collected = compiler.collect(root, "medrag")
            self.assertTrue(collected["verified"])
            self.assertEqual(collected["retrieval_calls"], 2)
            self.assertEqual(collected["empty_retrieval_calls"], 1)
            self.assertEqual(collected["deepened_retrieval_calls"], 1)
            self.assertEqual(collected["retrieval_calls_with_depth_audit"], 1)
            broken = copy.deepcopy(summary)
            broken["per_cat"][compiler.CHAPTERS[0]] = 0.5
            path.write_text(json.dumps(broken))
            with self.assertRaisesRegex(ValueError, "Chapter grades"):
                compiler.collect(root, "medrag")


if __name__ == "__main__":
    unittest.main()
