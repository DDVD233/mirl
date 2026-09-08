"""Guard against partial or mixed-grader HealthBench figure inputs."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import refresh_stage1_comparisons as refresh


class HealthBenchRefreshTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        root_patch = patch.object(refresh, "ROOT", self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)
        self.summary = {
            "grader_model": "gpt-5.4_2026-03-05",
            "n_examples": 525,
            "stamp": "test",
            "overall_score": 0.5,
            "overall_score_length_adjusted": 0.5,
        }
        self.examples = [
            {"prompt_id": str(i), "score": 0.5, "completion": [{"content": "x" * 2000}]} for i in range(525)
        ]
        self.inputs = [
            {
                "prompt_id": str(i),
                "example_tags": ["use_case:consult", "type:good_faith", "difficulty:typical"],
            }
            for i in range(525)
        ]

    def aggregate(self):
        (self.root / "result__test.json").write_text(json.dumps(self.summary))
        (self.root / "allresults_test.json").write_text(
            json.dumps({"metadata": {"example_level_metadata": self.examples}})
        )
        (self.root / "healthbench_professional_test.jsonl").write_text(
            "\n".join(json.dumps(row) for row in self.inputs)
        )
        return refresh.aggregate_healthbench(self.root)

    def test_complete_official_run(self):
        self.assertEqual(self.aggregate()["overall"], {"n": 525, "raw": 0.5, "length_adjusted": 0.5})

    def test_wrong_grader(self):
        self.summary["grader_model"] = "old-grader"
        with self.assertRaisesRegex(ValueError, "Wrong grader"):
            self.aggregate()

    def test_nonfinite_grade(self):
        self.examples[0]["score"] = float("nan")
        with self.assertRaisesRegex(ValueError, "Missing per-example grade"):
            self.aggregate()

    def test_duplicate_input(self):
        self.inputs.append(self.inputs[0])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.aggregate()

    def test_duplicate_grade(self):
        self.examples[-1] = self.examples[0]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.aggregate()

    def test_inconsistent_summary(self):
        self.summary["overall_score_length_adjusted"] = 0.6
        with self.assertRaisesRegex(ValueError, "disagrees"):
            self.aggregate()


if __name__ == "__main__":
    unittest.main()
