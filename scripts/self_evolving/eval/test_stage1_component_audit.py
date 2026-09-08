"""Offline checks for the component audit's comparisons and counting rules."""

import unittest

from stage1_component_audit import candidate_ok, generation_prompt, validator_metrics


class ComponentAuditTests(unittest.TestCase):
    def test_placeholder_is_an_outcome_not_a_valid_task(self):
        self.assertFalse(candidate_ok({"format": "free", "question": "...", "answer": "disease"}))
        candidate = {
            "format": "mcq",
            "question": "Which of these four proposed mechanisms explains the clinical finding?",
            "options": {"A": "one", "B": "two", "C": "three", "D": "four"},
            "answer": "A",
        }
        self.assertTrue(candidate_ok(candidate))
        candidate["answer"] = "E"
        self.assertFalse(candidate_ok(candidate))

    def test_feedback_ablation_removes_feedback_but_preserves_answer_rules(self):
        template = (
            "(3) the solver's recent accuracy, (4) format {required_format}\n"
            "DIFFICULTY CALIBRATION:\nAccuracy {accuracy:.0%} over {accuracy_count}; target 50%.\n"
            "ANSWER RULES:\nCorrect answers only."
        )
        off = generation_prompt(template, "mcq", None)
        self.assertNotIn("accuracy", off)
        self.assertNotIn("target", off)
        self.assertIn("Correct answers only.", off)
        self.assertIn("Accuracy 20%", generation_prompt(template, "mcq", 0.2))
        self.assertIn("Accuracy 80%", generation_prompt(template, "mcq", 0.8))

    def test_recall_uses_population_weights_not_equal_stratum_weights(self):
        results = [
            {"accepted": True, "assessment": {"quality": "valid"}},
            {"accepted": True, "assessment": {"quality": "invalid"}},
            {"accepted": False, "assessment": {"quality": "valid"}},
            {"accepted": False, "assessment": {"quality": "uncertain"}},
        ]
        metrics = validator_metrics(results, {"True": 900, "False": 100})
        self.assertEqual(metrics["accept_precision_decisive"], 0.5)
        self.assertEqual(metrics["valid_task_recall_decisive"], 0.9)


if __name__ == "__main__":
    unittest.main()
