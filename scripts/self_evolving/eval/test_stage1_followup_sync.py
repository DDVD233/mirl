"""Prevent publishing smoke runs or initial validation as a trained control."""

import unittest

from stage1_component_audit import ARMS
from sync_stage1_followups import component_table, control_table


class FollowupSyncTests(unittest.TestCase):
    def test_initial_validation_does_not_publish_training_row(self):
        self.assertEqual(control_table("unchanged", {"points": [{"step": 0}]}), "unchanged")

    def test_partial_control_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            control_table("unchanged", {"points": [{"step": 20, "n": 2451}]})

    def test_smoke_component_matrix_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            component_table({"arms": {arm: {"n": 2} for arm in ARMS}})

    def test_complete_components_render(self):
        summary = {
            "arms": {
                arm: {
                    "n": 64,
                    "reference_valid": 32,
                    "validator_accepted": 48,
                    "reference_valid_tasks": {"accuracy": 0.4, "mixed_reward_groups": 0.7},
                }
                for arm in ARMS
            }
        }
        rendered = component_table(summary)
        self.assertEqual(rendered.count("0.500 & 0.750 & 0.400 & 0.700"), 6)
        self.assertIn(r"50\%", rendered)


if __name__ == "__main__":
    unittest.main()
