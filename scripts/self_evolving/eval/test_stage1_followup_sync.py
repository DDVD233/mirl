"""Prevent publishing smoke runs or initial validation as a trained control."""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import sync_stage1_followups as sync
from stage1_component_audit import ARMS
from sync_stage1_followups import component_table, control_table


class FollowupSyncTests(unittest.TestCase):
    def test_fetch_falls_back_to_second_shared_storage_node(self):
        error = subprocess.CalledProcessError(255, "ssh", stderr="Connection refused")
        with patch.object(sync.subprocess, "check_output", side_effect=[error, '{"control": {}}']) as command:
            self.assertEqual(sync.fetch(), {"control": {}})
        self.assertIn("2335", command.call_args.args[0])

    def test_fetch_rejects_empty_endpoint(self):
        with patch.object(sync.subprocess, "check_output", return_value="{}"):
            with self.assertRaisesRegex(RuntimeError, "No stage-1 summaries"):
                sync.fetch()

    def test_pending_push_finishes_before_failed_fetch(self):
        with tempfile.TemporaryDirectory() as temp:
            archive = Path(temp)
            marker = archive / "publish_pending"
            marker.touch()
            with (
                patch.object(sync, "ARCHIVE", archive),
                patch.object(sync, "check_publish_workspace"),
                patch.object(sync, "publish", return_value="completed") as publish,
                patch.object(sync, "fetch", side_effect=RuntimeError("offline")),
            ):
                with self.assertRaisesRegex(RuntimeError, "offline"):
                    sync.refresh(push=True)
            publish.assert_called_once()
            self.assertFalse(marker.exists())
            self.assertEqual(json.loads((archive / "published.json").read_text())["commit"], "completed")

    def test_failed_push_retains_pending_marker(self):
        with tempfile.TemporaryDirectory() as temp:
            archive = Path(temp)
            marker = archive / "publish_pending"
            marker.touch()
            with (
                patch.object(sync, "ARCHIVE", archive),
                patch.object(sync, "publish", side_effect=RuntimeError("push failed")),
            ):
                with self.assertRaisesRegex(RuntimeError, "push failed"):
                    sync.finish_pending(True)
            self.assertTrue(marker.exists())

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
