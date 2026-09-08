import subprocess
import tempfile
import unittest
from pathlib import Path

from publish_stage1_paper import check_publish_workspace, git, publish


class PublisherTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.paper = self.root / "paper"
        self.paper.mkdir()
        subprocess.run(["git", "init", "--bare", str(self.root / "remote.git")], check=True, capture_output=True)
        git(self.paper, "init", "-b", "main")
        git(self.paper, "config", "user.name", "Test")
        git(self.paper, "config", "user.email", "test@example.com")
        git(self.paper, "remote", "add", "origin", str(self.root / "remote.git"))
        (self.paper / "tables").mkdir()
        (self.paper / "tables/main_results.tex").write_text("old")
        (self.paper / "unrelated.txt").write_text("untouched")
        git(self.paper, "add", ".")
        git(self.paper, "commit", "-m", "Initial")

    def test_publish_only_generated_files(self):
        (self.paper / "tables/main_results.tex").write_text("completed")
        (self.paper / "private-untracked.txt").write_text("never publish")
        revision = publish(self.paper)
        self.assertEqual(git(self.paper, "rev-parse", "origin/main"), revision)
        self.assertEqual(git(self.paper, "show", "--format=", "--name-only", "HEAD"), "tables/main_results.tex")

    def test_refuse_unrelated_edits(self):
        (self.paper / "unrelated.txt").write_text("user edit")
        with self.assertRaisesRegex(RuntimeError, "Existing paper edits"):
            publish(self.paper)

    def test_refuse_existing_staging(self):
        (self.paper / "tables/main_results.tex").write_text("user staged edit")
        git(self.paper, "add", "tables/main_results.tex")
        with self.assertRaisesRegex(RuntimeError, "staged"):
            publish(self.paper)

    def test_refuse_preexisting_generated_file_edits(self):
        (self.paper / "tables/main_results.tex").write_text("user edit")
        with self.assertRaisesRegex(RuntimeError, "Existing paper edits"):
            check_publish_workspace(self.paper)


if __name__ == "__main__":
    unittest.main()
