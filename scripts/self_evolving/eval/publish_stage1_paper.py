"""Publish generated paper updates without staging unrelated edits or force-pushing."""

import subprocess

GENERATED = ("tables/main_results.tex", "tables/component_audit.tex", "iclr2026_conference.pdf")


def git(paper, *args):
    return subprocess.check_output(["git", *args], cwd=paper, text=True, stderr=subprocess.STDOUT, timeout=120).strip()


def check_publish_workspace(paper, pending=False):
    if git(paper, "branch", "--show-current") != "main":
        raise RuntimeError("Automatic paper publication requires main")
    if git(paper, "diff", "--cached", "--name-only"):
        raise RuntimeError("Existing staged changes; refusing automatic publication")
    changed = set(git(paper, "diff", "--name-only").splitlines())
    if changed - set(GENERATED) or (changed and not pending):
        raise RuntimeError("Existing paper edits; refusing automatic publication")


def publish(paper):
    check_publish_workspace(paper, pending=True)
    paths = [p for p in GENERATED if (paper / p).exists()]
    git(paper, "add", "--", *paths)
    if git(paper, "diff", "--cached", "--name-only"):
        git(
            paper,
            "commit",
            "-m",
            "Update completed stage-1 results",
            "-m",
            "Co-authored-by: Codex <noreply@openai.com>",
        )
    git(paper, "push", "origin", "HEAD:refs/heads/main")
    return git(paper, "rev-parse", "HEAD")
