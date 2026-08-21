"""Domain-bundle integration tests (scripts/self_evolving/domains.py).

Three guarantees, all without GPU or network:

  (a) With SE_DOMAIN unset, every model-facing prompt constant in the four integrated
      files is BYTE-IDENTICAL to the version at git HEAD. The HealthBench arms are
      running on this code; the medical bundle must be a pure identity.
  (b) With SE_DOMAIN=prbench / profbench, no ``[[DOMAIN_`` token survives, the
      import-time asserts pass, and no prompt still says physician / clinician /
      clinical / medical / HealthBench / patient. Leftovers are printed with the
      constant name so the bundle's REBRAND list can be extended.
  (c) The bundle's DATASET_BRIEF leads the generator / proposer templates (simple and
      rubric variants) and the meta-optimizer / hack-memo prompts for non-medical
      domains, and appears nowhere for medical.

Each extraction runs in a SUBPROCESS because the bundle reads SE_DOMAIN at import.
Constants are read by executing each file's top-level statements one at a time (so a
missing heavy dependency skips a statement instead of aborting the extraction); the
HEAD version comes from ``git show HEAD:<path>`` and is executed the same way.

Run from the repo root:
    python -m pytest scripts/self_evolving/tests/test_domains.py -q
"""
from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SE_DIR = REPO / "scripts" / "self_evolving"

FILES = {
    "generation_server": "scripts/self_evolving/generation_server.py",
    "web_evidence": "scripts/self_evolving/kb/web_evidence.py",
    "agent_loop": "verl/experimental/agent_loop/retrieval_tool_agent_loop.py",
    "spec_gap": "verl/utils/reward_score/spec_gap.py",
}

# Constants that exist at HEAD and must be byte-identical for medical.
HEAD_CONSTANTS = {
    "generation_server": [
        # taxonomy
        "HB_USE_CASES", "HB_USE_CASE_DESC", "HB_SPECIALTIES", "HB_REDTEAM_SHARE",
        "HB_MODE_INSTR", "HB_REF_STATS", "HB_LANGUAGES",
        # templates
        "SIMPLE_PROPOSER_DEFAULT", "SIMPLE_GENERATOR_DEFAULT",
        "RUBRIC_PROPOSER_DEFAULT", "RUBRIC_GENERATOR_DEFAULT", "RUBRIC_SOLVER_SYSTEM",
        # every other model-facing constant
        "EVOLVE_PER_CASE_SYSTEM", "EVOLVE_AGGREGATE_SYSTEM", "COVERAGE_EVOLVE_SYSTEM",
        "HACK_MODES", "HACK_MINT_SYSTEM", "HACK_MINT_MULTI_SYSTEM", "HACK_MEMO_SYSTEM",
        "HACK_REWRITE_SYSTEM", "RUBRIC_GOLD_SYSTEM_PROMPT", "_GRADER_STRICT_NOTE",
        "HB_FARMER_SYSTEM_PROMPT", "SUMMARY_SYSTEM", "SUMMARY_USER", "SOLVER_EVOLVE_SYSTEM",
        "EVOLVE_FAILURE_MODES", "EVOLVE_RUBRIC_DEFECTS", "EVOLVE_TASK_DEFECTS",
        "HB_SCORE_FORMULA",
    ],
    "web_evidence": ["SYSTEM"],
    "agent_loop": ["RETRIEVE_INSTRUCTION", "WEB_INSTRUCTION", "HARD_ANSWER_INSTRUCTION",
                   "BUDGET_EXHAUSTED_ERROR", "CLOSED_NOTICE"],
    "spec_gap": ["REFEREE_SYSTEM", "REFEREE_TEMPLATE"],
}
# Added by the integration; checked for leftovers only.
NEW_CONSTANTS = {
    "agent_loop": ["WEB_ONLY_INSTRUCTION", "WEB_ONLY_HARD_ANSWER_INSTRUCTION",
                   "WEB_ONLY_BUDGET_EXHAUSTED_ERROR", "WEB_ONLY_CLOSED_NOTICE"],
}
# Closed vocabularies: list KEYS are counted in code and must never be renamed.
CLOSED_VOCAB = ["HACK_MODES", "EVOLVE_FAILURE_MODES", "EVOLVE_RUBRIC_DEFECTS",
                "EVOLVE_TASK_DEFECTS"]
# Templates that must carry the DATASET_BRIEF as a leading paragraph (non-medical).
BRIEF_CARRIERS = ["SIMPLE_PROPOSER_DEFAULT", "SIMPLE_GENERATOR_DEFAULT",
                  "RUBRIC_PROPOSER_DEFAULT", "RUBRIC_GENERATOR_DEFAULT",
                  "EVOLVE_AGGREGATE_SYSTEM", "HACK_MEMO_SYSTEM"]

FORBIDDEN = re.compile(r"physician|clinician|clinical|medical|healthbench|patient", re.I)
# Backtick-quoted identifiers are tool NAMES (e.g. the default `search_medical_kb`),
# owned by the tool yamls, not prompt wording.
_IDENT = re.compile(r"`[^`]*`")


# ----------------------------------------------------------------------------
# Extraction (runs in the subprocess)
# ----------------------------------------------------------------------------
def _exec_constants(src: str, filename: str, names: list[str]) -> tuple[dict, list[str]]:
    """Execute top-level statements one by one; return {name: value} and errors.

    Errors are recorded only for statements that MATTER: asserts (the import-time
    label check) and assignments to a requested name. Anything else that fails
    (a heavy import, a class body) is skipped.
    """
    tree = ast.parse(src, filename)
    ns: dict = {"__file__": filename, "__name__": "_se_probe"}
    errors: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            continue
        try:
            exec(compile(ast.Module([node], []), filename, "exec"), ns)  # noqa: S102
        except SystemExit as e:  # the bundle refuses an unknown SE_DOMAIN
            errors.append(f"line {node.lineno}: SystemExit {e}")
        except Exception as e:  # noqa: BLE001
            targets = []
            if isinstance(node, (ast.Assign, ast.AugAssign)):
                tg = node.targets if isinstance(node, ast.Assign) else [node.target]
                targets = [t.id for t in tg if isinstance(t, ast.Name)]
            if isinstance(node, ast.Assert) or any(t in names for t in targets):
                errors.append(f"line {node.lineno}: {type(e).__name__}: {e}")
    found = {n: ns[n] for n in names if n in ns}
    return found, errors


def _dump(source: str, domain: str | None) -> dict:
    """Subprocess body: extract every constant from HEAD or the working tree."""
    out: dict = {"errors": {}, "consts": {}}
    for key, rel in FILES.items():
        path = REPO / rel
        if source == "head":
            src = subprocess.check_output(["git", "show", f"HEAD:{rel}"], cwd=REPO, text=True)
            names = HEAD_CONSTANTS[key]
        else:
            src = path.read_text()
            names = HEAD_CONSTANTS[key] + NEW_CONSTANTS.get(key, [])
        found, errors = _exec_constants(src, str(path), names)
        missing = [n for n in names if n not in found]
        if missing:
            errors.append(f"constants not found: {missing}")
        out["consts"][key] = found
        out["errors"][key] = errors
    if source == "tree":
        # Real import of the server module: the import-time asserts run for real.
        try:
            import importlib
            importlib.import_module("generation_server")
            out["import_ok"] = True
        except BaseException as e:  # noqa: BLE001
            out["import_ok"] = False
            out["import_error"] = f"{type(e).__name__}: {e}"
        import domains  # noqa: E402
        out["brief"] = domains.BUNDLE["DATASET_BRIEF"]
        out["domain"] = domains.DOMAIN
    return out


def _run_dump(source: str, domain: str | None) -> dict:
    env = {k: v for k, v in os.environ.items() if k != "SE_DOMAIN"}
    if domain:
        env["SE_DOMAIN"] = domain
    env["PYTHONPATH"] = os.pathsep.join(p for p in [str(SE_DIR), str(REPO), env.get("PYTHONPATH")] if p)
    proc = subprocess.run(
        [sys.executable, __file__, "--dump", source],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, f"dump failed ({source}, {domain}):\n{proc.stderr[-4000:]}"
    marker = proc.stdout.rfind("\n__DUMP__")
    assert marker >= 0, f"no dump marker in output:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
    return json.loads(proc.stdout[marker + len("\n__DUMP__"):])


def _strings(value) -> list[str]:
    """Every string inside a constant (str, list, dict values)."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in _strings(v)]
    if isinstance(value, (list, tuple)):
        return [s for v in value for s in _strings(v)]
    return []


# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------
def test_medical_is_byte_identical_to_head():
    head = _run_dump("head", None)
    tree = _run_dump("tree", None)
    assert tree["domain"] == "medical"
    assert tree["import_ok"], tree.get("import_error")
    for key in FILES:
        assert not head["errors"][key], f"HEAD extraction ({key}): {head['errors'][key]}"
        assert not tree["errors"][key], f"tree extraction ({key}): {tree['errors'][key]}"
        for name in HEAD_CONSTANTS[key]:
            h, t = head["consts"][key][name], tree["consts"][key][name]
            assert t == h, f"{key}.{name} differs from HEAD for the medical domain"
    assert tree["brief"] == ""
    for name in BRIEF_CARRIERS:
        assert "TARGET BENCHMARK" not in tree["consts"]["generation_server"][name]


def _leftovers(dump: dict) -> list[str]:
    report = []
    for key in FILES:
        for name, value in dump["consts"][key].items():
            for s in _strings(value):
                if "[[DOMAIN_" in s:
                    report.append(f"{key}.{name}: unfilled [[DOMAIN_ token")
                masked = _IDENT.sub("`<tool>`", s)
                for m in FORBIDDEN.finditer(masked):
                    ctx = masked[max(0, m.start() - 50): m.end() + 50].replace("\n", " ")
                    report.append(f"{key}.{name}: '{m.group(0)}' in ...{ctx}...")
    return report


def _check_domain(domain: str):
    head = _run_dump("head", None)
    tree = _run_dump("tree", domain)
    assert tree["domain"] == domain
    assert tree["import_ok"], f"import under SE_DOMAIN={domain}: {tree.get('import_error')}"
    for key in FILES:
        assert not tree["errors"][key], f"{domain} extraction ({key}): {tree['errors'][key]}"
    report = _leftovers(tree)
    print(f"\n=== {domain}: {len(report)} leftover(s) ===")
    for line in report:
        print("  " + line)
    assert not report, f"{domain}: {len(report)} medical leftover(s); see report above"

    gs = tree["consts"]["generation_server"]
    # Closed vocabularies keep their keys; only descriptions (comments) may change.
    for name in CLOSED_VOCAB:
        assert gs[name] == head["consts"]["generation_server"][name], name
    # Taxonomy consistency (the import-time assert, re-stated here explicitly).
    assert set(gs["HB_REF_STATS"]["use_case_mix"]) == set(gs["HB_USE_CASES"])
    assert set(gs["HB_USE_CASE_DESC"]) == set(gs["HB_USE_CASES"])
    assert set(gs["HB_MODE_INSTR"]) == {"good_faith", "red_teaming"}
    assert gs["HB_SPECIALTIES"] and "cardiology" not in gs["HB_SPECIALTIES"]
    # DATASET_BRIEF leads the generator / proposer / meta prompts, once each.
    brief = tree["brief"]
    assert brief.startswith("TARGET BENCHMARK")
    for name in BRIEF_CARRIERS:
        assert gs[name].startswith(brief + "\n\n"), f"{name} does not lead with DATASET_BRIEF"
        assert gs[name].count("TARGET BENCHMARK") == 1, name
    for name in ("EVOLVE_PER_CASE_SYSTEM", "HACK_MINT_SYSTEM", "HACK_REWRITE_SYSTEM",
                 "RUBRIC_SOLVER_SYSTEM", "HB_FARMER_SYSTEM_PROMPT"):
        assert "TARGET BENCHMARK" not in gs[name], f"brief leaked into {name}"
    # The runtime placeholders the server fills per task are still there.
    for tok in ("[[K]]", "[[USE_CASE]]", "[[SPECIALTY]]", "[[GAP_GUIDANCE]]"):
        assert tok in gs["RUBRIC_PROPOSER_DEFAULT"], tok
    for tok in ("[[N_POSITIVE]]", "[[NEGATIVE_INSTR]]", "[[MODE_INSTR]]", "[[HACK_MEMO]]",
                "[[RECENT_SCORE]]"):
        assert tok in gs["RUBRIC_GENERATOR_DEFAULT"], tok
    # Non-domain content is untouched: counts, point ranges, difficulty target.
    for frag in ("WRITE EXACTLY [[N_POSITIVE]] POSITIVE CRITERIA", "+5..+10", "-5..-10",
                 "Target 0.4-0.6", "90-150 characters"):
        assert frag in gs["RUBRIC_GENERATOR_DEFAULT"], frag
    assert "length term is 0" in gs["HB_SCORE_FORMULA"]
    # Solver system prompt is the bundle's.
    assert "professional" in gs["RUBRIC_SOLVER_SYSTEM"] or "expert" in gs["RUBRIC_SOLVER_SYSTEM"]
    # Agent-loop web-only instruction names the web tool and keeps the budget token.
    al = tree["consts"]["agent_loop"]
    assert "{max_searches}" in al["WEB_ONLY_INSTRUCTION"]
    assert "`web_search`" in al["WEB_ONLY_INSTRUCTION"]
    assert "search_medical_kb" not in al["WEB_ONLY_INSTRUCTION"]
    assert "passage" not in al["WEB_ONLY_HARD_ANSWER_INSTRUCTION"]
    # Referee template keeps its str.format fields.
    for tok in ("{task}", "{answers}", "{labels}", '{{"tiers"'):
        assert tok in tree["consts"]["spec_gap"]["REFEREE_TEMPLATE"], tok


def test_prbench_prompts_are_domain_clean():
    _check_domain("prbench")


def test_profbench_prompts_are_domain_clean():
    _check_domain("profbench")


def test_unknown_domain_is_refused():
    env = dict(os.environ, SE_DOMAIN="bogus")
    env["PYTHONPATH"] = os.pathsep.join(p for p in [str(SE_DIR), env.get("PYTHONPATH")] if p)
    proc = subprocess.run([sys.executable, "-c", "import domains"], cwd=REPO, env=env,
                          capture_output=True, text=True)
    assert proc.returncode != 0
    assert "SE_DOMAIN='bogus' unknown" in proc.stderr


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--dump":
        sys.path.insert(0, str(SE_DIR))
        sys.path.insert(0, str(REPO))
        result = _dump(sys.argv[2], os.environ.get("SE_DOMAIN"))
        sys.stdout.write("\n__DUMP__" + json.dumps(result))
    else:
        sys.exit("usage: python test_domains.py --dump head|tree   (or run under pytest)")
