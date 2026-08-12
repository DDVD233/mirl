# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A retrieved passage must be able to say where it came from.

23.9M of the KB's 57.2M rows are medrag_pubmed abstracts stored with the bibliography
stripped -- no title, author, journal or year field exists in the schema. Probed directly,
the KB returns the ACORN trial's own abstract and Lau et al.'s NEJM conclusion as top hits
for their own questions while the strings "ACORN" and "Lau" appear nowhere. So retrieval
found the right evidence and the policy could not name it, and every rubric criterion
demanding a named source was unreachable at any level of retrieval skill.

attach_titles() joins titles back on entry_id. These tests pin the join, the header format
the summarizer is told to read, and -- most importantly -- that an absent or broken database
degrades to today's behaviour instead of breaking retrieval.
"""

import importlib
import os
import sqlite3
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "scripts", "self_evolving", "kb"))

ACORN_ID = "pubmed23n0859_11771"
ACORN_TITLE = ("Acute Kidney Injury in Sepsis with Piperacillin-Tazobactam versus "
               "Cefepime (ACORN): a randomized clinical trial")


@pytest.fixture
def R(tmp_path, monkeypatch):
    """retrieval.py bound to a throwaway titles DB holding one real row."""
    db = tmp_path / "titles.sqlite"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE titles (id TEXT PRIMARY KEY, title TEXT) WITHOUT ROWID")
    con.execute("INSERT INTO titles VALUES (?,?)", (ACORN_ID, ACORN_TITLE))
    con.commit()
    con.close()
    monkeypatch.setenv("PUBMED_TITLES_DB", str(db))
    mod = importlib.reload(importlib.import_module("retrieval"))
    return mod


def _passages():
    return [
        {"source": "medrag_pubmed", "entry_id": ACORN_ID,
         "text": "To prospectively evaluate the observed incidence of acute kidney injury"},
        {"source": "statpearls", "entry_id": "statpearls_999", "text": "Sepsis overview"},
    ]


def test_the_known_id_gets_its_title(R):
    ps = _passages()
    assert R.attach_titles(ps) == 1
    assert ps[0]["title"] == ACORN_TITLE
    assert "title" not in ps[1], "an unknown id must not gain an empty title"


def test_the_title_lands_in_the_header_not_the_body(R):
    """The summarizer is told 'a passage header of the form title=... IS the citation'.

    It must not be prepended to the passage text, where it would read as a clinical claim
    the passage makes rather than as provenance.
    """
    ps = _passages()
    R.attach_titles(ps)
    block = R.format_passages(ps)
    first = block.split("\n")[0]
    assert first.startswith("[passage 1 | source=medrag_pubmed | title=")
    assert "ACORN" in first
    assert not block.split("\n")[1].startswith(ACORN_TITLE[:20])
    # the untitled passage keeps exactly the old header shape
    assert "[passage 2 | source=statpearls]" in block


def test_the_identifier_is_now_present_at_all(R):
    """The whole point: 'ACORN' must appear somewhere the model can read it."""
    ps = _passages()
    assert "acorn" not in R.format_passages(ps).lower()   # before the join
    R.attach_titles(ps)
    assert "acorn" in R.format_passages(ps).lower()       # after


def test_absent_db_is_a_silent_no_op(tmp_path, monkeypatch):
    """A run predating the build must behave EXACTLY as before, not fail."""
    monkeypatch.setenv("PUBMED_TITLES_DB", str(tmp_path / "does_not_exist.sqlite"))
    mod = importlib.reload(importlib.import_module("retrieval"))
    ps = _passages()
    assert mod.attach_titles(ps) == 0
    assert mod.format_passages(ps).split("\n")[0] == "[passage 1 | source=medrag_pubmed]"


def test_corrupt_db_does_not_break_retrieval(tmp_path, monkeypatch):
    """Retrieval is on the generation critical path: a bad DB costs titles, not rollouts."""
    bad = tmp_path / "corrupt.sqlite"
    bad.write_bytes(b"this is not a database")
    monkeypatch.setenv("PUBMED_TITLES_DB", str(bad))
    mod = importlib.reload(importlib.import_module("retrieval"))
    ps = _passages()
    assert mod.attach_titles(ps) == 0          # must not raise
    assert "passage 1" in mod.format_passages(ps)


def test_empty_and_idless_passages(R):
    assert R.attach_titles([]) == 0
    assert R.attach_titles([{"source": "x", "text": "y"}]) == 0   # no entry_id


def test_lookup_is_one_query_for_the_whole_list(R):
    """8-24 passages per call on the critical path: N queries would be N round trips.

    Traced with set_trace_callback, sqlite3's own hook -- Connection.execute is a read-only
    attribute and cannot be monkeypatched.
    """
    conn = R._titles_conn()
    seen = []
    conn.set_trace_callback(seen.append)
    try:
        R.attach_titles(_passages() * 6)        # 12 passages, 6 distinct ids
    finally:
        conn.set_trace_callback(None)
    selects = [s for s in seen if s.lstrip().upper().startswith("SELECT")]
    assert len(selects) == 1, f"expected 1 batched query, got {len(selects)}: {selects}"
    # set_trace_callback reports the EXPANDED statement, so count the ids, not the
    # placeholders. Each distinct id must appear exactly once: duplicates arrive routinely
    # from multi-query merging and would otherwise bloat the IN clause.
    assert selects[0].count(ACORN_ID) == 1, f"ids should be deduped: {selects[0][:200]}"
    assert selects[0].count("statpearls_999") == 1


# --------------------------------------------------------------------------
# The DB is built in two stages (titles first, then journal/year/author), so the
# code must work against every intermediate schema. A schema older than the code
# must degrade one field at a time -- silently reverting to unlabelled passages
# would undo the whole point while every log still said retrieval was on.
# --------------------------------------------------------------------------
FULL_TITLE = ("Effect of intravenous omeprazole on recurrent bleeding after endoscopic "
              "treatment of bleeding peptic ulcers.")


def _db(tmp_path, *, pmid_col=True, meta=True):
    p = tmp_path / "t.sqlite"
    con = sqlite3.connect(p)
    if pmid_col:
        con.execute("CREATE TABLE titles (id TEXT PRIMARY KEY, title TEXT, pmid TEXT) "
                    "WITHOUT ROWID")
        con.execute("INSERT INTO titles VALUES (?,?,?)",
                    ("pubmed23n0362_9027", FULL_TITLE, "10922420"))
    else:
        con.execute("CREATE TABLE titles (id TEXT PRIMARY KEY, title TEXT) WITHOUT ROWID")
        con.execute("INSERT INTO titles VALUES (?,?)",
                    ("pubmed23n0362_9027", FULL_TITLE))
    if meta:
        con.execute("CREATE TABLE meta (pmid TEXT PRIMARY KEY, journal TEXT, year TEXT, "
                    "author TEXT, n_authors TEXT) WITHOUT ROWID")
        con.execute("INSERT INTO meta VALUES (?,?,?,?,?)",
                    ("10922420", "N Engl J Med", "2000", "Lau JY", "12"))
    con.commit()
    con.close()
    return p


def _load(tmp_path, monkeypatch, **kw):
    monkeypatch.setenv("PUBMED_TITLES_DB", str(_db(tmp_path, **kw)))
    return importlib.reload(importlib.import_module("retrieval"))


def _one():
    return [{"source": "medrag_pubmed", "entry_id": "pubmed23n0362_9027", "text": "body"}]


def test_full_schema_yields_the_reference_a_rubric_asks_for(tmp_path, monkeypatch):
    """"the 2000 NEJM trial by Lau et al." is the actual criterion text. This is it."""
    R = _load(tmp_path, monkeypatch)
    ps = _one()
    assert R.attach_titles(ps) == 1
    assert ps[0]["citation"] == "Lau JY et al., N Engl J Med 2000"
    head = R.format_passages(ps).splitlines()[0]
    for want in ("title=", "cite=Lau JY et al., N Engl J Med 2000", "pmid=10922420"):
        assert want in head, head


def test_titles_and_pmid_but_no_meta_still_labels(tmp_path, monkeypatch):
    """The state between the two build jobs: title + PMID, no reference."""
    R = _load(tmp_path, monkeypatch, meta=False)
    ps = _one()
    assert R.attach_titles(ps) == 1
    assert ps[0]["title"] == FULL_TITLE
    assert ps[0].get("pmid") == "10922420"
    assert "citation" not in ps[0]
    assert "cite=" not in R.format_passages(ps)


def test_oldest_schema_titles_only_still_labels(tmp_path, monkeypatch):
    """A DB predating the pmid column must still deliver titles, not nothing."""
    R = _load(tmp_path, monkeypatch, pmid_col=False, meta=False)
    ps = _one()
    assert R.attach_titles(ps) == 1
    assert ps[0]["title"] == FULL_TITLE
    assert "pmid" not in ps[0] and "citation" not in ps[0]


def test_title_present_but_metadata_missing_is_a_left_join(tmp_path, monkeypatch):
    """A PMID absent from meta must not drop the title -- inner join would lose it."""
    p = tmp_path / "t.sqlite"
    con = sqlite3.connect(p)
    con.execute("CREATE TABLE titles (id TEXT PRIMARY KEY, title TEXT, pmid TEXT) "
                "WITHOUT ROWID")
    con.execute("INSERT INTO titles VALUES (?,?,?)", ("x_1", FULL_TITLE, "999999"))
    con.execute("CREATE TABLE meta (pmid TEXT PRIMARY KEY, journal TEXT, year TEXT, "
                "author TEXT, n_authors TEXT) WITHOUT ROWID")
    con.commit(); con.close()
    monkeypatch.setenv("PUBMED_TITLES_DB", str(p))
    R = importlib.reload(importlib.import_module("retrieval"))
    ps = [{"source": "medrag_pubmed", "entry_id": "x_1", "text": "b"}]
    assert R.attach_titles(ps) == 1
    assert ps[0]["title"] == FULL_TITLE
    assert "citation" not in ps[0]


@pytest.mark.parametrize("author,n,journal,year,expected", [
    ("Lau JY", "5", "N Engl J Med", "2000", "Lau JY et al., N Engl J Med 2000"),
    ("Smith J", "1", "BMJ", "1998", "Smith J, BMJ 1998"),      # no invented co-authors
    (None, None, "Lancet", "2011", "Lancet 2011"),
    ("Jones A", "3", None, None, "Jones A et al."),
    (None, None, None, None, ""),
    ("Lau JY", "notanumber", "NEJM", "2000", "Lau JY, NEJM 2000"),   # bad n_authors
])
def test_citation_assembly(author, n, journal, year, expected):
    import retrieval as R
    assert R._citation(author, n, journal, year) == expected


def test_it_works_with_NO_env_var_set():
    """Citations must not depend on an env var being passed at launch.

    retrieval.py's default has to be the same path both build scripts publish to, or a run
    launched without PUBMED_TITLES_DB silently serves unlabelled passages while every commit
    message says citations are on. Checked against the builders' own argparse defaults
    rather than a copy of the string.
    """
    import ast
    import importlib
    import pathlib

    for var in ("PUBMED_TITLES_DB",):
        os.environ.pop(var, None)
    R = importlib.reload(importlib.import_module("retrieval"))
    default = R._TITLES_DB
    assert default.endswith("pubmed_titles.sqlite"), default

    kb = pathlib.Path(R.__file__).parent
    published = set()
    for script in ("build_pubmed_titles.py", "build_pubmed_meta.py"):
        tree = ast.parse((kb / script).read_text())
        for node in ast.walk(tree):
            # ap.add_argument("--out", default="...")
            if (isinstance(node, ast.Call)
                    and getattr(node.func, "attr", "") == "add_argument"
                    and node.args and getattr(node.args[0], "value", "") == "--out"):
                for kw in node.keywords:
                    if kw.arg == "default":
                        published.add(ast.literal_eval(kw.value))
    assert published, "could not find the builders' --out defaults"
    assert published == {default}, (
        f"retrieval.py defaults to {default!r} but the builders publish to {published!r}; "
        "a run without the env var would find no database")


# --------------------------------------------------------------------------
# The source list is APPENDED IN CODE, never requested from the summarizer.
# Measured reason: the same prompt on the same passages produced a Sources
# section on one sample and omitted it entirely on the next. A citation list that
# appears at the model's discretion is worse than none, because the metric would
# then move with sampling noise instead of with what the KB knows.
# --------------------------------------------------------------------------
def test_sources_block_is_deterministic_and_complete():
    import retrieval as R
    ps = [
        {"source": "statpearls", "text": "x"},                                  # unlabelled
        {"source": "medrag_pubmed", "text": "y", "title": "A title", "pmid": "111"},
        {"source": "medrag_pubmed", "text": "z", "title": "B title", "pmid": "222",
         "citation": "Lau JY et al., N Engl J Med 2000"},
    ]
    out = R.sources_block(ps)
    assert out.startswith("Sources:\n")
    # Numbering must match the passage positions the brief cites, so [p2]/[p3] not [p1]/[p2].
    assert "[p2] A title (PMID 111)" in out
    assert "[p3] Lau JY et al., N Engl J Med 2000 -- B title (PMID 222)" in out
    assert "[p1]" not in out, "an unlabelled passage must not appear"
    assert R.sources_block(ps) == out, "must be deterministic"


def test_sources_block_empty_when_nothing_is_labelled():
    """No labels means no section -- not an empty 'Sources:' header."""
    import retrieval as R
    assert R.sources_block([{"source": "statpearls", "text": "x"}]) == ""
    assert R.sources_block([]) == ""


def test_the_summarizer_is_told_NOT_to_write_its_own_sources():
    """Two source lists that disagree is worse than one. The code owns this output."""
    import ast
    import pathlib
    gs = pathlib.Path(__file__).parents[2] / "scripts" / "self_evolving" / "generation_server.py"
    tree = ast.parse(gs.read_text())
    prompt = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", "") == "SUMMARY_SYSTEM" for t in node.targets):
            prompt = ast.literal_eval(node.value)
    assert prompt, "SUMMARY_SYSTEM not found"
    assert "Do NOT write a 'Sources:' section yourself" in prompt
    assert "Never invent a reference" in prompt


# --------------------------------------------------------------------------
# Web evidence must never stall a rollout. The budget is a HARD ceiling covering
# queue wait plus the call: with the semaphore acquired inside the deadline, a
# request arriving while every slot is busy would wait an unbounded time before its
# own timeout started -- the exact "rate limited, so we hang" failure.
# --------------------------------------------------------------------------
def test_web_timeout_covers_queue_wait_not_just_the_call():
    import asyncio
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                     "scripts", "self_evolving", "kb"))
    import web_evidence as W

    async def go():
        we = W.WebEvidence(api_base="http://x/v1", api_key="k", model="m",
                           concurrency=1, use_cache=False, timeout_s=1)

        async def slow(_q):
            await asyncio.sleep(30)
            return "", [], {}, 0

        we._call = slow
        t0 = asyncio.get_running_loop().time()
        r1, r2 = await asyncio.gather(we.search("a"), we.search("b"))
        return asyncio.get_running_loop().time() - t0, r1, r2

    elapsed, r1, r2 = asyncio.run(go())
    # The queued call must give up at ITS deadline, not after the first call finishes.
    assert elapsed < 5, f"queued call waited {elapsed:.1f}s; bound excludes queue time"
    for r in (r1, r2):
        assert "Timeout" in (r.error or ""), r.error
        assert not r.text and not r.sources, "a timeout must yield no evidence, not partial"


def test_web_failure_yields_no_evidence_so_the_caller_keeps_milvus():
    """A failure must be reported as absence, never as an exception or a partial brief."""
    import asyncio
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                     "scripts", "self_evolving", "kb"))
    import web_evidence as W

    async def go():
        we = W.WebEvidence(api_base="http://x/v1", api_key="k", model="m",
                           use_cache=False, timeout_s=5)

        async def boom(_q):
            raise RuntimeError("429 rate limited")

        we._call = boom
        return await we.search("q")

    r = asyncio.run(go())
    assert r.error and "RuntimeError" in r.error
    assert r.text == "" and r.sources == []


def test_breaker_opens_and_then_skips_without_calling():
    """After repeated failures the tool is skipped outright, so a rate-limit storm cannot
    keep costing 60s per retrieval."""
    import asyncio
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                     "scripts", "self_evolving", "kb"))
    import web_evidence as W

    async def go():
        we = W.WebEvidence(api_base="http://x/v1", api_key="k", model="m",
                           use_cache=False, timeout_s=5)
        we._breaker_trip = 2
        calls = []

        async def boom(_q):
            calls.append(1)
            raise RuntimeError("429")

        we._call = boom
        for _ in range(4):
            await we.search("q")
        return len(calls), we.breaker_skips

    n_calls, skips = asyncio.run(go())
    assert n_calls == 2, f"breaker should stop calling after 2 failures, made {n_calls}"
    assert skips >= 1


def test_a_refusal_is_not_a_breaker_failure():
    """A drafting task has nothing to look up. That is a RESULT, not a service failure.

    Counting refusals against the breaker opened it after five writing tasks in a row, and
    the 43 queries that then failed with "breaker open" each counted as another failure and
    kept it open -- 178 failures against 14 successes. The breaker is for rate limits and
    outages; it must never react to the content of a request.
    """
    import asyncio
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                     "scripts", "self_evolving", "kb"))
    import web_evidence as W

    async def go():
        we = W.WebEvidence(api_base="http://x/v1", api_key="k", model="m",
                           use_cache=False, timeout_s=5)
        we._breaker_trip = 2
        calls = []

        async def refuse(_q):
            calls.append(1)
            raise W.NotALookup("no web_search_call: nothing was looked up")

        we._call = refuse
        results = [await we.search(f"draft a note {i}") for i in range(6)]
        return len(calls), we.breaker_skips, we.failures, we.refusals, results

    n_calls, skips, failures, refusals, results = asyncio.run(go())
    assert n_calls == 6, f"breaker must stay closed through refusals, only {n_calls} calls"
    assert skips == 0 and failures == 0, (skips, failures)
    assert refusals == 6
    for r in results:
        assert r.refused is True and r.text == "" and r.sources == []


def test_a_real_failure_still_trips_the_breaker():
    """The protection must survive the fix above."""
    import asyncio
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                     "scripts", "self_evolving", "kb"))
    import web_evidence as W

    async def go():
        we = W.WebEvidence(api_base="http://x/v1", api_key="k", model="m",
                           use_cache=False, timeout_s=5)
        we._breaker_trip = 2
        calls = []

        async def boom(_q):
            calls.append(1)
            raise RuntimeError("429 rate limited")

        we._call = boom
        for _ in range(5):
            await we.search("what is the eGFR cutoff for metformin")
        return len(calls), we.breaker_skips

    n_calls, skips = asyncio.run(go())
    assert n_calls == 2, f"expected the breaker to stop calling after 2, got {n_calls}"
    assert skips >= 1


def test_a_refusal_after_failures_resets_the_streak():
    """A working service that merely has nothing to look up is evidence it is UP."""
    import asyncio
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                     "scripts", "self_evolving", "kb"))
    import web_evidence as W

    async def go():
        we = W.WebEvidence(api_base="http://x/v1", api_key="k", model="m",
                           use_cache=False, timeout_s=5)
        we._breaker_trip = 3
        mode = {"v": "boom"}

        async def maybe(_q):
            if mode["v"] == "boom":
                raise RuntimeError("timeout")
            raise W.NotALookup("draft")

        we._call = maybe
        await we.search("a")
        await we.search("b")          # 2 failures, one short of tripping
        mode["v"] = "refuse"
        await we.search("c")          # refusal -> streak reset
        mode["v"] = "boom"
        await we.search("d")          # 1 failure again, must NOT trip
        return we.breaker_skips, we._fails

    skips, fails = asyncio.run(go())
    assert skips == 0, "breaker opened despite the streak being broken by a refusal"
    assert fails == 1


# --------------------------------------------------------------------------
# The appended Sources block was built only from PubMed-URL annotations, which
# covered ~26% of entries -- while 98% named an organisation, journal or PMID in
# the model's own SOURCE: line. A guideline or an FDA label is fully citable and has
# no PMID, so reading the blocks is what makes the list reflect what was retrieved.
# --------------------------------------------------------------------------
BLOCKS = """SOURCE: Qian et al., JAMA, 2023 (PMID 37837651)

TITLE: Cefepime vs Piperacillin-Tazobactam in Adults Hospitalized With Acute Infection

STATES: no significant difference in AKI.

SOURCE: American College of Gastroenterology, 2022

TITLE: ACG Clinical Guideline for the Diagnosis and Management of GERD

STATES: 8-week empiric PPI trial for typical symptoms.
"""


def _we():
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                     "scripts", "self_evolving", "kb"))
    import web_evidence as W
    return W, W.WebEvidence(api_base="http://x/v1", api_key="k", model="m", use_cache=False)


def test_source_blocks_parse_pubmed_and_guideline_alike():
    _W, we = _we()
    got = we._sources_from_text(BLOCKS)
    assert len(got) == 2, [s.render() for s in got]
    paper, guideline = got
    assert paper.pmid == "37837651"
    assert "PMID 37837651" in paper.render()
    # The PMID must not be duplicated inside the citation text.
    assert paper.cite_text == "Qian et al., JAMA, 2023"
    # A guideline is citable with no PMID at all -- the case the annotation path missed.
    assert guideline.pmid == ""
    assert "American College of Gastroenterology, 2022" in guideline.render()
    assert "ACG Clinical Guideline" in guideline.render()
    assert "PMID" not in guideline.render()


def test_resolved_metadata_wins_over_the_models_wording():
    """Verified fields beat self-reported ones; cite_text is the fallback, not the default."""
    _W, _inst = _we()
    s = _W.WebSource(title="T", pmid="10922420", cite_text="Somebody, Some Journal, 1999",
                     author="Lau JY", n_authors="12", journal="N Engl J Med", year="2000")
    assert s.citation() == "Lau JY et al., N Engl J Med 2000"
    s2 = _W.WebSource(title="T", cite_text="American College of Gastroenterology, 2022")
    assert s2.citation() == "American College of Gastroenterology, 2022"


def test_no_blocks_means_no_sources_not_a_crash():
    _W, we = _we()   # noqa: F841 -- `we` is used below
    assert we._sources_from_text("") == []
    assert we._sources_from_text("NO SOURCES FOUND") == []
    assert we._sources_from_text("some prose with no блок structure") == []
