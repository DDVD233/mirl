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
