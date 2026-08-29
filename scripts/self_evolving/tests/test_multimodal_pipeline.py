"""Unit tests for the multimodal minting path.

These cover the parts that fail SILENTLY -- where a bug does not raise, it just
produces a task nobody can answer or a grade nobody can justify:

  * placeholder/image count mismatch (the trainer asserts on it mid-rollout);
  * an image dropped from a chat call, so a role reasons about a picture blind;
  * a manifest naming files this machine cannot see;
  * image paths not surviving into `extra_info`, where the judge reads them from
    (RLHFDataset pops the top-level column).

The live end-to-end mint is a separate script (mint_multimodal_smoke.py) because it
needs the teacher and the staged media.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

SE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(os.path.dirname(SE_DIR))
for _p in (SE_DIR, REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

gs = pytest.importorskip("generation_server")


def _count(conv):
    return sum(str(m.get("content") or "").count("<image>") for m in conv)


# ---------------------------------------------------------------- placeholders
def test_placeholder_added_when_generator_omitted_it():
    conv = [{"role": "user", "content": "What does this study show?"}]
    out = gs._reconcile_image_placeholders(conv, 1)
    assert _count(out) == 1
    assert out[0]["content"].startswith("<image>")


def test_placeholder_count_trimmed_when_generator_overshot():
    conv = [{"role": "user", "content": "<image> and <image> and <image> here"}]
    out = gs._reconcile_image_placeholders(conv, 1)
    assert _count(out) == 1, "extra placeholders must be dropped, not left to crash the trainer"
    assert "here" in out[0]["content"], "surrounding text must survive"


def test_placeholder_multi_image_exact():
    conv = [{"role": "user", "content": "compare <image> with <image>"}]
    out = gs._reconcile_image_placeholders(conv, 2)
    assert _count(out) == 2


def test_placeholder_shortfall_topped_up():
    conv = [{"role": "user", "content": "only one <image> given"}]
    out = gs._reconcile_image_placeholders(conv, 3)
    assert _count(out) == 3


def test_placeholders_never_left_on_assistant_turns():
    conv = [
        {"role": "user", "content": "<image> read this"},
        {"role": "assistant", "content": "I see <image> clearly"},
        {"role": "user", "content": "and now?"},
    ]
    out = gs._reconcile_image_placeholders(conv, 1)
    assert _count(out) == 1
    assistant = [m for m in out if m["role"] == "assistant"][0]
    assert "<image>" not in assistant["content"]


def test_placeholder_inserts_user_turn_if_none_exists():
    conv = [{"role": "assistant", "content": "hello"}]
    out = gs._reconcile_image_placeholders(conv, 1)
    assert _count(out) == 1
    assert any(m["role"] == "user" for m in out)


# ------------------------------------------------------------------- accessors
def test_entry_images_handles_both_shapes():
    assert gs._entry_images({"images": ["/a.jpg"]}) == ["/a.jpg"]
    assert gs._entry_images({"images": [{"image": "/b.jpg"}]}) == ["/b.jpg"]
    assert gs._entry_images({}) == []
    assert gs._entry_images({"images": [123]}) == []


def test_case_images_falls_back_to_the_spec_index():
    """A case posted back by the trainer carries no pixels -- only a question_id."""
    class _S:
        pass
    state = _S()
    state.__dict__["spec_index"] = {"q1": {"images": ["/img/x.jpg"]}}
    assert gs._case_images(state, {"question_id": "q1"}) == ["/img/x.jpg"]
    assert gs._case_images(state, {"question_id": "missing"}) == []
    assert gs._case_images(state, {"images": ["/direct.jpg"]}) == ["/direct.jpg"]


# -------------------------------------------------------------------- manifest
def test_manifest_skips_rows_whose_files_are_absent(tmp_path, monkeypatch):
    """A manifest naming files this machine cannot see must yield NOTHING.

    Otherwise the run mints tasks whose images become black placeholders in the
    trainer, and the arm trains on unanswerable questions that look fine.
    """
    from PIL import Image

    real = tmp_path / "images" / "ok.jpg"
    real.parent.mkdir(parents=True)
    Image.new("RGB", (8, 8), (10, 20, 30)).save(real)

    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"images": ["ok.jpg"], "question": "q", "answer": "a",
                    "modality": "ct"}) + "\n"
        + json.dumps({"images": ["gone.jpg"], "question": "q", "answer": "a",
                      "modality": "ct"}) + "\n")

    monkeypatch.setattr(gs, "HB_MM_SHARE", 1.0)
    monkeypatch.setattr(gs, "HB_MM_MANIFEST", str(manifest))
    monkeypatch.setattr(gs, "HB_MM_ROOT", str(tmp_path / "images"))
    monkeypatch.setattr(gs, "_MM_ROWS", None)

    rows = gs._mm_rows()
    assert len(rows) == 1, "the row naming a missing file must be dropped"
    assert rows[0]["_abs"] == [str(real)]


def test_mm_path_is_off_by_default(monkeypatch):
    monkeypatch.setattr(gs, "HB_MM_SHARE", 0.0)
    monkeypatch.setattr(gs, "_MM_ROWS", None)
    assert gs._mm_rows() == []
    assert gs._mm_draw() is None


# ------------------------------------------------------------------- data URIs
def test_image_data_uri_round_trips(tmp_path):
    import base64

    from PIL import Image

    p = tmp_path / "x.png"
    Image.new("RGB", (32, 16), (200, 100, 50)).save(p)
    uri = gs._image_data_uri(str(p))
    assert uri.startswith("data:image/jpeg;base64,")
    assert len(base64.b64decode(uri.split(",", 1)[1])) > 100


def test_image_data_uri_downscales(tmp_path):
    import base64
    import io as _io

    from PIL import Image

    p = tmp_path / "big.png"
    Image.new("RGB", (2000, 2000), (5, 5, 5)).save(p)
    uri = gs._image_data_uri(str(p), max_pixels=65536)  # 256x256
    raw = base64.b64decode(uri.split(",", 1)[1])
    with Image.open(_io.BytesIO(raw)) as im:
        assert im.size[0] * im.size[1] <= 65536 * 1.05


# ------------------------------------------------- entry carries paths in both
def test_build_entry_rubric_puts_images_in_both_places(tmp_path):
    """`extra_info["images"]` is not redundant: the judge has nothing else."""
    from PIL import Image

    img = tmp_path / "s.jpg"
    Image.new("RGB", (8, 8)).save(img)

    class _S:
        question_counter = 0

    gen = {
        "conversation": [{"role": "user", "content": "What is shown?"}],
        "rubric_items": [{"criterion_text": "names the finding", "points": 5}],
        "use_case": "diagnosis", "specialty": "ct", "difficulty": "typical",
    }
    mm = {"_abs": [str(img)], "modality": "ct", "answer": "PE", "dataset": "rspect"}
    entry = gs._build_entry_rubric(_S(), gen, "", "q", mm)

    assert entry["images"] == [str(img)]
    assert entry["extra_info"]["images"] == [str(img)]
    assert entry["extra_info"]["mm_modality"] == "ct"
    assert sum(m["content"].count("<image>") for m in entry["prompt"]) == 1
    # the recorded finding must never be handed to the solver
    assert "PE" not in json.dumps(entry["prompt"])


def test_build_entry_rubric_text_task_unchanged():
    class _S:
        question_counter = 0

    gen = {
        "conversation": [{"role": "user", "content": "plain text task"}],
        "rubric_items": [{"criterion_text": "c", "points": 5}],
        "use_case": "diagnosis", "specialty": "x", "difficulty": "typical",
    }
    entry = gs._build_entry_rubric(_S(), gen, "", "q", None)
    assert "images" not in entry
    assert "images" not in entry["extra_info"]
    assert "<image>" not in entry["prompt"][0]["content"]


# --------------------------------------------------- chat payload construction
def test_vision_content_is_plain_text_without_images():
    assert gs._vision_content("hello", None) == "hello"
    assert gs._vision_content("hello", []) == "hello"


def test_vision_content_attaches_every_image_then_the_text(tmp_path):
    from PIL import Image

    ps = []
    for i in range(2):
        p = tmp_path / f"i{i}.jpg"
        Image.new("RGB", (8, 8), (i, i, i)).save(p)
        ps.append(str(p))
    parts = gs._vision_content("grade this", ps)
    assert [p["type"] for p in parts] == ["image_url", "image_url", "text"]
    assert parts[-1]["text"] == "grade this"
    assert all(p["image_url"]["url"].startswith("data:image/") for p in parts[:2])


def test_vision_content_raises_on_unreadable_image(tmp_path):
    """Never degrade to a text-only call: that grades a picture nobody saw."""
    with pytest.raises(Exception):
        gs._vision_content("x", [str(tmp_path / "does_not_exist.jpg")])


def test_trainer_judge_vision_content_matches(tmp_path):
    """The judge side has its own copy; it must behave identically."""
    se = pytest.importorskip("verl.utils.reward_score.self_evolving")
    from PIL import Image

    p = tmp_path / "j.jpg"
    Image.new("RGB", (8, 8)).save(p)
    assert se._vision_content("t", None) == "t"
    parts = se._vision_content("t", [str(p)])
    assert [x["type"] for x in parts] == ["image_url", "text"]


def test_row_images_reads_extra_info_not_the_popped_column():
    hp = pytest.importorskip("verl.utils.reward_score.healthbench_pro")
    assert hp._row_images({"images": ["/a.jpg"]}) == ["/a.jpg"]
    assert hp._row_images({"images": [{"image": "/b.jpg"}]}) == ["/b.jpg"]
    assert hp._row_images({}) == []
    assert hp._row_images(None) == []
