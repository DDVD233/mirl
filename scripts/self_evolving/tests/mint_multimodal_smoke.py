"""LIVE end-to-end proof that a multimodal task can be minted, served and graded.

The unit tests cover the shapes; this covers the thing that actually matters and
that no mock can establish: that a real teacher, looking at a real staged image,
mints a task whose rubric depends on the image, that the trainer's dataset turns
that task into pixel tensors, and that the judge grades it WITH the image.

Four stages, each a hard gate:

  1. MINT     -- proposer + rubric generator, both with the image attached.
  2. ENTRY    -- exactly one <image> per image, paths in both places, and the
                 recorded finding never leaked into what the solver sees.
  3. INGEST   -- the real RLHFDataset path (the AgentLoop contract: raw_prompt ->
                 process_vision_info -> processor) yields real pixels, not the
                 black placeholder that a broken path silently substitutes.
  4. GRADE    -- the judge scores a deliberately WRONG answer low and a
                 finding-matching answer higher. Grading text-blind would score
                 them the same, so this is what proves the judge really looked.

Run on a pod, from the repo root:
    CHAT_PROVIDER=trapi python scripts/self_evolving/tests/mint_multimodal_smoke.py \
        --manifest /scratch/sheng/self_evolving/mm_media/manifest.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

SE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(os.path.dirname(SE_DIR))
for _p in (SE_DIR, REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _fail(stage: str, msg: str) -> None:
    print(f"\nFAIL [{stage}]: {msg}")
    raise SystemExit(1)


async def _run(args) -> int:
    os.environ.setdefault("HB_MM_SHARE", "1.0")
    os.environ["HB_MM_MANIFEST"] = args.manifest
    if args.images_root:
        os.environ["HB_MM_ROOT"] = args.images_root

    import generation_server as gs

    gs.HB_MM_SHARE = 1.0
    gs.HB_MM_MANIFEST = args.manifest
    gs.HB_MM_ROOT = args.images_root or os.path.join(os.path.dirname(args.manifest), "images")
    gs._MM_ROWS = None

    rows = gs._mm_rows()
    print(f"[manifest] {len(rows)} usable rows")
    if not rows:
        _fail("manifest", "no usable rows -- staged media missing on this machine")

    # ---------------------------------------------------------------- 1. MINT
    # A real args namespace + a real http client, but no server: this exercises the
    # minting functions directly rather than standing up FastAPI and workers.
    import httpx

    srv_args = gs.build_arg_parser().parse_args(
        (args.server_args.split() if args.server_args else []) + ["--rubric_mode"])
    if not srv_args.prompt_dir:
        srv_args.prompt_dir = os.path.join(args.tmp_dir, "prompts")
    os.makedirs(srv_args.prompt_dir, exist_ok=True)
    state = gs.ServerState(srv_args, asyncio.get_running_loop())
    state.http_client = httpx.AsyncClient(
        limits=httpx.Limits(max_connections=32, max_keepalive_connections=16))
    try:
        mm = gs._mm_draw()
        print(f"[mint] modality={mm['modality']} dataset={mm.get('dataset')} "
              f"image={mm['_abs'][0]}")
        print(f"[mint] recorded finding: {str(mm.get('answer'))[:120]!r}")

        requests = await gs.agent_task_proposer(
            state, use_case=args.use_case, specialty=mm["modality"], mm=mm)
        if not requests:
            _fail("mint", "proposer returned nothing")
        request = requests[0]
        print(f"[mint] request: {request[:220]!r}")

        gen = await gs.agent_task_rubric_generator(
            state, request, args.use_case, mm["modality"], "", "good_faith", mm)
        crit = gen.get("rubric_items") or []
        print(f"[mint] rubric: {len(crit)} criteria")
        for c in crit[:4]:
            print(f"    [{c.get('points')}] {str(c.get('criterion_text'))[:130]}")
        if not crit:
            _fail("mint", "generator produced no rubric")

        # ------------------------------------------------------------ 2. ENTRY
        entry = gs._build_entry_rubric(state, gen, "", request, mm)
        n_ph = sum(str(m.get("content") or "").count("<image>") for m in entry["prompt"])
        if n_ph != len(mm["_abs"]):
            _fail("entry", f"{n_ph} placeholders for {len(mm['_abs'])} images")
        if entry.get("images") != mm["_abs"]:
            _fail("entry", "top-level images column wrong")
        if entry["extra_info"].get("images") != mm["_abs"]:
            _fail("entry", "extra_info.images missing -- the judge would grade blind")
        solver_sees = json.dumps(entry["prompt"]).lower()
        ans = str(mm.get("answer") or "").strip().lower()
        leaked = [tok for tok in ans.replace(",", " ").split()
                  if len(tok) > 4 and tok in solver_sees]
        print(f"[entry] placeholders={n_ph} images={len(entry['images'])} "
              f"leaked_finding_tokens={leaked}")
        if leaked and not args.allow_leak:
            _fail("entry", f"the recorded finding leaked into the solver prompt: {leaked}")

        # ----------------------------------------------------------- 3. INGEST
        import pandas as pd
        from omegaconf import OmegaConf

        from verl.utils import hf_processor, hf_tokenizer
        from verl.utils.dataset.rl_dataset import RLHFDataset

        pq = os.path.join(args.tmp_dir, "mm_mint_smoke.parquet")
        os.makedirs(args.tmp_dir, exist_ok=True)
        pd.DataFrame([{
            "data_source": entry["data_source"],
            "prompt": entry["prompt"],
            "images": entry["images"],
            "reward_model": entry["reward_model"],
            "extra_info": entry["extra_info"],
        }]).to_parquet(pq, index=False)

        tok = hf_tokenizer(args.model, trust_remote_code=True)
        proc = hf_processor(args.model, trust_remote_code=True, use_fast=True)
        cfg = OmegaConf.create({
            "prompt_key": "prompt", "image_key": "images", "video_key": "videos",
            "max_prompt_length": 16384, "truncation": "left", "return_raw_chat": True,
            "filter_overlong_prompts": False,
            "apply_chat_template_kwargs": {"enable_thinking": True},
        })
        ds = RLHFDataset(data_files=pq, tokenizer=tok, config=cfg, processor=proc)
        item = ds[0]
        images, _ = await RLHFDataset.process_vision_info(
            item["raw_prompt"], ds.image_patch_size, cfg)
        images = images or []
        if len(images) != len(mm["_abs"]):
            _fail("ingest", f"process_vision_info returned {len(images)} images")
        black = [i for i, im in enumerate(images)
                 if im.size == (224, 224) and not im.convert("RGB").getbbox()]
        if black:
            _fail("ingest", f"images {black} are BLACK PLACEHOLDERS (unreadable file)")
        raw_text = proc.apply_chat_template(item["raw_prompt"], add_generation_prompt=True,
                                            tokenize=False)
        inputs = proc(text=[raw_text], images=images, return_tensors="pt")
        ids = inputs["input_ids"][0]
        pad_id = tok.convert_tokens_to_ids("<|image_pad|>")
        n_img_tok = int((ids == pad_id).sum()) if pad_id is not None else -1
        print(f"[ingest] sizes={[im.size for im in images]} prompt_tokens={len(ids)} "
              f"image_tokens={n_img_tok}")
        if n_img_tok == 0:
            _fail("ingest", "image tokens never expanded into input_ids")

        # ------------------------------------------------------------ 4. GRADE
        # Two questions, deliberately separated:
        #
        #  4a. IS THE JUDGE ACTUALLY LOOKING? Grade the SAME criterion and the SAME
        #      answer against two synthetic images that differ only in colour. The
        #      image is then the ONLY variable, so a verdict that flips proves the
        #      pixels reached the judge, and a verdict that does not flip proves they
        #      did not. Grading a real medical rubric cannot establish this: a
        #      text-blind judge can often guess a plausible verdict from the prose.
        #
        #  4b. Does the real minted rubric grade sanely WITH the image attached?
        from PIL import Image

        from verl.utils.reward_score.healthbench_pro import _row_images
        from verl.utils.reward_score.self_evolving import _call_api

        task_text = "\n".join(str(m.get("content") or "") for m in entry["prompt"])
        judge_imgs = _row_images(entry["extra_info"])
        if judge_imgs != mm["_abs"]:
            _fail("grade", "_row_images did not recover the paths from extra_info")

        async def judge_one(criterion: str, answer: str, images: list, task: str) -> bool | None:
            prompt = (f"Conversation:\n{task}\n\nAssistant response:\n{answer}\n\n"
                      f"Rubric item: {criterion}\n\n"
                      f"Did the response meet the rubric item? Reply yes or no.")
            raw = await _call_api(args.judge_base, args.judge_key, args.judge_model,
                                  "You grade one rubric item against the conversation "
                                  "and any attached image. Reply yes or no.",
                                  prompt, max_tokens=8, provider=args.judge_provider,
                                  images=images)
            t = (raw or "").strip().lower()
            if args.debug_judge:
                print(f"    [judge raw] {raw!r}")
            if t.startswith("yes") or " yes" in t[:40]:
                return True
            if t.startswith("no") or " no" in t[:40]:
                return False
            return None

        colours = {"red": (220, 20, 20), "blue": (20, 20, 220)}
        probe_paths = {}
        for name, rgb in colours.items():
            fp = os.path.join(args.tmp_dir, f"probe_{name}.jpg")
            Image.new("RGB", (256, 256), rgb).save(fp)
            probe_paths[name] = fp

        probe_task = "<image>\nA user has attached an image and asks about it."
        probe_criterion = "The response correctly states that the attached image is predominantly RED."
        probe_answer = "The attached image is predominantly red."
        v_red, v_blue = await asyncio.gather(
            judge_one(probe_criterion, probe_answer, [probe_paths["red"]], probe_task),
            judge_one(probe_criterion, probe_answer, [probe_paths["blue"]], probe_task),
        )
        print(f"[grade 4a] same answer+criterion, red image -> met={v_red}; "
              f"blue image -> met={v_blue}")
        if v_red is not True or v_blue is not False:
            _fail("grade", "the judge's verdict did not track the IMAGE (expected "
                           f"met=True on red and met=False on blue, got {v_red}/{v_blue}) "
                           "-- it is not really seeing the pixels")

        # 4b. the real rubric, with the image, on an answer built FROM the criteria
        # versus one that calls the study normal.
        ideal = " ".join(str(c.get("criterion_text") or "") for c in crit[:args.max_criteria])
        ideal_answer = ("Reviewing the attached study: " + ideal.replace("Mentions in some way that", "")
                        .replace("mentions in some way that", ""))
        null_answer = ("The attached study is entirely normal. There is no abnormality, "
                       "no further imaging is required and no follow-up is needed.")
        met_ideal, met_null = 0, 0
        for c in crit[:args.max_criteria]:
            ctext = str(c.get("criterion_text") or "")
            a, b = await asyncio.gather(
                judge_one(ctext, ideal_answer, judge_imgs, task_text),
                judge_one(ctext, null_answer, judge_imgs, task_text),
            )
            met_ideal += 1 if a else 0
            met_null += 1 if b else 0
        n = min(args.max_criteria, len(crit))
        print(f"[grade 4b] criteria={n} met(rubric-derived)={met_ideal} met(normal-study)={met_null}")
        if met_ideal <= met_null:
            _fail("grade", "the real rubric did not separate a rubric-derived answer from "
                           "a 'study is normal' answer")

        print("\nPASS: minted -> served -> ingested -> graded, with the image at every step")
        return 0
    finally:
        await state.http_client.aclose()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="/scratch/sheng/self_evolving/mm_media/manifest.jsonl")
    ap.add_argument("--images_root", default="")
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--use_case", default="diagnosis")
    ap.add_argument("--tmp_dir", default="/scratch/sheng/self_evolving/tmp_mm_smoke")
    ap.add_argument("--max_criteria", type=int, default=3)
    ap.add_argument("--judge_base", default=os.environ.get("SMOKE_JUDGE_BASE", ""))
    ap.add_argument("--judge_key", default=os.environ.get("SMOKE_JUDGE_KEY", "EMPTY"))
    ap.add_argument("--judge_model", default=os.environ.get("SMOKE_JUDGE_MODEL", ""))
    ap.add_argument("--judge_provider", default=os.environ.get("SMOKE_JUDGE_PROVIDER", "trapi"))
    ap.add_argument("--allow_leak", action="store_true")
    ap.add_argument("--debug_judge", action="store_true", help="print raw judge replies")
    ap.add_argument("--server_args", default="", help="args for the gen server ArgumentParser")
    args = ap.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
