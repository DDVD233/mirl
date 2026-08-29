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
        # A judge that cannot see the image scores these two the same.
        from verl.utils.reward_score.healthbench_pro import _row_images
        from verl.utils.reward_score.self_evolving import _call_api

        task_text = "\n".join(str(m.get("content") or "") for m in entry["prompt"])
        judge_imgs = _row_images(entry["extra_info"])
        if judge_imgs != mm["_abs"]:
            _fail("grade", "_row_images did not recover the paths from extra_info")

        async def grade(answer: str) -> int:
            met = 0
            for c in crit[:args.max_criteria]:
                prompt = (f"Conversation:\n{task_text}\n\nAssistant response:\n{answer}\n\n"
                          f"Rubric item: {c.get('criterion_text')}\n\n"
                          f"Did the response meet the rubric item? Reply yes or no.")
                raw = await _call_api(args.judge_base, args.judge_key, args.judge_model,
                                      "You grade one rubric item. Reply yes or no.",
                                      prompt, max_tokens=8, provider=args.judge_provider,
                                      images=judge_imgs)
                if "yes" in (raw or "").strip().lower()[:6]:
                    met += 1
            return met

        good = str(mm.get("answer") or "")
        good_answer = (f"On review of the study, the findings are consistent with "
                       f"{good}. This is the diagnosis and management should follow "
                       f"accordingly.")
        bad_answer = ("The study is entirely normal with no abnormality of any kind. "
                      "No further action is needed.")
        n = min(args.max_criteria, len(crit))
        met_good, met_bad = await asyncio.gather(grade(good_answer), grade(bad_answer))
        print(f"[grade] criteria={n} met(finding-matching)={met_good} met(wrong)={met_bad}")
        if met_good <= met_bad:
            _fail("grade", "the judge did not prefer the finding-matching answer -- it is "
                           "likely grading without the image")

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
    ap.add_argument("--server_args", default="", help="args for the gen server ArgumentParser")
    args = ap.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
