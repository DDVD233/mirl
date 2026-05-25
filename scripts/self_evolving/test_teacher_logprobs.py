"""Smoke-test the external OPD teacher's /v1/completions endpoint.

Mirrors EXACTLY what verl.experimental.teacher_loop.external_client.py
sends to the teacher when on-policy distillation is enabled:
    POST {base_url}/completions
      model: <teacher model id>
      prompt: list[int]   # token ids
      max_tokens: 1
      temperature: 1.0
      prompt_logprobs: K  # 0 for k1 (use_topk=False), topk otherwise

Run this BEFORE bringing up gen/trainer for an OPD experiment so OOMs or
shape mismatches surface here (where retrying costs nothing) instead of
mid-step in the ray cluster.

Pipeline:
  Phase 1: single small request (server-alive smoke)
  Phase 2: one request per increasing prompt length (catches per-length
           prefill OOM)
  Phase 3: concurrent burst at the worst-case length matching
           VERL_EXTERNAL_TEACHER_CONN_LIMIT (the burst that hit OOM
           on the last attempt)

Token IDs are random integers in [1000, vocab_size); vLLM treats them as
opaque ids during prompt_logprobs (the log-likelihood numbers are
meaningless, but the engine still runs the full prefill + LM head on
them, which is exactly what we want to test).

Usage:
  python scripts/self_evolving/test_teacher_logprobs.py \\
      --url http://vps3.dd.works:18005/v1 \\
      --model Qwen/Qwen3.5-397B-A17B-FP8
  # tighter: --burst 16 --lengths 512 4096 8192
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from typing import Any

import aiohttp


DEFAULT_URL = os.environ.get("TEACHER_URL", "http://vps3.dd.works:18005/v1")
DEFAULT_MODEL = os.environ.get("TEACHER_MODEL", "Qwen/Qwen3.5-397B-A17B-FP8")
DEFAULT_API_KEY = os.environ.get("TEACHER_API_KEY", "EMPTY")
# Defaults match verl.experimental.teacher_loop.external_client
DEFAULT_TIMEOUT = float(os.environ.get("VERL_EXTERNAL_TEACHER_TIMEOUT", "600"))
DEFAULT_CONN_LIMIT = int(os.environ.get("VERL_EXTERNAL_TEACHER_CONN_LIMIT", "32"))


def _make_prompt(length: int, vocab_size: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    return [rng.randint(1000, vocab_size - 1) for _ in range(length)]


async def _one_request(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    api_key: str,
    prompt_ids: list[int],
    prompt_logprobs: int,
    request_id: str,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompt_ids,
        "max_tokens": 1,
        "temperature": 1.0,
        "prompt_logprobs": prompt_logprobs,
    }
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    t0 = time.time()
    try:
        async with session.post(
            f"{url.rstrip('/')}/completions",
            json=payload,
            headers=headers,
        ) as resp:
            text = await resp.text()
            elapsed = time.time() - t0
            if resp.status != 200:
                return {
                    "id": request_id,
                    "ok": False,
                    "status": resp.status,
                    "elapsed": elapsed,
                    "err": text[:300],
                }
            data = json.loads(text)
            pl = data["choices"][0].get("prompt_logprobs")
            if pl is None:
                return {
                    "id": request_id,
                    "ok": False,
                    "status": 200,
                    "elapsed": elapsed,
                    "err": "server returned no prompt_logprobs field",
                }
            return {
                "id": request_id,
                "ok": True,
                "elapsed": elapsed,
                "prompt_len": len(prompt_ids),
                "lp_len": len(pl),
                "first_lp_is_none": pl[0] is None,
                "k_per_position": len(pl[1]) if pl[1] is not None else 0,
            }
    except Exception as e:
        return {
            "id": request_id,
            "ok": False,
            "elapsed": time.time() - t0,
            "err": f"{type(e).__name__}: {e}",
        }


async def _probe_vocab_size(session: aiohttp.ClientSession, url: str, api_key: str) -> int:
    # vLLM /v1/models returns max_model_len but not vocab_size. The Qwen3.5
    # tokenizer has 152064 tokens; default to that and let caller override.
    return int(os.environ.get("TEACHER_VOCAB_SIZE", "152064"))


def _fmt(r: dict) -> str:
    if r["ok"]:
        return (
            f"OK  id={r['id']:<14} prompt={r['prompt_len']:>6}  "
            f"lp_len={r['lp_len']:>6}  k_per_pos={r['k_per_position']:>3}  "
            f"first_lp_None={r['first_lp_is_none']}  elapsed={r['elapsed']:.2f}s"
        )
    return (
        f"FAIL id={r['id']:<14} elapsed={r['elapsed']:.2f}s  "
        f"status={r.get('status', '-')}  err={r.get('err', '')[:220]}"
    )


async def main_async(args: argparse.Namespace) -> int:
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    connector = aiohttp.TCPConnector(
        limit=max(args.burst, args.concurrency),
        force_close=True,
        enable_cleanup_closed=True,
    )

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        vocab = await _probe_vocab_size(session, args.url, args.api_key)
        print(f"Teacher: {args.url}  model={args.model}  vocab_size={vocab}", flush=True)
        print(
            f"Config:  prompt_logprobs={args.prompt_logprobs}  "
            f"lengths={args.lengths}  burst={args.burst}  timeout={args.timeout}s",
            flush=True,
        )

        # ---- Phase 1: smallest single ----
        print("\n=== phase 1: single small request ===", flush=True)
        smallest = min(args.lengths)
        prompt = _make_prompt(smallest, vocab, seed=0)
        r = await _one_request(
            session, args.url, args.model, args.api_key,
            prompt, args.prompt_logprobs, "smoke-0",
        )
        print("  " + _fmt(r), flush=True)
        if not r["ok"]:
            print("Smoke test failed — teacher is not serving prompt_logprobs requests.", flush=True)
            return 1

        # ---- Phase 2: one per length, sequential ----
        print("\n=== phase 2: per-length sequential ===", flush=True)
        per_length_ok = True
        for L in args.lengths:
            prompt = _make_prompt(L, vocab, seed=L)
            r = await _one_request(
                session, args.url, args.model, args.api_key,
                prompt, args.prompt_logprobs, f"len-{L}",
            )
            print("  " + _fmt(r), flush=True)
            if not r["ok"]:
                per_length_ok = False
        if not per_length_ok:
            print("\nOne or more single-prompt lengths failed — stop here.", flush=True)
            return 2

        # ---- Phase 3: concurrent burst at worst-case length ----
        worst = max(args.lengths)
        print(
            f"\n=== phase 3: burst of {args.burst} concurrent requests at length {worst} ===",
            flush=True,
        )
        t0 = time.time()
        tasks = [
            _one_request(
                session, args.url, args.model, args.api_key,
                _make_prompt(worst, vocab, seed=10_000 + i),
                args.prompt_logprobs, f"burst-{i:03d}",
            )
            for i in range(args.burst)
        ]
        results = await asyncio.gather(*tasks)
        wall = time.time() - t0
        ok_results = [r for r in results if r["ok"]]
        fail_results = [r for r in results if not r["ok"]]
        latencies = sorted(r["elapsed"] for r in results)
        print(
            f"  total={args.burst}  ok={len(ok_results)}  fail={len(fail_results)}  wall={wall:.1f}s",
            flush=True,
        )
        if latencies:
            print(
                f"  per-req latency:  min={latencies[0]:.1f}s  "
                f"p50={latencies[len(latencies) // 2]:.1f}s  "
                f"p95={latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))]:.1f}s  "
                f"max={latencies[-1]:.1f}s",
                flush=True,
            )
        for r in fail_results[:10]:
            print("  " + _fmt(r), flush=True)
        if fail_results:
            return 3
        print("\nAll phases passed — teacher endpoint is OPD-ready.", flush=True)
        return 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--api_key", default=DEFAULT_API_KEY)
    p.add_argument(
        "--prompt_logprobs", type=int, default=1,
        help="K for prompt_logprobs. Match the trainer's DISTILL_TOPK: "
             "k1 loss + use_topk=False sends 0; forward_kl_topk sends DISTILL_TOPK.",
    )
    p.add_argument(
        "--lengths", type=int, nargs="+",
        default=[512, 4096, 8192, 12288],
        help="Per-prompt token lengths to test. Default covers the trainer's "
             "max_prompt_length (8192) + max_response_length (4096) envelope.",
    )
    p.add_argument(
        "--burst", type=int, default=DEFAULT_CONN_LIMIT,
        help=f"Concurrent requests in phase 3. Default = {DEFAULT_CONN_LIMIT} "
             "(matches VERL_EXTERNAL_TEACHER_CONN_LIMIT).",
    )
    p.add_argument("--concurrency", type=int, default=DEFAULT_CONN_LIMIT)
    p.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    args = p.parse_args()
    sys.exit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
