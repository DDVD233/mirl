"""Throughput benchmark for the vLLM teacher.

Fires N concurrent /v1/completions requests with ~2000-token prompts and a
fixed output budget. Reports per-request latency stats + aggregate token/s.
Run from vps3 itself so we don't measure WAN RTT.
"""
import asyncio
import sys
import time

import aiohttp

URL = "http://127.0.0.1:8005/v1/completions"
MODEL = "Qwen/Qwen3.5-397B-A17B-FP8"
N_PARALLEL = int(sys.argv[1]) if len(sys.argv) > 1 else 32
MAX_OUTPUT = int(sys.argv[2]) if len(sys.argv) > 2 else 200
TARGET_INPUT_TOKENS = int(sys.argv[3]) if len(sys.argv) > 3 else 2000

SEED = (
    "Describe in technical detail how modern transformer language models are "
    "trained for medical reasoning. Cover attention mechanics, optimization "
    "schedules, model parallelism, mixture-of-experts routing, quantization "
    "techniques, retrieval augmentation, and reward-model alignment. Discuss "
    "tradeoffs between FP8 and BF16 weights for inference latency. "
)
PROMPT = (SEED * 1000)[: TARGET_INPUT_TOKENS * 4]


async def one(session, idx, t0):
    payload = {
        "model": MODEL,
        "prompt": PROMPT,
        "max_tokens": MAX_OUTPUT,
        "temperature": 0.7,
    }
    start = time.perf_counter()
    async with session.post(
        URL, json=payload, headers={"Authorization": "Bearer EMPTY"}
    ) as resp:
        body = await resp.json()
    elapsed = time.perf_counter() - start
    usage = body.get("usage") or {}
    err = body.get("error")
    return {
        "i": idx,
        "elapsed": elapsed,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "since_start": time.perf_counter() - t0,
        "err": err,
    }


async def main():
    print(
        f"benchmark: N={N_PARALLEL} parallel, target_input={TARGET_INPUT_TOKENS}, "
        f"max_output={MAX_OUTPUT}"
    )
    timeout = aiohttp.ClientTimeout(total=600)
    connector = aiohttp.TCPConnector(limit=N_PARALLEL + 4)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        warm = await one(session, -1, time.perf_counter())
        print(
            f"warmup: {warm['elapsed']:.2f}s "
            f"(input={warm.get('prompt_tokens')}, output={warm.get('completion_tokens')}, "
            f"err={warm.get('err')})"
        )

        t0 = time.perf_counter()
        results = await asyncio.gather(
            *(one(session, i, t0) for i in range(N_PARALLEL))
        )
        wall = time.perf_counter() - t0

    errors = [r for r in results if r.get("err")]
    if errors:
        print(f"  ERRORS ({len(errors)}): first = {errors[0]['err']}")
    ok = [r for r in results if not r.get("err")]
    latencies = sorted(r["elapsed"] for r in ok)
    completion_tokens_total = sum(r.get("completion_tokens") or 0 for r in ok)
    prompt_tokens_total = sum(r.get("prompt_tokens") or 0 for r in ok)
    actual_input = (prompt_tokens_total // len(ok)) if ok else 0
    actual_output = (completion_tokens_total // len(ok)) if ok else 0

    def pct(p):
        if not latencies:
            return float("nan")
        return latencies[min(len(latencies) - 1, int(p * len(latencies)))]

    print()
    print(f"wall (all {N_PARALLEL} in flight): {wall:.2f}s")
    print(
        f"per-req latency  min={latencies[0]:.2f}s  p50={pct(0.50):.2f}s  "
        f"p90={pct(0.90):.2f}s  p99={pct(0.99):.2f}s  max={latencies[-1]:.2f}s"
    )
    print(f"actual tokens    input/req={actual_input}  output/req={actual_output}")
    print(
        f"output tput      {completion_tokens_total / wall:.1f} tok/s  "
        f"({completion_tokens_total / wall / N_PARALLEL:.1f} tok/s per stream)"
    )
    print(f"prompt tput      {prompt_tokens_total / wall:.1f} tok/s  (prefill-bound)")
    print(
        f"total tput       "
        f"{(prompt_tokens_total + completion_tokens_total) / wall:.1f} tok/s"
    )


asyncio.run(main())
