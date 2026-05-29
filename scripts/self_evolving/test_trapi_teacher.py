"""Validate that a TRAPI endpoint can serve as the external OPD teacher.

TRAPI (Microsoft's internal OpenAI-compatible gateway) is a candidate teacher
endpoint for on-policy distillation. Two things have to hold before we can
point `run_qwen36_27b_full_opd.sh` at it:

  1. It must accept the *exact* request that
     `verl.experimental.teacher_loop.external_client.ExternalLLMServerClient`
     sends: POST {base_url}/completions with a token-id `prompt` and vLLM's
     `prompt_logprobs` extension, returning `choices[0].prompt_logprobs`.

  2. Auth must survive a long run. TRAPI uses Microsoft Entra ID (AAD) tokens
     that expire ~hourly, so a *static* Bearer header (what external_client.py
     sends today, and what test_teacher_logprobs.py mimics) will 401 mid-run.
     This script resolves a fresh token per request via azure-identity, which
     is the pattern external_client.py would need to adopt for TRAPI.

Verified working against (2026-05):
    instance = redmond/interactive
    model    = Qwen/Qwen3.5-397B-A17B-GPTQ-Int4

Prereqs (on the host that has the SC-ALT login, e.g. point.dd.works):
    az login --scope api://trapi/.default
    pip install openai azure-identity requests

Usage:
    python scripts/self_evolving/test_trapi_teacher.py
    # other instance/model:
    python scripts/self_evolving/test_trapi_teacher.py \\
        --base-url https://trapi.research.microsoft.com/redmond/interactive/openai/v1 \\
        --model Qwen/Qwen3.5-397B-A17B-GPTQ-Int4 \\
        --prompt-logprobs 1

Caveats for actually wiring this into OPD:
  - external_client.py sets `Authorization: Bearer {external_api_key}` once at
    init and never refreshes (external_client.py:120). To use TRAPI you must
    either pass a long-lived TRAPI API-KEY as TEACHER_API_KEY, or patch the
    client to refresh the AAD token per request (this script shows how).
  - The trainer bursts ~batch_size*n teacher calls per step (client-capped at
    VERL_EXTERNAL_TEACHER_CONN_LIMIT=32). A shared/interactive TRAPI instance
    may throttle (429); confirm rate limits with the instance delegate.
  - TRAPI here serves the GPTQ-Int4 quant, not the FP8 the OPD config names.
    Same architecture + tokenizer, so token-id prompts align and logprob
    lengths match; the teacher signal is just from a different quant.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Callable

import requests

DEFAULT_SCOPE = "api://trapi/.default"
DEFAULT_BASE_URL = os.environ.get(
    "TRAPI_BASE_URL",
    "https://trapi.research.microsoft.com/redmond/interactive/openai/v1",
)
DEFAULT_MODEL = os.environ.get("TRAPI_MODEL", "Qwen/Qwen3.5-397B-A17B-GPTQ-Int4")


def _make_token_provider(scope: str) -> Callable[[], str]:
    """Return a callable that yields a fresh AAD bearer token on each call.

    Tries `az login` creds first, then a managed identity — same chain the
    TRAPI getting-started guide uses.
    """
    from azure.identity import (
        AzureCliCredential,
        ChainedTokenCredential,
        ManagedIdentityCredential,
        get_bearer_token_provider,
    )

    return get_bearer_token_provider(
        ChainedTokenCredential(AzureCliCredential(), ManagedIdentityCredential()),
        scope,
    )


def test_chat(base_url: str, model: str, token_provider: Callable[[], str]) -> bool:
    """Phase 1: plain chat completion — proves auth + the instance are alive."""
    print("\n=== phase 1: chat completion sanity ===", flush=True)
    resp = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {token_provider()}"},
        json={
            "model": model,
            "messages": [
                {"role": "user", "content": "Give a one word answer, what is the capital of France?"}
            ],
        },
        timeout=120,
    )
    if resp.status_code != 200:
        print(f"  FAIL  HTTP {resp.status_code}: {resp.text[:300]}", flush=True)
        return False
    content = resp.json()["choices"][0]["message"]["content"].strip()
    print(f"  OK    response={content!r}", flush=True)
    return True


def test_prompt_logprobs(
    base_url: str,
    model: str,
    token_provider: Callable[[], str],
    prompt_logprobs: int,
) -> bool:
    """Phase 2: the OPD contract — token-id prompt + vLLM prompt_logprobs.

    Mirrors ExternalLLMServerClient.generate() / _normalize_prompt_logprobs:
    we send a list[int] prompt and require choices[0].prompt_logprobs back,
    shaped [None, {tok_id: {logprob, rank, ...}}, ...] with len == len(prompt).
    """
    print("\n=== phase 2: prompt_logprobs (OPD teacher contract) ===", flush=True)
    # "The capital of France is" under the Qwen3.5 tokenizer; opaque to the
    # server, which scores each position without re-tokenizing.
    prompt_ids = [785, 6722, 315, 9625, 374]
    resp = requests.post(
        f"{base_url.rstrip('/')}/completions",
        headers={"Authorization": f"Bearer {token_provider()}"},
        json={
            "model": model,
            "prompt": prompt_ids,
            "max_tokens": 1,
            "temperature": 1.0,
            "prompt_logprobs": prompt_logprobs,
        },
        timeout=120,
    )
    if resp.status_code != 200:
        print(f"  FAIL  HTTP {resp.status_code}: {resp.text[:300]}", flush=True)
        return False
    pl = resp.json()["choices"][0].get("prompt_logprobs")
    if pl is None:
        print("  FAIL  endpoint returned no prompt_logprobs field "
              "(not vLLM, or prompt_logprobs disabled)", flush=True)
        return False
    if len(pl) != len(prompt_ids):
        print(f"  FAIL  prompt_logprobs len={len(pl)} != prompt len={len(prompt_ids)}", flush=True)
        return False
    # external_client expects entry[0] is None, then dicts of {tok_id: {logprob, rank}}.
    first_none = pl[0] is None
    sample = pl[1] if len(pl) > 1 and pl[1] is not None else {}
    keys = list(sample.keys())[:3]
    has_logprob = bool(sample) and all("logprob" in v for v in sample.values())
    has_rank = bool(sample) and all("rank" in v for v in sample.values())
    print(f"  OK    lp_len={len(pl)}  first_is_None={first_none}  "
          f"k_per_pos={len(sample)}  sample_keys={keys}", flush=True)
    print(f"        entries carry logprob={has_logprob} rank={has_rank} "
          f"(both required by _normalize_prompt_logprobs)", flush=True)
    return first_none and has_logprob and has_rank


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--scope", default=DEFAULT_SCOPE)
    p.add_argument(
        "--prompt-logprobs", type=int, default=1,
        help="K for prompt_logprobs. Match the trainer's DISTILL_TOPK: k1 + "
             "use_topk=False sends 0; forward_kl_topk sends DISTILL_TOPK.",
    )
    args = p.parse_args()

    print(f"base_url = {args.base_url}", flush=True)
    print(f"model    = {args.model}", flush=True)
    try:
        token_provider = _make_token_provider(args.scope)
        token_provider()  # fail fast if not logged in
    except Exception as e:  # noqa: BLE001
        print(f"\nAuth failed ({type(e).__name__}: {e}).\n"
              f"Run: az login --scope {args.scope}", flush=True)
        sys.exit(1)

    ok = test_chat(args.base_url, args.model, token_provider)
    ok = test_prompt_logprobs(args.base_url, args.model, token_provider, args.prompt_logprobs) and ok

    if ok:
        print("\nAll phases passed — TRAPI endpoint speaks the OPD teacher contract.\n"
              "Remember: external_client.py needs token refresh (or a long-lived "
              "API-KEY) before a real run.", flush=True)
        sys.exit(0)
    print("\nOne or more phases failed — see above.", flush=True)
    sys.exit(1)


if __name__ == "__main__":
    main()
