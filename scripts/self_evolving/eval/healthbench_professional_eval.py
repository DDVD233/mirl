"""Evaluate a model on HealthBench Professional (OpenAI, 2026) and log to W&B.

HealthBench Professional is a 525-example, *text-only* benchmark of real
clinician chats (care consult / writing & documentation / medical research).
Each example is a conversation plus physician-written rubric criteria, each
with a point value in [-10, +10]; a model-based grader marks every criterion
met / not-met and the example score is

    score = sum(points of met criteria) / sum(positive points)

with a length adjustment (default: penalize answers longer than 2000 chars)
and the overall score is the mean of per-example length-adjusted scores,
clipped to [0, 1]. See the paper for details.

This is a *different* task and scoring scheme from our MIMIC-IV rare-disease
eval (scripts/self_evolving/eval_sota.py), so we do NOT reuse compute_score.
Instead we drive OpenAI's reference rubric grader from `openai/simple-evals`:
we vendor that repo at runtime, convert the HuggingFace dataset
(`openai/healthbench-professional`) into the HealthBench JSONL format the
grader expects, point the model-under-test sampler at our vLLM endpoint, and
wire up either the official GPT-5.4-low grader (for leaderboard-comparable
numbers) or a local vLLM judge (cheap, internal-only).

Exact simple-evals flags this reproduces (confirmed against the live repo):
    --eval=healthbench --healthbench-professional-mode
    --healthbench-input-path=<our converted jsonl>
    --healthbench-use-gpt-5-4-low-grader
    --healthbench-length-adjustment-center=2000
    --healthbench-length-adjustment-penalty-per-500-chars=0.0147

Usage (our vLLM model, official GPT-5.4-low grader -> comparable score):
    OPENAI_API_KEY=sk-... \
    python scripts/self_evolving/eval/healthbench_professional_eval.py \
        --model-base http://localhost:8000/v1 \
        --model-name Qwen/Qwen3.6-27B \
        --grader openai --grader-model gpt-5.4-2026-03-05 --grader-effort low \
        --wandb-project self_evolving_eval

Usage (local Qwen judge -> internal-only, no OpenAI key needed):
    python scripts/self_evolving/eval/healthbench_professional_eval.py \
        --model-base http://localhost:8000/v1 --model-name Qwen/Qwen3.6-27B \
        --grader local --grader-base http://node2500:8002/v1 \
        --grader-model Qwen/Qwen3.6-27B

Contamination: OpenAI asks that examples not be revealed online. By default we
log ONLY scalar scores + categorical tags + token counts to W&B. Pass
--log-examples to additionally push the rubric/conversation HTML report (only
do this to a private project).
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Length-adjustment constants from the HealthBench Professional paper:
#   s_len = s - 2.94e-5 * (chars - 2000)   ==>   penalty_per_500 = 2.94e-5 * 500
LENGTH_ADJ_CENTER = 2000.0
LENGTH_ADJ_PENALTY_PER_500 = 0.0147

SIMPLE_EVALS_URL = "https://github.com/openai/simple-evals"
HF_DATASET = "openai/healthbench-professional"
GRADER_GPT54 = "gpt-5.4-2026-03-05"        # OpenAI deployment id
GRADER_GPT54_TRAPI = "gpt-5.4_2026-03-05"  # same model, TRAPI proxy deployment id
_PKG_NAME = "hbpro_simple_evals"  # stable import alias for the vendored repo


# --------------------------------------------------------------------------- #
# Vendoring openai/simple-evals
# --------------------------------------------------------------------------- #
def ensure_simple_evals(src_dir: str):
    """Clone openai/simple-evals into `src_dir` (if absent) and import it under
    a stable package name regardless of the directory name (handles hyphens).
    Returns (healthbench_eval, common, chat_completion_sampler, responses_sampler).
    We deliberately avoid importing simple_evals.py (it pulls in the Claude /
    Anthropic sampler); only healthbench_eval + the two samplers are needed.
    """
    src = Path(src_dir).expanduser()
    if not (src / "healthbench_eval.py").exists():
        src.parent.mkdir(parents=True, exist_ok=True)
        print(f"[simple-evals] cloning {SIMPLE_EVALS_URL} -> {src}")
        subprocess.run(
            ["git", "clone", "--depth", "1", SIMPLE_EVALS_URL, str(src)],
            check=True,
        )
    # Make the dir importable as a package under a clean alias.
    for pkg in (src, src / "sampler"):
        init = pkg / "__init__.py"
        if not init.exists():
            init.touch()
    if _PKG_NAME not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            _PKG_NAME, str(src / "__init__.py"),
            submodule_search_locations=[str(src)],
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[_PKG_NAME] = module
        spec.loader.exec_module(module)

    hb = importlib.import_module(f"{_PKG_NAME}.healthbench_eval")
    common = importlib.import_module(f"{_PKG_NAME}.common")
    ccs = importlib.import_module(f"{_PKG_NAME}.sampler.chat_completion_sampler")
    rsp = importlib.import_module(f"{_PKG_NAME}.sampler.responses_sampler")

    sha = "unknown"
    try:
        sha = subprocess.check_output(
            ["git", "-C", str(src), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        pass
    return hb, common, ccs, rsp, sha


# --------------------------------------------------------------------------- #
# Dataset: download from HF and convert HF schema -> HealthBench JSONL schema
# --------------------------------------------------------------------------- #
def _download_hf_rows(repo_id: str, revision: str | None) -> list[dict]:
    """Return the raw HealthBench Professional rows from HuggingFace.

    HF schema: id, conversation{messages:[{role,content}]}, rubric_items[
    {criterion_text, points}], use_case, type, difficulty, specialty,
    physician_response, canary_string.
    """
    try:
        from datasets import load_dataset

        ds = load_dataset(repo_id, split="test", revision=revision)
        return [dict(r) for r in ds]
    except Exception as e_ds:  # fall back to raw parquet via the hub
        print(f"[data] `datasets` path failed ({e_ds}); trying raw parquet")
        try:
            from huggingface_hub import snapshot_download
            import pyarrow.parquet as pq

            local = snapshot_download(
                repo_id, repo_type="dataset", revision=revision
            )
            parquets = sorted(Path(local).rglob("*.parquet"))
            if not parquets:
                raise RuntimeError(f"no parquet files under {local}")
            rows: list[dict] = []
            for p in parquets:
                rows.extend(pq.read_table(p).to_pylist())
            return rows
        except Exception as e_pq:
            raise RuntimeError(
                "Could not load the HealthBench Professional dataset. Install "
                "`datasets` (pip install datasets) or `huggingface_hub`+`pyarrow`. "
                f"datasets error: {e_ds}; parquet error: {e_pq}"
            )


def _example_tags(row: dict) -> list[str]:
    """Categorical metadata -> HealthBench example_tags (unique, prefixed)."""
    tags = []
    for key in ("use_case", "type", "difficulty", "specialty"):
        val = row.get(key)
        if val:
            tags.append(f"{key}:{val}")
    return tags


def convert_to_healthbench(rows: list[dict]) -> tuple[list[dict], dict[str, dict]]:
    """Convert HF rows to HealthBench-format dicts and a prompt_id->tags map.

    HealthBench format (what healthbench_eval.HealthBenchEval expects per line):
        prompt:       list[{role, content}]   (the conversation, ends on user)
        rubrics:      list[{criterion, points, tags}]
        example_tags: list[str]
        prompt_id:    str
    """
    out: list[dict] = []
    tag_map: dict[str, dict] = {}
    for row in rows:
        conv = row["conversation"]
        messages = conv["messages"] if isinstance(conv, dict) else conv
        prompt = [
            {"role": m["role"], "content": m["content"]} for m in messages
        ]
        rubrics = []
        for item in row["rubric_items"]:
            criterion = item.get("criterion_text", item.get("criterion"))
            rubrics.append(
                {
                    "criterion": criterion,
                    "points": item["points"],
                    "tags": list(item.get("tags", []) or []),
                }
            )
        pid = row["id"]
        out.append(
            {
                "prompt": prompt,
                "rubrics": rubrics,
                "example_tags": _example_tags(row),
                "prompt_id": pid,
                "ideal_completions_data": None,  # HealthBenchEval reads this key
            }
        )
        tag_map[pid] = {
            "use_case": row.get("use_case"),
            "type": row.get("type"),
            "difficulty": row.get("difficulty"),
            "specialty": row.get("specialty"),
            "n_rubrics": len(rubrics),
        }
    return out, tag_map


# --------------------------------------------------------------------------- #
# Samplers
# --------------------------------------------------------------------------- #
def _norm_usage(u):
    """Normalize an OpenAI/vLLM/TRAPI `usage` object into the shape
    simple-evals' get_usage_dict expects (it subscripts
    *_tokens_details["cached_tokens"]/["reasoning_tokens"], which are None on
    vLLM and crash). Returns a SimpleNamespace or None."""
    if u is None:
        return None
    from types import SimpleNamespace
    ctd = getattr(u, "completion_tokens_details", None)
    reasoning = 0
    if ctd is not None:
        reasoning = (getattr(ctd, "reasoning_tokens", None)
                     if not isinstance(ctd, dict) else ctd.get("reasoning_tokens")) or 0
    return SimpleNamespace(
        prompt_tokens=getattr(u, "prompt_tokens", None),
        completion_tokens=getattr(u, "completion_tokens", None),
        total_tokens=getattr(u, "total_tokens", None),
        prompt_tokens_details={"cached_tokens": 0},
        completion_tokens_details={"reasoning_tokens": reasoning},
    )


def make_chat_sampler(ccs_module, *, base_url, api_key, model, provider="vllm",
                      temperature=0.0, max_tokens=4096, system_message=None,
                      enable_thinking=None, reasoning_effort=None):
    """A ChatCompletionSampler pointed at an arbitrary OpenAI-compatible base_url.

    provider="vllm": our vLLM servers. Sends temperature + max_tokens, supports
        the Qwen `enable_thinking` toggle, and falls back to reasoning_content
        when content is empty (truncated mid-thinking).
    provider="trapi": the TRAPI (Azure-OpenAI) proxy serving gpt-5.x. gpt-5.x
        reject `temperature` and `max_tokens`, so we send `max_completion_tokens`
        and an optional `reasoning_effort` (no temperature, no chat_template_kwargs).
        Mirrors the request shaping in eval_sota.py / generation_server.py.
    """
    from openai import OpenAI
    import openai as openai_mod

    base = ccs_module.ChatCompletionSampler
    SamplerResponse = importlib.import_module(f"{_PKG_NAME}.types").SamplerResponse
    is_trapi = provider == "trapi"

    class _Sampler(base):
        def __init__(self):
            self.client = OpenAI(base_url=base_url, api_key=api_key or "EMPTY")
            self.model = model
            self.system_message = system_message
            self.temperature = temperature
            self.max_tokens = max_tokens
            self.image_format = "url"
            self._extra_body = (
                {"chat_template_kwargs": {"enable_thinking": enable_thinking}}
                if (not is_trapi and enable_thinking is not None)
                else None
            )

        def __call__(self, message_list):
            if self.system_message:
                message_list = [
                    self._pack_message("system", self.system_message)
                ] + message_list
            trial = 0
            while True:
                try:
                    if is_trapi:
                        kwargs = dict(
                            model=self.model,
                            messages=message_list,
                            max_completion_tokens=self.max_tokens,
                        )
                        if reasoning_effort:
                            kwargs["reasoning_effort"] = reasoning_effort
                    else:
                        kwargs = dict(
                            model=self.model,
                            messages=message_list,
                            temperature=self.temperature,
                            max_tokens=self.max_tokens,
                        )
                        if self._extra_body:
                            kwargs["extra_body"] = self._extra_body
                    response = self.client.chat.completions.create(**kwargs)
                    msg = response.choices[0].message
                    content = msg.content
                    if not content:
                        content = getattr(msg, "reasoning_content", None)
                    if not content:
                        raise ValueError("empty response; retrying")
                    return SamplerResponse(
                        response_text=content,
                        response_metadata={"usage": _norm_usage(response.usage)},
                        actual_queried_message_list=message_list,
                    )
                except openai_mod.BadRequestError as e:
                    print("Bad Request Error", e)
                    return SamplerResponse(
                        response_text="No response (bad request).",
                        response_metadata={"usage": None},
                        actual_queried_message_list=message_list,
                    )
                except Exception as e:
                    backoff = min(2 ** trial, 60)
                    print(f"sampler retry {trial} after {backoff}s: {e}")
                    time.sleep(backoff)
                    trial += 1

    return _Sampler()


import re as _re

_HERMES_TOOLCALL_RE = _re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", _re.DOTALL)
# qwen3_coder XML form: <function=name><parameter=query>...</parameter></function>
_XML_QUERY_RE = _re.compile(r"<parameter=query>\s*(.*?)\s*</parameter>", _re.DOTALL | _re.IGNORECASE)


def _last_user_text(message_list) -> str:
    """Last user turn's text (handles str or multimodal-list content)."""
    for m in reversed(message_list):
        if m.get("role") != "user":
            continue
        c = m.get("content", "")
        if isinstance(c, str):
            return c
        if isinstance(c, list):
            return " ".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text").strip()
    return ""


def _parse_search_query(text: str) -> str | None:
    """Pull a `search_medical_kb` query out of a hermes <tool_call> block."""
    for m in _HERMES_TOOLCALL_RE.finditer(text or ""):
        try:
            obj = json.loads(m.group(1))
        except Exception:
            continue
        args = obj.get("arguments", obj)
        if isinstance(args, dict) and args.get("query"):
            return str(args["query"])
    xm = _XML_QUERY_RE.search(text or "")
    if xm and xm.group(1).strip():
        return xm.group(1).strip()
    return None


def wrap_retrieval_sampler(base_sampler, ccs_module, *, retrieval_url, top_k, tool_name="search_medical_kb"):
    """Wrap a solver sampler in the mandatory 2-turn retrieval flow used at train
    time: (turn 1) the model writes a `search_medical_kb` query, (tool) the KB
    passages are injected, (turn 2) the model answers grounded in them. If the
    model emits no parseable query, fall back to the raw user question — matching
    RetrievalToolAgentLoop's guarantee that every rollout retrieves.
    """
    import requests

    SamplerResponse = importlib.import_module(f"{_PKG_NAME}.types").SamplerResponse
    query_instr = (
        f"You have a tool `{tool_name}` that searches a medical knowledge base. "
        f"Before answering, output a single tool call and nothing else, exactly as:\n"
        f'<tool_call>{{"name": "{tool_name}", "arguments": {{"query": "<your focused clinical search query>"}}}}</tool_call>'
    )

    class _RetrievalSampler:
        def __init__(self):
            # Surface base sampler attributes HealthBenchEval may introspect.
            self.model = getattr(base_sampler, "model", "")
            self.system_message = getattr(base_sampler, "system_message", None)

        def _pack(self, role, content):
            return {"role": role, "content": content}

        def __call__(self, message_list):
            user_q = _last_user_text(message_list)
            # Turn 1: elicit a search query.
            q_msgs = list(message_list) + [self._pack("user", query_instr)]
            try:
                r1 = base_sampler(q_msgs)
                query = _parse_search_query(r1.response_text) or user_q
            except Exception:
                query = user_q
            # Retrieve (never fatal).
            try:
                resp = requests.post(retrieval_url, json={"query": query, "top_k": top_k}, timeout=30)
                resp.raise_for_status()
                passages = resp.json().get("text") or "No relevant passages found."
            except Exception as e:
                print(f"[retrieval] failed ({type(e).__name__}: {e}); answering ungrounded")
                passages = "Retrieval unavailable; answer from your own knowledge."
            # Turn 2: answer grounded in the retrieved passages.
            ctx = (
                "Retrieved medical knowledge (use it to ground your answer; it may be "
                f"incomplete):\n\n{passages}\n\nNow answer the original request."
            )
            a_msgs = list(message_list) + [self._pack("user", ctx)]
            r2 = base_sampler(a_msgs)
            return SamplerResponse(
                response_text=r2.response_text,
                response_metadata=getattr(r2, "response_metadata", {"usage": None}),
                actual_queried_message_list=a_msgs,
            )

    return _RetrievalSampler()


def build_grader(args, ccs_module, rsp_module):
    """Construct the rubric grader sampler. Returns (sampler, description)."""
    effort = args.grader_effort or None
    if args.grader == "local":
        sampler = make_chat_sampler(
            ccs_module,
            base_url=args.grader_base,
            api_key=args.grader_key,
            model=args.grader_model_local or args.grader_model,
            provider="vllm",
            temperature=0.0,
            max_tokens=2048,
            enable_thinking=False,  # judge: never burn tokens thinking
        )
        return sampler, (
            f"local vLLM judge {args.grader_model_local or args.grader_model} "
            f"@ {args.grader_base} (NOT leaderboard-comparable)"
        )
    if args.grader == "trapi":
        # GPT-5.4 low via the TRAPI proxy == the official Professional grader
        # (chat-completions transport instead of the Responses API).
        # Map the OpenAI default id to the TRAPI deployment id if unchanged.
        grader_model = (GRADER_GPT54_TRAPI if args.grader_model == GRADER_GPT54
                        else args.grader_model)
        sampler = make_chat_sampler(
            ccs_module,
            base_url=args.grader_base,
            api_key=args.grader_key,
            model=grader_model,
            provider="trapi",
            max_tokens=2048,
            reasoning_effort=effort,
        )
        comparable = grader_model == GRADER_GPT54_TRAPI and effort == "low"
        note = "" if comparable else "  (NOTE: not the official GPT-5.4-low setting)"
        return sampler, f"TRAPI grader {grader_model} effort={effort}{note}"
    # openai grader: the official professional setting is GPT-5.4 low via OpenAI.
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY is required for --grader openai. Use --grader trapi "
            "(GPT-5.4 low via the TRAPI proxy) or --grader local instead."
        )
    if args.grader_api == "chat":
        OPENAI_SYS = ccs_module.OPENAI_SYSTEM_MESSAGE_API
        sampler = ccs_module.ChatCompletionSampler(
            model=args.grader_model, system_message=OPENAI_SYS, max_tokens=2048
        )
        return sampler, f"OpenAI chat grader {args.grader_model}"
    sampler = rsp_module.ResponsesSampler(
        model=args.grader_model,
        reasoning_model=bool(effort),
        reasoning_effort=effort,
        max_tokens=2048,
    )
    comparable = args.grader_model == GRADER_GPT54 and effort == "low"
    note = "" if comparable else "  (NOTE: not the official GPT-5.4-low setting)"
    return sampler, f"OpenAI responses grader {args.grader_model} effort={effort}{note}"


# --------------------------------------------------------------------------- #
# W&B logging
# --------------------------------------------------------------------------- #
def log_to_wandb(args, result, tag_map, config, html, log_examples):
    import wandb

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=args.wandb_run_name
        or f"hbpro-{args.model_name.replace('/', '_')}",
        group="healthbench_professional",
        job_type="eval",
        config=config,
        mode=args.wandb_mode,
        reinit=True,
    )

    prefix = "healthbench_professional"
    metrics = dict(result.metrics or {})
    flat = {f"{prefix}/{k}": v for k, v in metrics.items()}
    flat[f"{prefix}/score"] = result.score
    wandb.log(flat)

    # Headline numbers go to the run summary for the project table.
    headline = {
        f"{prefix}/overall_score": metrics.get("overall_score", result.score),
    }
    if "overall_score_length_adjusted" in metrics:
        headline[f"{prefix}/overall_score_length_adjusted"] = metrics[
            "overall_score_length_adjusted"
        ]
    wandb.summary.update(headline)

    # Per-example table (scores + categorical tags only, unless --log-examples).
    meta = (result.metadata or {}).get("example_level_metadata", []) or []
    cols = [
        "prompt_id", "use_case", "type", "difficulty", "specialty",
        "n_rubrics", "score", "score_length_adjusted", "answer_chars",
        "output_tokens",
    ]
    if log_examples:
        cols.append("completion")
    table = wandb.Table(columns=cols)
    for m in meta:
        pid = m.get("prompt_id", "")
        tags = tag_map.get(pid, {})
        completion = ""
        comp = m.get("completion") or []
        if comp:
            completion = comp[0].get("content", "")
        chars = len(completion)
        score = m.get("score")
        la = None
        if score is not None and not args.no_length_adjustment:
            la = score - args.length_penalty_per_500 * (
                (chars - args.length_center) / 500.0
            )
        usage = m.get("usage") or {}
        rowvals = [
            pid, tags.get("use_case"), tags.get("type"),
            tags.get("difficulty"), tags.get("specialty"),
            tags.get("n_rubrics"), score, la, chars,
            usage.get("output_tokens"),
        ]
        if log_examples:
            rowvals.append(completion)
        table.add_data(*rowvals)
    wandb.log({f"{prefix}/per_example": table})

    if log_examples and html:
        wandb.log({f"{prefix}/report": wandb.Html(html)})

    run.finish()


# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(
        description="Run HealthBench Professional on a model and log to W&B.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Model under test (OpenAI-compatible / vLLM)
    p.add_argument("--model-base", default=os.environ.get("MODEL_BASE", "http://localhost:8000/v1"),
                   help="OpenAI-compatible base URL of the model under test (vLLM or TRAPI proxy).")
    p.add_argument("--model-name", default=os.environ.get("MODEL_NAME", "Qwen/Qwen3.6-27B"))
    p.add_argument("--model-key", default=os.environ.get("MODEL_KEY", "EMPTY"))
    p.add_argument("--model-provider", choices=["vllm", "trapi"],
                   default=os.environ.get("MODEL_PROVIDER", "vllm"),
                   help="vllm=temperature+max_tokens; trapi=max_completion_tokens, no temperature (gpt-5.x).")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--system-message", default=None,
                   help="Optional system prompt for the model under test.")
    thinking = p.add_mutually_exclusive_group()
    thinking.add_argument("--enable-thinking", dest="enable_thinking", action="store_true", default=None,
                          help="Inject chat_template_kwargs.enable_thinking=true (Qwen3 reasoning).")
    thinking.add_argument("--disable-thinking", dest="enable_thinking", action="store_false",
                          help="Inject chat_template_kwargs.enable_thinking=false.")

    # Grader
    p.add_argument("--grader", choices=["openai", "trapi", "local"], default="openai",
                   help="openai/trapi=GPT-5.4-low reference grader (comparable); local=your vLLM judge.")
    p.add_argument("--grader-api", choices=["responses", "chat"], default="responses",
                   help="OpenAI grader API. GPT-5.4 reasoning uses responses.")
    p.add_argument("--grader-model", default=os.environ.get("GRADER_MODEL", GRADER_GPT54),
                   help="OpenAI grader model id.")
    p.add_argument("--grader-effort", default="low",
                   help="Reasoning effort for the OpenAI grader ('' for non-reasoning).")
    p.add_argument("--grader-base", default=os.environ.get("GRADER_BASE", "http://node2500:8002/v1"),
                   help="Base URL for --grader local.")
    p.add_argument("--grader-key", default=os.environ.get("GRADER_KEY", "EMPTY"))
    p.add_argument("--grader-model-local", default=os.environ.get("GRADER_MODEL_LOCAL"),
                   help="Model id for --grader local (defaults to --grader-model).")

    # Eval config
    p.add_argument("--limit", type=int, default=0, help="Eval only N examples (0 = all 525).")
    p.add_argument("--n-repeats", type=int, default=1)
    p.add_argument("--concurrency", type=int, default=16,
                   help="Outer threads. NOTE: each example also fans out one grader call per rubric item.")
    p.add_argument("--no-length-adjustment", action="store_true",
                   help="Report raw overall_score only (skip the length-adjusted primary metric).")
    p.add_argument("--length-center", type=float, default=LENGTH_ADJ_CENTER)
    p.add_argument("--length-penalty-per-500", type=float, default=LENGTH_ADJ_PENALTY_PER_500)

    # Data / repo
    p.add_argument("--hf-dataset", default=HF_DATASET)
    p.add_argument("--hf-revision", default=None)
    p.add_argument("--input-path", default=None,
                   help="Skip HF download: use this HealthBench-format JSONL (e.g. from the official assets.zip).")
    p.add_argument("--simple-evals-dir",
                   default=os.environ.get("SIMPLE_EVALS_DIR", str(Path.home() / ".cache" / "simple_evals_src")))
    p.add_argument("--output-dir",
                   default=os.environ.get("OUTPUT_DIR", "/scratch/sheng/self_evolving/logs/healthbench_professional"))

    # W&B
    p.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "self_evolving_eval"))
    p.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    p.add_argument("--wandb-run-name", default=os.environ.get("WANDB_RUN_NAME"))
    p.add_argument("--wandb-mode", default=os.environ.get("WANDB_MODE", "online"))
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--log-examples", action="store_true",
                   help="Also push the rubric/conversation HTML report + completions to W&B (private projects only; contamination).")

    p.add_argument("--dry-run", action="store_true",
                   help="Download+convert the dataset and exit (no model/grader calls).")
    p.add_argument("--retrieval", action="store_true",
                   help="Wrap the solver in the mandatory 2-turn retrieval flow "
                        "(matches RetrievalToolAgentLoop at train time).")
    p.add_argument("--retrieval-url",
                   default=os.environ.get("RETRIEVAL_URL", "http://localhost:8006/retrieve"),
                   help="Gen-server /retrieve endpoint used when --retrieval is set.")
    p.add_argument("--retrieval-topk", type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args()

    out_dir = Path(args.output_dir)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        out_dir = Path("scripts/self_evolving/logs/healthbench_professional")
        out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    hb, common, ccs, rsp, sha = ensure_simple_evals(args.simple_evals_dir)

    # ---- dataset ---------------------------------------------------------- #
    jsonl_path = out_dir / f"healthbench_professional_{stamp}.jsonl"
    if args.input_path:
        input_path = args.input_path
        tag_map: dict[str, dict] = {}
        n_examples = None
        print(f"[data] using provided HealthBench-format input: {input_path}")
    else:
        print(f"[data] downloading {args.hf_dataset} from HuggingFace ...")
        rows = _download_hf_rows(args.hf_dataset, args.hf_revision)
        examples, tag_map = convert_to_healthbench(rows)
        with jsonl_path.open("w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
        input_path = str(jsonl_path)
        n_examples = len(examples)
        print(f"[data] converted {n_examples} examples -> {input_path}")

    if args.dry_run:
        print("[dry-run] dataset ready; exiting before any model/grader calls.")
        return

    # ---- samplers --------------------------------------------------------- #
    model_sampler = make_chat_sampler(
        ccs,
        base_url=args.model_base,
        api_key=args.model_key,
        model=args.model_name,
        provider=args.model_provider,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        system_message=args.system_message,
        enable_thinking=args.enable_thinking,
    )
    if args.retrieval:
        model_sampler = wrap_retrieval_sampler(
            model_sampler, ccs,
            retrieval_url=args.retrieval_url,
            top_k=args.retrieval_topk,
        )
        print(f"[retrieval] ON — 2-turn flow via {args.retrieval_url} (top_k={args.retrieval_topk})")
    grader, grader_desc = build_grader(args, ccs, rsp)
    print(f"[grader] {grader_desc}")

    length_center = None if args.no_length_adjustment else args.length_center
    length_pen = None if args.no_length_adjustment else args.length_penalty_per_500

    eval_obj = hb.HealthBenchEval(
        grader_model=grader,
        num_examples=(args.limit or None),
        n_repeats=args.n_repeats,
        n_threads=args.concurrency,
        input_path=input_path,
        length_adjustment_center=length_center,
        length_adjustment_penalty_per_500_chars=length_pen,
    )

    print(f"[run] evaluating {args.model_name} @ {args.model_base} ...")
    t0 = time.time()
    result = eval_obj(model_sampler)
    dt = time.time() - t0

    metrics = dict(result.metrics or {})
    print("\n=== HealthBench Professional results ===")
    print(f"overall_score                 : {metrics.get('overall_score')}")
    if "overall_score_length_adjusted" in metrics:
        print(f"overall_score_length_adjusted : {metrics['overall_score_length_adjusted']}  <-- primary")
    print(f"(elapsed {dt:.0f}s over {len(eval_obj.examples)} graded examples)")

    # ---- persist full results locally (safe to contain text) -------------- #
    html = common.make_report(result)
    (out_dir / f"report_{stamp}.html").write_text(html)
    (out_dir / f"metrics_{stamp}.json").write_text(
        json.dumps({"score": result.score, "metrics": metrics}, indent=2)
    )
    (out_dir / f"allresults_{stamp}.json").write_text(
        json.dumps(
            {
                "score": result.score,
                "metrics": metrics,
                "metadata": result.metadata,
            },
            indent=2,
        )
    )
    # Self-describing, stably-named summary (model + grader + headline + per-tag
    # metrics) so results are aggregatable from disk, not just chat/W&B.
    label = args.wandb_run_name or f"hbpro-{args.model_name.replace('/', '_')}"
    safe = "".join(c if (c.isalnum() or c in "._-") else "_" for c in label)
    (out_dir / f"result__{safe}.json").write_text(json.dumps({
        "label": label,
        "benchmark": "healthbench_professional",
        "model_under_test": args.model_name,
        "model_provider": args.model_provider,
        "model_base": args.model_base,
        "grader": grader_desc,
        "grader_mode": args.grader,
        "grader_model": args.grader_model,
        "n_examples": len(eval_obj.examples),
        "limit": args.limit or None,
        "overall_score": metrics.get("overall_score"),
        "overall_score_length_adjusted": metrics.get("overall_score_length_adjusted"),
        "score": result.score,
        "metrics": metrics,
        "stamp": stamp,
    }, indent=2))
    print(f"[out] wrote report/metrics/allresults/result__{safe} to {out_dir}")

    # ---- W&B -------------------------------------------------------------- #
    if not args.no_wandb:
        config = {
            "benchmark": "healthbench_professional",
            "model_under_test": args.model_name,
            "model_base": args.model_base,
            "model_provider": args.model_provider,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "enable_thinking": args.enable_thinking,
            "grader": grader_desc,
            "grader_mode": args.grader,
            "grader_model": (args.grader_model_local or args.grader_model)
            if args.grader == "local" else args.grader_model,
            "grader_comparable": (
                (args.grader == "openai" and args.grader_model == GRADER_GPT54
                 and args.grader_effort == "low")
                or (args.grader == "trapi"
                    and args.grader_model in (GRADER_GPT54, GRADER_GPT54_TRAPI)
                    and args.grader_effort == "low")
            ),
            "length_adjustment": not args.no_length_adjustment,
            "length_center": args.length_center,
            "length_penalty_per_500": args.length_penalty_per_500,
            "n_examples": n_examples,
            "limit": args.limit or None,
            "n_repeats": args.n_repeats,
            "simple_evals_sha": sha,
            "hf_dataset": args.hf_dataset,
        }
        try:
            log_to_wandb(args, result, tag_map, config, html, args.log_examples)
            print("[wandb] logged.")
        except Exception as e:
            print(f"[wandb] logging failed ({e}); local artifacts are in {out_dir}")


if __name__ == "__main__":
    main()
