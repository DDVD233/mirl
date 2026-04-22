"""
Reasoning evaluation: four-mode inference in a single model load.

For each dataset entry, produces:
  1. Direct prediction   (no thinking, greedy) + label logit distribution
  2. Reasoning prediction (thinking, greedy)   + label logit distribution + trace tokens
  3. N stochastic reasoning samples             (temperature=0.6, top_p=0.95, top_k=20)
  4. Paraphrased-input reasoning prediction     (thinking, greedy)

All outputs saved to a single JSONL. Run compute_reasoning_metrics.py afterwards.

Usage:
    CUDA_VISIBLE_DEVICES=0 python reasoning_eval.py \
        --model Qwen/Qwen2.5-Omni-7B \
        --input_jsonl   /data/test_dreaddit_prompts.jsonl \
        --para_input_jsonl /data/test_dreaddit_prompts_paraphrased.jsonl \
        --output_jsonl  /results/reasoning_eval/dreaddit_reasoning.jsonl \
        --data_loading  verl_style
"""

import sys
import os
import argparse
import json
import re
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from inference import (
    load_model,
    build_entry_inputs,
    extract_answer,
    _get_device,
    _generate_kwargs,
)

import torch
import torch.nn.functional as F
from transformers import LogitsProcessor, LogitsProcessorList
from tqdm import tqdm


# ── Logit capture ──────────────────────────────────────────────────────────────

class CaptureAnswerLogitsProcessor(LogitsProcessor):
    """
    Captures the model's probability distribution over class labels at the
    position of the first answer token (the token generated right after \\boxed{).

    For thinking mode: waits until </think> has been generated before capturing,
    preventing false triggers from \\boxed{} references inside the reasoning trace.

    Usage:
        capture_lp = CaptureAnswerLogitsProcessor(tokenizer, label_token_ids, prompt_length, thinking_mode)
        model.generate(..., logits_processor=LogitsProcessorList([capture_lp]))
        probs = capture_lp.captured_probs  # dict[label -> float] or None
    """

    def __init__(self, tokenizer, label_token_ids: dict, prompt_length: int,
                 thinking_mode: bool = True):
        self.tokenizer = tokenizer
        self.label_token_ids = label_token_ids
        self.prompt_length = prompt_length
        self._decoded_so_far = ""
        self._after_think_end = not thinking_mode   # direct mode: start ready
        self._post_think_offset = 0
        self.captured_probs: dict | None = None

    def __call__(self, input_ids: torch.LongTensor,
                 scores: torch.FloatTensor) -> torch.FloatTensor:
        # Only act on generated tokens (skip the initial prompt call where
        # input_ids still has length == prompt_length).
        if input_ids.shape[1] > self.prompt_length and self.captured_probs is None:
            # Append the last committed token to the accumulation buffer.
            last_tok = self.tokenizer.decode(
                [input_ids[0, -1].item()], skip_special_tokens=True
            )
            self._decoded_so_far += last_tok

            # In thinking mode, wait until </think> before searching for \boxed{.
            if not self._after_think_end:
                if "</think>" in self._decoded_so_far:
                    self._after_think_end = True
                    self._post_think_offset = (
                        self._decoded_so_far.index("</think>") + len("</think>")
                    )

            if self._after_think_end:
                searchable = self._decoded_so_far[self._post_think_offset:]
                if "\\boxed{" in searchable:
                    # scores here are for the token immediately after \boxed{
                    # — i.e., the first character of the answer.
                    label_ids = [self.label_token_ids[lbl] for lbl in self.label_token_ids]
                    label_logits = scores[0, label_ids].float()
                    label_probs = F.softmax(label_logits, dim=-1)
                    self.captured_probs = {
                        lbl: label_probs[i].item()
                        for i, lbl in enumerate(self.label_token_ids)
                    }

        return scores  # never modify scores


def get_label_token_ids(processor, labels: list[str]) -> dict[str, int]:
    """Map each class label to its first token ID (no special tokens)."""
    result = {}
    for lbl in labels:
        ids = processor.tokenizer.encode(lbl, add_special_tokens=False)
        if not ids:
            ids = processor.tokenizer.encode(" " + lbl, add_special_tokens=False)
        if len(ids) > 1:
            print(f"[INFO] Label '{lbl}' → {len(ids)} tokens; using first token for logit capture.")
        result[lbl] = ids[0]
    return result


# ── Per-entry helpers ──────────────────────────────────────────────────────────

def extract_think_trace(response: str) -> str:
    """Extract text between <think>...</think>. Returns '' if absent."""
    m = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
    return m.group(1).strip() if m else ""


def count_tokens_precisely(processor, text: str) -> int:
    """Exact token count of text (no special tokens)."""
    return len(processor.tokenizer.encode(text, add_special_tokens=False))


def _build_processor_inputs(entry: dict, base_dir: str, thinking: bool,
                             data_loading: str, processor):
    """Return (inputs_dict, prompt_length) for a single entry."""
    content, audio_list, imgs, vframes = build_entry_inputs(
        entry, base_dir, thinking, data_loading
    )
    msgs = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    proc_kwargs = dict(text=text, return_tensors="pt", padding=True)
    if audio_list is not None:
        proc_kwargs["audio"] = audio_list
    if imgs:
        proc_kwargs["images"] = imgs
    if vframes:
        proc_kwargs["videos"] = [vframes]

    device = _get_device(processor if hasattr(processor, 'device') else None)
    inputs = processor(**proc_kwargs)
    return inputs, inputs["input_ids"].shape[1]


def infer_with_logit_capture(
    model, processor, entry: dict, base_dir: str,
    thinking: bool, max_new_tokens: int, data_loading: str,
    label_token_ids: dict,
) -> tuple[str, dict | None]:
    """
    Single-sample greedy inference with logit capture at the answer position.

    Returns:
        response_text: decoded model output (excludes prompt)
        captured_probs: dict[label -> prob] or None if \\boxed{ was not found
    """
    content, audio_list, imgs, vframes = build_entry_inputs(
        entry, base_dir, thinking, data_loading
    )
    msgs = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    proc_kwargs = dict(text=text, return_tensors="pt", padding=True)
    if audio_list is not None:
        proc_kwargs["audio"] = audio_list
    if imgs:
        proc_kwargs["images"] = imgs
    if vframes:
        proc_kwargs["videos"] = [vframes]

    device = _get_device(model)
    inputs = processor(**proc_kwargs)
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
              for k, v in inputs.items()}
    prompt_length = inputs["input_ids"].shape[1]

    capture_lp = CaptureAnswerLogitsProcessor(
        tokenizer=processor.tokenizer,
        label_token_ids=label_token_ids,
        prompt_length=prompt_length,
        thinking_mode=thinking,
    )

    with torch.inference_mode():
        raw = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            logits_processor=LogitsProcessorList([capture_lp]),
            **_generate_kwargs(processor),
        )

    out_ids = raw[0] if isinstance(raw, (tuple, list)) else raw
    response = processor.decode(out_ids[0][prompt_length:], skip_special_tokens=True)
    return response, capture_lp.captured_probs


def infer_stochastic(
    model, processor, entry: dict, base_dir: str,
    max_new_tokens: int, data_loading: str,
) -> str:
    """Single stochastic reasoning sample (temperature=0.6, top_p=0.95, top_k=20)."""
    content, audio_list, imgs, vframes = build_entry_inputs(
        entry, base_dir, thinking=True, data_loading=data_loading
    )
    msgs = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    proc_kwargs = dict(text=text, return_tensors="pt", padding=True)
    if audio_list is not None:
        proc_kwargs["audio"] = audio_list
    if imgs:
        proc_kwargs["images"] = imgs
    if vframes:
        proc_kwargs["videos"] = [vframes]

    device = _get_device(model)
    inputs = processor(**proc_kwargs)
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
              for k, v in inputs.items()}
    prompt_length = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        raw = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            **_generate_kwargs(processor),
        )

    out_ids = raw[0] if isinstance(raw, (tuple, list)) else raw
    return processor.decode(out_ids[0][prompt_length:], skip_special_tokens=True)


# ── Four inference mode runners ────────────────────────────────────────────────

def run_mode_direct(model, processor, entries: list[dict], base_dir: str,
                    max_new_tokens: int, data_loading: str,
                    label_token_ids: dict, results: list[dict],
                    save_every: int, flush_fn) -> None:
    """Mode 1: greedy direct prediction (no thinking) + label logit capture."""
    for i, entry in enumerate(tqdm(entries, desc="Mode 1/4 — Direct")):
        try:
            response, label_probs = infer_with_logit_capture(
                model, processor, entry, base_dir,
                thinking=False, max_new_tokens=max_new_tokens,
                data_loading=data_loading, label_token_ids=label_token_ids,
            )
        except Exception as exc:
            print(f"\n[WARN] Direct mode entry {i}: {exc.__class__.__name__}: {exc}")
            traceback.print_exc()
            response, label_probs = "", None

        results[i]["direct_response"] = response
        results[i]["direct_answer"]   = extract_answer(response)
        results[i]["direct_label_probs"] = label_probs

        if save_every > 0 and (i + 1) % save_every == 0:
            flush_fn()

    flush_fn()


def run_mode_reasoning(model, processor, entries: list[dict], base_dir: str,
                       max_new_tokens: int, data_loading: str,
                       label_token_ids: dict, results: list[dict],
                       save_every: int, flush_fn) -> None:
    """Mode 2: greedy reasoning prediction (thinking) + label logit capture + trace tokens."""
    for i, entry in enumerate(tqdm(entries, desc="Mode 2/4 — Reasoning")):
        try:
            response, label_probs = infer_with_logit_capture(
                model, processor, entry, base_dir,
                thinking=True, max_new_tokens=max_new_tokens,
                data_loading=data_loading, label_token_ids=label_token_ids,
            )
        except Exception as exc:
            print(f"\n[WARN] Reasoning mode entry {i}: {exc.__class__.__name__}: {exc}")
            traceback.print_exc()
            response, label_probs = "", None

        trace = extract_think_trace(response)
        trace_tokens = count_tokens_precisely(processor, trace) if trace else 0

        results[i]["reasoning_response"]      = response
        results[i]["reasoning_answer"]         = extract_answer(response)
        results[i]["reasoning_trace"]          = trace
        results[i]["reasoning_trace_tokens"]   = trace_tokens
        results[i]["reasoning_label_probs"]    = label_probs

        if save_every > 0 and (i + 1) % save_every == 0:
            flush_fn()

    flush_fn()


def run_mode_stochastic(model, processor, entries: list[dict], base_dir: str,
                        max_new_tokens: int, data_loading: str,
                        n_samples: int, results: list[dict],
                        save_every: int, flush_fn) -> None:
    """Mode 3: N stochastic reasoning samples (temperature=0.6, top_p=0.95, top_k=20)."""
    for i, entry in enumerate(tqdm(entries, desc=f"Mode 3/4 — Stochastic (N={n_samples})")):
        stoc_responses:     list[str] = []
        stoc_answers:       list[str] = []
        stoc_trace_tokens:  list[int] = []

        for _ in range(n_samples):
            try:
                resp = infer_stochastic(
                    model, processor, entry, base_dir, max_new_tokens, data_loading
                )
            except Exception as exc:
                print(f"\n[WARN] Stochastic mode entry {i}: {exc.__class__.__name__}: {exc}")
                resp = ""

            trace = extract_think_trace(resp)
            stoc_responses.append(resp)
            stoc_answers.append(extract_answer(resp))
            stoc_trace_tokens.append(count_tokens_precisely(processor, trace) if trace else 0)

        results[i]["stochastic_responses"]    = stoc_responses
        results[i]["stochastic_answers"]      = stoc_answers
        results[i]["stochastic_trace_tokens"] = stoc_trace_tokens

        if save_every > 0 and (i + 1) % save_every == 0:
            flush_fn()

    flush_fn()


def run_mode_para_reasoning(model, processor, entries: list[dict],
                            para_entries: list[dict], base_dir: str,
                            max_new_tokens: int, data_loading: str,
                            results: list[dict], save_every: int, flush_fn) -> None:
    """Mode 4: greedy reasoning on paraphrased inputs."""
    for i, para_entry in enumerate(tqdm(para_entries, desc="Mode 4/4 — Para-Reasoning")):
        try:
            response, _ = infer_with_logit_capture(
                model, processor, para_entry, base_dir,
                thinking=True, max_new_tokens=max_new_tokens,
                data_loading=data_loading,
                label_token_ids={},   # no logit capture needed
            )
        except Exception as exc:
            print(f"\n[WARN] Para-reasoning mode entry {i}: {exc.__class__.__name__}: {exc}")
            traceback.print_exc()
            response = ""

        results[i]["para_reasoning_response"] = response
        results[i]["para_reasoning_answer"]   = extract_answer(response)

        if save_every > 0 and (i + 1) % save_every == 0:
            flush_fn()

    flush_fn()


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    # ── Load source JSONL ─────────────────────────────────────────────────────
    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        entries = [json.loads(ln) for ln in f if ln.strip()]

    for i, e in enumerate(entries):
        e.setdefault("_orig_idx", i)

    if args.max_samples:
        entries = entries[: args.max_samples]

    # ── Load paraphrased JSONL, align by _orig_idx ────────────────────────────
    para_map: dict[int, dict] = {}
    if args.para_input_jsonl and os.path.exists(args.para_input_jsonl):
        with open(args.para_input_jsonl, "r", encoding="utf-8") as f:
            for ln in f:
                if ln.strip():
                    e = json.loads(ln)
                    para_map[e.get("_orig_idx", -1)] = e
        print(f"Loaded {len(para_map)} paraphrased entries.")
    else:
        print("[WARN] No paraphrased JSONL provided — para-reasoning mode will use original inputs.")

    para_entries = [
        para_map.get(e.get("_orig_idx", i), e)
        for i, e in enumerate(entries)
    ]

    # ── Infer class label set ─────────────────────────────────────────────────
    if args.multilabel:
        labels = sorted({
            lbl.strip().lower()
            for e in entries
            for lbl in e["answer"].split(",")
            if lbl.strip()
        })
    else:
        labels = sorted({e["answer"].strip().lower() for e in entries if e.get("answer")})

    print(f"Class labels ({len(labels)}): {labels}")

    base_dir = os.path.abspath(
        args.data_base_dir if args.data_base_dir else os.path.dirname(args.input_jsonl)
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.output_jsonl)), exist_ok=True)

    # ── Load model once ───────────────────────────────────────────────────────
    model, processor = load_model(args.model)
    label_token_ids  = get_label_token_ids(processor, labels)

    # ── Initialise result records ─────────────────────────────────────────────
    results = [
        {
            "answer":   e.get("answer", ""),
            "dataset":  e.get("dataset", os.path.basename(args.input_jsonl).replace(".jsonl", "")),
            "_orig_idx": e.get("_orig_idx", i),
        }
        for i, e in enumerate(entries)
    ]

    def flush():
        with open(args.output_jsonl, "w", encoding="utf-8") as wf:
            for r in results:
                wf.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ── Run four modes sequentially ───────────────────────────────────────────
    run_mode_direct(
        model, processor, entries, base_dir,
        args.max_new_tokens, args.data_loading,
        label_token_ids, results, args.save_every, flush,
    )
    torch.cuda.empty_cache()

    run_mode_reasoning(
        model, processor, entries, base_dir,
        args.max_new_tokens, args.data_loading,
        label_token_ids, results, args.save_every, flush,
    )
    torch.cuda.empty_cache()

    run_mode_stochastic(
        model, processor, entries, base_dir,
        args.max_new_tokens, args.data_loading,
        args.n_stochastic, results, args.save_every, flush,
    )
    torch.cuda.empty_cache()

    run_mode_para_reasoning(
        model, processor, entries, para_entries, base_dir,
        args.max_new_tokens, args.data_loading,
        results, args.save_every, flush,
    )

    print(f"\nSaved {len(results)} entries → {args.output_jsonl}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reasoning evaluation: 4-mode inference")

    parser.add_argument("--model",            required=True)
    parser.add_argument("--input_jsonl",      required=True)
    parser.add_argument("--para_input_jsonl", default=None,
                        help="Paraphrased JSONL (from paraphrase_inputs.py). "
                             "If absent, original inputs are used for mode 4.")
    parser.add_argument("--output_jsonl",     required=True)
    parser.add_argument("--data_base_dir",    default=None)
    parser.add_argument("--max_new_tokens",   type=int, default=1024)
    parser.add_argument("--n_stochastic",     type=int, default=3,
                        help="Number of stochastic reasoning samples per entry")
    parser.add_argument("--data_loading",     default="verl_style",
                        choices=["default", "verl_style"])
    parser.add_argument("--multilabel",       action="store_true",
                        help="Treat answer field as comma-separated multilabel")
    parser.add_argument("--max_samples",      type=int, default=None)
    parser.add_argument("--save_every",       type=int, default=50)

    # W&B (unused here — metrics are logged by compute_reasoning_metrics.py)
    parser.add_argument("--wandb_project",  default=None)
    parser.add_argument("--wandb_run_id",   default=None)
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--wandb_entity",   default=None)

    args = parser.parse_args()
    main(args)
