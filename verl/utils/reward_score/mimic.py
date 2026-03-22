"""Reward scoring for MIMIC clinical QA.

Supports two formats:
- Multiple choice (q3, q5, q6): exact string match on \\boxed{} content.
- Long response (q1, q2): BioBERT embedding cosine similarity between
  \\boxed{} content and ground truth. Runs on CPU since the reward model
  may not have GPU access.
"""

import re

# Long-response qa_types (free-text diagnosis answers)
LONG_RESPONSE_TYPES = {"1", "2"}


def extract_boxed_answer(predict_str: str) -> str | None:
    """Extract the content inside the last \\boxed{...} in the prediction."""
    idx = predict_str.rfind("\\boxed{")
    if idx < 0:
        return None
    depth = 0
    i = idx + len("\\boxed{") - 1
    while i < len(predict_str):
        if predict_str[i] == "{":
            depth += 1
        elif predict_str[i] == "}":
            depth -= 1
            if depth == 0:
                return predict_str[idx + len("\\boxed{"):i]
        i += 1
    return None


def format_reward(predict_str: str) -> float:
    """Check for <think>...</think>...\\boxed{...} format."""
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    return 1.0 if re.fullmatch(pattern, predict_str) else 0.0


# ---------------------------------------------------------------------------
# BioBERT embedding similarity (CPU-only)
# ---------------------------------------------------------------------------


def _get_biobert():
    import torch
    from transformers import AutoModel, AutoTokenizer

    model_name = "dmis-lab/biobert-base-cased-v1.2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Force loading on CPU with no device_map inference to avoid meta tensors
    # in Ray worker environments
    with torch.device("cpu"):
        model = AutoModel.from_pretrained(
            model_name, device_map=None, low_cpu_mem_usage=False
        )
    model.eval()
    return tokenizer, model


def _embed(text: str):
    """Return a mean-pooled BioBERT embedding on CPU."""
    import torch

    tokenizer, model = _get_biobert()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(torch.device("cpu")) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    # Mean pool over token dimension
    mask = inputs["attention_mask"].unsqueeze(-1).float()
    embedding = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
    return embedding.squeeze(0)


def embedding_similarity(pred_text: str, gt_text: str) -> float:
    """Cosine similarity between BioBERT embeddings of pred and gt."""
    import time
    import torch

    if not pred_text or not gt_text:
        print(f"[mimic reward] embedding_similarity: empty input, pred={repr(pred_text[:100])}, gt={repr(gt_text[:100])}, sim=0.0")
        return 0.0
    t0 = time.time()
    pred_emb = _embed(pred_text)
    gt_emb = _embed(gt_text)
    cos_sim = torch.nn.functional.cosine_similarity(pred_emb.unsqueeze(0), gt_emb.unsqueeze(0)).item()
    # Clamp to [0, 1] since negative similarity is not meaningful here
    result = max(0.0, cos_sim)
    elapsed = time.time() - t0
    print(f"[mimic reward] embedding_similarity: sim={result:.4f}, time={elapsed:.3f}s, pred={repr(pred_text[:80])}, gt={repr(gt_text[:80])}")
    return result


# ---------------------------------------------------------------------------
# Multiple choice scoring
# ---------------------------------------------------------------------------
def mc_accuracy(predict_str: str, ground_truth: str) -> float:
    """Exact match of boxed answer against ground truth (case-insensitive)."""
    boxed = extract_boxed_answer(predict_str)
    if boxed is None:
        return 0.0
    return 1.0 if boxed.strip().upper() == ground_truth.strip().upper() else 0.0


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def compute_score(
    predict_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    format_weight: float = 0.1,
) -> dict:
    """Compute reward for MIMIC clinical QA.

    For multiple choice (q3/q5/q6):
        score = (1 - format_weight) * acc + format_weight * format

    For long response (q1/q2):
        score = (1 - format_weight) * similarity + format_weight * format

    Returns dict with score and diagnostic metrics.
    """
    qa_type = extra_info.get("qa_type", "") if extra_info else ""
    fmt = format_reward(predict_str)

    if qa_type in LONG_RESPONSE_TYPES:
        boxed = extract_boxed_answer(predict_str)
        sim = embedding_similarity(boxed or "", ground_truth)
        acc = 0.0
        score = (1.0 - format_weight) * sim + format_weight * fmt
    else:
        acc = mc_accuracy(predict_str, ground_truth)
        sim = 0.0
        score = (1.0 - format_weight) * acc + format_weight * fmt

    return {
        "score": score,
        "acc": acc,
        "similarity": sim,
        "format": fmt,
    }