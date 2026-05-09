# Copyright 2025 Bytedance Ltd. and/or its affiliates
import logging
import os

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def merge_consecutive_same_role_messages(messages: list[dict]) -> list[dict]:
    """Merge adjacent messages with the same role.

    Some chat templates (e.g. Gemma-3) reject consecutive same-role messages —
    typically when a dataset emits ``[system, user]`` and an upstream step has
    already folded ``system`` into a ``user`` message, leaving ``[user, user]``.
    Merging joins their content so the template sees a single message per turn.

    Handles both string ``content`` and the multimodal list-of-parts form.
    """
    if not messages:
        return messages
    merged = [dict(messages[0])]
    for msg in messages[1:]:
        prev = merged[-1]
        if msg["role"] != prev["role"]:
            merged.append(dict(msg))
            continue
        prev_c, cur_c = prev.get("content", ""), msg.get("content", "")
        if isinstance(prev_c, str) and isinstance(cur_c, str):
            sep = "\n\n" if prev_c and cur_c else ""
            prev["content"] = prev_c + sep + cur_c
        else:
            prev_list = prev_c if isinstance(prev_c, list) else [{"type": "text", "text": prev_c}]
            cur_list = cur_c if isinstance(cur_c, list) else [{"type": "text", "text": cur_c}]
            prev["content"] = list(prev_list) + list(cur_list)
    return merged


def initialize_system_prompt(tokenizer, **apply_chat_template_kwargs) -> list[int]:
    """
    Initialize system prompt tokens for chat templates that support them.

    Args:
        tokenizer: The tokenizer with a chat template
        **apply_chat_template_kwargs: Additional arguments for apply_chat_template

    Returns:
        List of token IDs for the system prompt, or empty list if not supported
    """
    try:
        token1 = tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}], add_generation_prompt=False, tokenize=True
        )
        token2 = tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}] * 2, add_generation_prompt=False, tokenize=True
        )
    except Exception:
        # Some templates (e.g. Gemma-3) reject two consecutive user messages.
        # Fall back to no system prompt — these models don't auto-inject one.
        return []
    # get system prompt tokens
    system_prompt = token1[: -(len(token2) - len(token1))]
    return system_prompt


def extract_system_prompt_and_generation(tokenizer):
    token1 = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}], add_generation_prompt=False, tokenize=True
    )
    token2 = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}] * 2, add_generation_prompt=False, tokenize=True
    )
    # get system prompt tokens
    system_prompt = token1[: -(len(token2) - len(token1))]
    # get generate prompt tokens
    token3 = tokenizer.apply_chat_template([{"role": "user", "content": ""}], add_generation_prompt=True, tokenize=True)
    generate_prompt = token3[len(token1) :]

    return system_prompt, generate_prompt
