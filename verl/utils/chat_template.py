# Copyright 2025 Bytedance Ltd. and/or its affiliates
import logging
import os

from transformers import PreTrainedTokenizerBase, ProcessorMixin

from verl.utils.tokenizer import normalize_token_ids

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _render(tokenizer, messages) -> list[int]:
    return normalize_token_ids(
        tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=True)
    )


def _system_prompt_tokens(tokenizer) -> list[int]:
    """Derive the per-conversation system-prompt token prefix.

    The prefix is whatever the chat template emits before the first user turn
    (typically just BOS, sometimes a literal system header). We compute it by
    finding the size of one user turn and subtracting that from a single-user
    rendering.

    Two probe strategies, in order:

      1. ``[user, user]`` minus ``[user]`` — works for templates that allow
         repeated user turns (Qwen, Llama, Mistral, ...).
      2. ``[user, assistant, user]`` minus ``[user, assistant]`` — works for
         templates that enforce strict user/assistant alternation (Gemma 3).

    If both fail, return an empty list (no truncation downstream is fine).
    """
    msg_user = {"role": "user", "content": ""}
    msg_asst = {"role": "assistant", "content": ""}

    one_user = _render(tokenizer, [msg_user])

    # Strategy 1: two consecutive user turns.
    try:
        two_user = _render(tokenizer, [msg_user, msg_user])
        user_turn_len = len(two_user) - len(one_user)
        if user_turn_len > 0:
            return one_user[:-user_turn_len]
    except Exception:
        pass

    # Strategy 2: alternating user/assistant — for Gemma 3 etc.
    try:
        ua = _render(tokenizer, [msg_user, msg_asst])
        uau = _render(tokenizer, [msg_user, msg_asst, msg_user])
        user_turn_len = len(uau) - len(ua)
        if user_turn_len > 0 and user_turn_len <= len(one_user):
            return one_user[:-user_turn_len]
    except Exception as e:
        logger.debug(
            "system-prompt probe (alternating) failed (%s); assuming empty.",
            type(e).__name__,
        )

    return []


def initialize_system_prompt(tokenizer, **apply_chat_template_kwargs) -> list[int]:
    """
    Initialize system prompt tokens for chat templates that support them.

    Args:
        tokenizer: The tokenizer with a chat template
        **apply_chat_template_kwargs: Additional arguments for apply_chat_template

    Returns:
        List of token IDs for the system prompt, or empty list if not supported
    """
    return _system_prompt_tokens(tokenizer)


def extract_system_prompt_and_generation(tokenizer):
    token1 = normalize_token_ids(
        tokenizer.apply_chat_template([{"role": "user", "content": ""}], add_generation_prompt=False, tokenize=True)
    )
    system_prompt = _system_prompt_tokens(tokenizer)
    # get generate prompt tokens
    token3 = normalize_token_ids(
        tokenizer.apply_chat_template([{"role": "user", "content": ""}], add_generation_prompt=True, tokenize=True)
    )
    generate_prompt = token3[len(token1) :]

    return system_prompt, generate_prompt


def apply_chat_template(
    processor: PreTrainedTokenizerBase | ProcessorMixin,
    messages: list[dict],
    *,
    tokenize: bool = True,
    add_generation_prompt: bool = True,
    tools=None,
    return_dict: bool = False,
    **kwargs,
) -> list[int] | str:
    """apply_chat_template to messages with special attention to template requiring
    at least one user message, e.g. Qwen3.5.

    Args:
        processor: tokenizer or processor.
        messages: list[dict], messages.
        tokenize: bool, whether to tokenize the output.
        add_generation_prompt: bool, whether to add generation prompt.
        tools: list[dict], tools schema.
        return_dict: bool, whether to return a dict.
        **kwargs: additional arguments for apply_chat_template.

    Returns:
        list[int] | str: tokenized ids or text string.
    """
    try:
        return processor.apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
            return_dict=return_dict,
            **kwargs,
        )
    except Exception:
        # Qwen3.5 apply_chat_template needs messages with at least one user message
        dummy_user_message = [{"role": "user", "content": [{"type": "text", "text": ""}]}]
        dummy_user_prefix = processor.apply_chat_template(
            dummy_user_message,
            tokenize=tokenize,
            add_generation_prompt=False,
            tools=tools,
            return_dict=return_dict,
            **kwargs,
        )
        output = processor.apply_chat_template(
            dummy_user_message + messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
            return_dict=return_dict,
            **kwargs,
        )

        if not tokenize:  # tokenize=False
            return output[len(dummy_user_prefix) :]
        elif not return_dict:  # tokenize=True and return_dict=False
            if isinstance(output[0], list):  # transformers>=5
                assert len(output) == 1, "output must be a list[int] or list[list[int]]"
                dummy_user_prefix = dummy_user_prefix[0]
                output = output[0]
            return output[len(dummy_user_prefix) :]
        else:  # tokenize=True and return_dict=True and return_tensors="pt"
            dummy_user_prefix = dict(dummy_user_prefix)
            output = dict(output)
            prefix_len = dummy_user_prefix["input_ids"].shape[1]
            output["input_ids"] = output["input_ids"][:, prefix_len:]
            output["attention_mask"] = output["attention_mask"][:, prefix_len:]
            if "mm_token_type_ids" in output:
                output["mm_token_type_ids"] = output["mm_token_type_ids"][:, prefix_len:]
            return output
