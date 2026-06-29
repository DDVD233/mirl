"""
Standalone utilities extracted from verl for use in dataset.
Eliminates the dependency on verl.utils.torch_functional and verl.utils.model.
"""
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict


def pad_sequence_to_length(tensors, max_seq_len, pad_token_id, left_pad=False):
    """Pad a 2D tensor in the last dim to max_seq_length."""
    if tensors.shape[-1] >= max_seq_len:
        return tensors
    pad_tuple = (max_seq_len - tensors.shape[-1], 0) if left_pad else (0, max_seq_len - tensors.shape[-1])
    return F.pad(tensors, pad_tuple, "constant", pad_token_id)


def postprocess_data(input_ids, attention_mask, max_length, pad_token_id, left_pad=True, truncation="error"):
    """Pad or truncate input_ids and attention_mask to max_length."""
    assert truncation in ["left", "right", "middle", "error"]
    assert input_ids.ndim == 2

    seq_len = input_ids.shape[-1]
    if seq_len < max_length:
        input_ids = pad_sequence_to_length(input_ids, max_seq_len=max_length, pad_token_id=pad_token_id, left_pad=left_pad)
        attention_mask = pad_sequence_to_length(attention_mask, max_seq_len=max_length, pad_token_id=0, left_pad=left_pad)
    elif seq_len > max_length:
        if truncation == "left":
            input_ids = input_ids[:, -max_length:]
            attention_mask = attention_mask[:, -max_length:]
        elif truncation == "right":
            input_ids = input_ids[:, :max_length]
            attention_mask = attention_mask[:, :max_length]
        elif truncation == "middle":
            left_half = max_length // 2
            right_half = max_length - left_half
            input_ids = torch.cat([input_ids[:, :left_half], input_ids[:, -right_half:]], dim=-1)
            attention_mask = torch.cat([attention_mask[:, :left_half], attention_mask[:, -right_half:]], dim=-1)
        elif truncation == "error":
            raise RuntimeError(f"Prompt length {seq_len} is longer than max_length {max_length}.")
    return input_ids, attention_mask


def compute_position_id_with_mask(mask):
    """Compute position IDs from an attention mask."""
    return torch.clip(torch.cumsum(mask, dim=-1) - 1, min=0, max=None)


def collate_fn(data_list: list[dict]) -> dict:
    """
    Collate a batch of sample dicts into batched tensors and arrays.
    Tensor entries are stacked; non-tensor entries become np.ndarray of dtype object.
    """
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.fromiter(val, dtype=object, count=len(val))

    return {**tensors, **non_tensors}


# Native-video/image tensors that Qwen3-VL consumes as a single batch-flattened block:
# pixel patches are concatenated along dim 0 across the whole batch, and the *_grid_thw
# tensors ([num_visuals, 3]) tell the model how to slice them back per sample.
_VL_MM_CAT_KEYS = ("pixel_values_videos", "video_grid_thw", "pixel_values", "image_grid_thw")


def vl_collate_fn(data_list: list[dict]) -> dict:
    """
    collate_fn + native-video/image merging for Qwen3-VL.

    The per-sample processor outputs live in each row's `multi_modal_inputs` dict (an
    object array after the base collate). Qwen3-VL expects the visual tensors NOT stacked
    but concatenated along dim 0 across the batch (variable patch counts per sample), with
    the grid_thw tensors concatenated in the same sample order so placeholder tokens in
    `input_ids` line up with their embeddings. Samples without video/image contribute
    nothing, so mixed text/video batches work. No-op when no visual tensors are present.
    """
    batch = collate_fn(data_list)
    mm_inputs = batch.get("multi_modal_inputs", None)
    if mm_inputs is None:
        return batch

    merged = defaultdict(list)
    for d in mm_inputs:  # object array of per-sample dicts (in batch order)
        if not isinstance(d, dict):
            continue
        for k in _VL_MM_CAT_KEYS:
            v = d.get(k, None)
            if isinstance(v, torch.Tensor):
                merged[k].append(v)
    for k, lst in merged.items():
        if lst:
            batch[k] = torch.cat(lst, dim=0)
    return batch
