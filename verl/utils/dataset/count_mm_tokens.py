from typing import Dict, Any, Optional

MEDIA_MARKER_STRINGS = [
            "<image>", "<video>", "<audio>",
            "<|vision_start|>", "<|vision_end|>",
            "<|video_start|>", "<|video_end|>",
            "<|image_start|>", "<|image_end|>",
            "<|audio_start|>", "<|audio_end|>",
        ]

def get_media_token_ids(tokenizer=None):
    _vocab = tokenizer.get_vocab()
    _media_token_ids = {k: tokenizer.convert_tokens_to_ids(k)
            for k in MEDIA_MARKER_STRINGS if k in _vocab}
    
    return _media_token_ids

def count_grid_tokens(grid_thw):
    """Sum product(T*H*W) over grids; robust to tensor/list/tuple nesting."""
    if grid_thw is None:
        return 0
    arr = grid_thw
    if hasattr(arr, "cpu"):  # torch tensor
        arr = arr.cpu().numpy().tolist()
    if isinstance(arr, (list, tuple)) and arr and not isinstance(arr[0], (list, tuple)):
        arr = [arr]
    total = 0
    for g in arr or []:
        prod = 1
        for x in g:
            try:
                prod *= int(x)
            except Exception:
                prod *= int(float(x))
        total += int(prod)
    return int(total)

def count_placeholders(input_ids_1d, _media_token_ids):
    counts = {"image_ph": 0, "video_ph": 0, "audio_ph": 0, "vision_start": 0, "vision_end": 0}
    # add any found ids to buckets
    idset = set(input_ids_1d.tolist())
    # coarse buckets
    if "<image>" in _media_token_ids:  counts["image_ph"] += input_ids_1d.tolist().count(_media_token_ids["<image>"])
    if "<video>" in _media_token_ids:  counts["video_ph"] += input_ids_1d.tolist().count(_media_token_ids["<video>"])
    if "<audio>" in _media_token_ids:  counts["audio_ph"] += input_ids_1d.tolist().count(_media_token_ids["<audio>"])
    if "<|vision_start|>" in _media_token_ids:
        counts["vision_start"] += input_ids_1d.tolist().count(_media_token_ids["<|vision_start|>"])
    if "<|vision_end|>" in _media_token_ids:
        counts["vision_end"] += input_ids_1d.tolist().count(_media_token_ids["<|vision_end|>"])
    return counts

# --- END: per-sample modality token breakdown ---

def compute_modality_token_breakdown(
    model_inputs: Dict[str, Any],
    tokenizer,
    row_dict: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compute a clean, comparable budget breakdown for one sample:
      - input_ids_len: total text tokens (incl. placeholders)
      - text_tokens_est: input_ids minus placeholder/control tokens
      - placeholders: counts by type (image/video/audio/vision_start/vision_end)
      - image_kv_tokens_est / video_kv_tokens_est: grid-based visual token estimates
      - audio_secs: summed seconds from provided audio arrays (if present in row_dict['multi_modal_data'])
    """
    media_token_ids = get_media_token_ids(tokenizer)

    # visual KV-token estimates via grids
    image_kv_tokens = count_grid_tokens(model_inputs.get("image_grid_thw"))
    video_kv_tokens = count_grid_tokens(model_inputs.get("video_grid_thw"))

    # peek input_ids BEFORE popping them
    ids = model_inputs.get("input_ids")

    # print("Ids are", ids)

    if ids is not None and len(ids) > 0:
        ids1d = ids[0]
        ph_counts = count_placeholders(ids1d, media_token_ids)
        input_ids_len = int(len(ids1d))
        media_placeholder_total = int(sum(ph_counts.values()))
        text_token_est = int(input_ids_len - media_placeholder_total)
    else:
        ph_counts = {"image_ph": 0, "video_ph": 0, "audio_ph": 0, "vision_start": 0, "vision_end": 0}
        input_ids_len = 0
        text_token_est = 0

    # audio proxy: total seconds from multi_modal_data["audio"] = [(np_array, sr), ...]
    audio_secs_total = 0.0
    mmd = row_dict.get("multi_modal_data") or {}
    for item in (mmd.get("audio") or []):
        try:
            arr, sr = item
            audio_secs_total += float(len(arr)) / float(sr)
        except Exception:
            # ignore malformed entries
            pass

    return {
        "input_ids_len": int(input_ids_len),
        "text_tokens_est": int(text_token_est),
        "placeholders": {k: int(v) for k, v in ph_counts.items()},
        "image_kv_tokens_est": int(image_kv_tokens),
        "video_kv_tokens_est": int(video_kv_tokens),
        "audio_secs": float(round(audio_secs_total, 6)),
    }