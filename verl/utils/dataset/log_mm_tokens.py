# ---- utils: log modality budgets to W&B ----
import numpy as np, wandb

def _len_from_batch(batch):
    am = batch.batch.get("attention_mask", None)
    return int(am.shape[0]) if am is not None else len(next(iter(batch.batch.values())))

def _extract_vecs(bd_store, N, key, sub=None):
    """
    Supports either:
      - list of dicts (one dict per sample), or
      - dict of arrays (vectorized per key), including nested dict for "placeholders".
    Returns a Python list length N with ints/floats.
    """
    out = [0] * N
    if isinstance(bd_store, list):
        for i in range(min(N, len(bd_store))):
            v = bd_store[i].get(key, 0)
            if sub is not None:
                v = (v or {}).get(sub, 0)
            out[i] = v if isinstance(v, (int, float)) else float(v)
        return out

    # dict-of-arrays path
    if sub is None:
        arr = bd_store.get(key, None)
        if arr is None:
            return out
        # arr may be np.ndarray/torch tensor/list
        if hasattr(arr, "cpu"):
            arr = arr.cpu().numpy()
        arr = list(arr)
        for i in range(min(N, len(arr))):
            out[i] = float(arr[i])
        return out
    else:
        ph = bd_store.get(key, {})
        arr = ph.get(sub, None)
        if arr is None:
            return out
        if hasattr(arr, "cpu"):
            arr = arr.cpu().numpy()
        arr = list(arr)
        for i in range(min(N, len(arr))):
            out[i] = float(arr[i])
        return out

def log_modality_budgets(batch, step):
    """Aggregate per-sample modality budgets and push compact stats to W&B."""
    bd_store = batch.non_tensor_batch.get("modality_token_breakdown", None)
    if bd_store is None:
        return

    N = _len_from_batch(batch)
    sigs = batch.non_tensor_batch.get("modality_signature", None)
    # Normalize signatures to list[str]
    if sigs is not None:
        if hasattr(sigs, "tolist"):
            sigs = sigs.tolist()
        sigs = list(sigs)

    # pull vectors
    input_ids_len = _extract_vecs(bd_store, N, "input_ids_len")
    text_tokens   = _extract_vecs(bd_store, N, "text_tokens_est")
    img_kv        = _extract_vecs(bd_store, N, "image_kv_tokens_est")
    vid_kv        = _extract_vecs(bd_store, N, "video_kv_tokens_est")
    aud_secs      = _extract_vecs(bd_store, N, "audio_secs")
    ph_img        = _extract_vecs(bd_store, N, "placeholders", "image_ph")
    ph_vid        = _extract_vecs(bd_store, N, "placeholders", "video_ph")
    ph_aud        = _extract_vecs(bd_store, N, "placeholders", "audio_ph")

    def _p50(x): return float(np.median(x)) if x else 0.0
    def _p95(x): return float(np.percentile(x, 95)) if x else 0.0
    def _mx(x):  return float(np.max(x)) if x else 0.0
    def _mn(x):  return float(np.min(x)) if x else 0.0

    # batch-level summary
    wandb.log({
        "budget/input_ids/p50": _p50(input_ids_len),
        "budget/input_ids/p95": _p95(input_ids_len),
        "budget/input_ids/min": _mn(input_ids_len),
        "budget/input_ids/max": _mx(input_ids_len),

        "budget/text/p50": _p50(text_tokens),
        "budget/text/p95": _p95(text_tokens),

        "budget/img_kv/p50": _p50(img_kv),
        "budget/vid_kv/p50": _p50(vid_kv),
        "budget/audio_secs/p50": _p50(aud_secs),

        "budget/ph/image/p50": _p50(ph_img),
        "budget/ph/video/p50": _p50(ph_vid),
        "budget/ph/audio/p50": _p50(ph_aud),
    }, step=step)

    # per-signature medians (helps you spot imbalance across modality combos)
    if sigs is not None and len(sigs) == N:
        per_sig = {}
        for s, L, T, Ikv, Vkv, As in zip(sigs, input_ids_len, text_tokens, img_kv, vid_kv, aud_secs):
            d = per_sig.setdefault(s, {"L": [], "T": [], "Ikv": [], "Vkv": [], "As": []})
            d["L"].append(L); d["T"].append(T); d["Ikv"].append(Ikv); d["Vkv"].append(Vkv); d["As"].append(As)

        payload = {}
        for s, d in per_sig.items():
            payload.update({
                f"budget_by_sig/{s}/n": len(d["L"]),
                f"budget_by_sig/{s}/input_ids_p50": _p50(d["L"]),
                f"budget_by_sig/{s}/text_p50": _p50(d["T"]),
                f"budget_by_sig/{s}/img_kv_p50": _p50(d["Ikv"]),
                f"budget_by_sig/{s}/vid_kv_p50": _p50(d["Vkv"]),
                f"budget_by_sig/{s}/audio_secs_p50": _p50(d["As"]),
            })
        if payload:
            wandb.log(payload, step=step)

    # light-weight histograms (optional; comment out if noisy)
    try:
        wandb.log({
            "budget/hist_input_ids": wandb.Histogram(input_ids_len),
            "budget/hist_img_kv": wandb.Histogram(img_kv),
            "budget/hist_vid_kv": wandb.Histogram(vid_kv),
        }, step=step)
    except Exception:
        pass
# ---- end utils ----
