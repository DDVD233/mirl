# ---- utils: log modality budgets straight from batch_dict (pre-repeat, pre-DataProto) ----
import numpy as np, wandb
from collections import defaultdict

def _normalize_bd_store_to_list_of_dicts(bd_store, N):
    """Always return list[dict] of length N."""
    import numpy as np
    # numpy array of dicts -> list
    if isinstance(bd_store, np.ndarray):
        try:
            bd_store = bd_store.tolist()
        except Exception:
            pass
    # list of dicts
    if isinstance(bd_store, list):
        return list(bd_store)[:N]
    # dict-of-arrays (possibly with nested dict-of-arrays under 'placeholders')
    if isinstance(bd_store, dict):
        def _ith(v, i):
            if hasattr(v, "cpu"):
                v = v.cpu().numpy()
            if isinstance(v, (list, tuple, np.ndarray)):
                x = v[i]
            else:
                x = v
            if hasattr(x, "item"):
                x = x.item()
            return x
        out = []
        for i in range(N):
            d = {}
            for k, v in bd_store.items():
                if isinstance(v, dict):
                    d[k] = {kk: _ith(vv, i) for kk, vv in v.items()}
                else:
                    d[k] = _ith(v, i)
            out.append(d)
        return out
    return []

def _infer_batch_size_from_batch_dict(batch_dict):
    # Prefer the same source you’ll read: modality_token_breakdown
    bd = batch_dict.get("modality_token_breakdown", None)
    if bd is None:
        # fallback: try modality_signature
        sigs = batch_dict.get("modality_signature", None)
        if sigs is not None:
            return len(sigs)
        # last resort: try any list-like field
        for v in batch_dict.values():
            try:
                return len(v)
            except Exception:
                continue
        return 0
    import numpy as np
    if isinstance(bd, (list, np.ndarray)):
        return len(bd)
    if isinstance(bd, dict):
        # use first non-dict value’s length
        for v in bd.values():
            if isinstance(v, dict):
                # nested dict: take its first value’s length
                for vv in v.values():
                    try: return len(vv)
                    except Exception: pass
            else:
                try: return len(v)
                except Exception: pass
    return 0

def _vec_from_bd_list(bd_list, key, sub=None):
    xs = []
    for d in bd_list:
        val = d.get(key, 0)
        if sub is not None:
            val = (val or {}).get(sub, 0)
        try:
            xs.append(float(val))
        except Exception:
            xs.append(0.0)
    return xs

def log_modality_budgets(batch_dict, step):
    bd_store = batch_dict.get("modality_token_breakdown", None)
    if bd_store is None:
        return
    N = _infer_batch_size_from_batch_dict(batch_dict)
    if N <= 0:
        return

    bd_list = _normalize_bd_store_to_list_of_dicts(bd_store, N)
    if not bd_list:
        return

    sigs = batch_dict.get("modality_signature", None)
    if sigs is not None and hasattr(sigs, "tolist"):
        sigs = sigs.tolist()

    input_ids_len = _vec_from_bd_list(bd_list, "input_ids_len")          # pre-truncation
    text_tokens   = _vec_from_bd_list(bd_list, "text_tokens_est")        # pre-truncation
    img_kv        = _vec_from_bd_list(bd_list, "image_kv_tokens_est")
    vid_kv        = _vec_from_bd_list(bd_list, "video_kv_tokens_est")
    aud_secs      = _vec_from_bd_list(bd_list, "audio_secs")
    ph_img        = _vec_from_bd_list(bd_list, "placeholders", "image_ph")
    ph_vid        = _vec_from_bd_list(bd_list, "placeholders", "video_ph")
    ph_aud        = _vec_from_bd_list(bd_list, "placeholders", "audio_ph")

    p50 = lambda x: float(np.median(x)) if x else 0.0
    p95 = lambda x: float(np.percentile(x, 95)) if x else 0.0
    pmin = lambda x: float(np.min(x)) if x else 0.0
    pmax = lambda x: float(np.max(x)) if x else 0.0

    # Batch-level summaries
    wandb.log({
        "budget/input_ids/p50": p50(input_ids_len),
        "budget/input_ids/p95": p95(input_ids_len),
        "budget/input_ids/min": pmin(input_ids_len),
        "budget/input_ids/max": pmax(input_ids_len),
        "budget/text/p50":      p50(text_tokens),
        "budget/text/p95":      p95(text_tokens),
        "budget/img_kv/p50":    p50(img_kv),
        "budget/vid_kv/p50":    p50(vid_kv),
        "budget/audio_secs/p50":p50(aud_secs),
        "budget/ph/image/p50":  p50(ph_img),
        "budget/ph/video/p50":  p50(ph_vid),
        "budget/ph/audio/p50":  p50(ph_aud),
    }, step=step)

    # Per-signature medians
    if sigs is not None and len(sigs) == len(bd_list):
        buckets = defaultdict(lambda: {"L":[], "T":[], "Ikv":[], "Vkv":[], "As":[]})
        for s, d in zip(sigs, bd_list):
            buckets[s]["L"].append(float(d.get("input_ids_len", 0)))
            buckets[s]["T"].append(float(d.get("text_tokens_est", 0)))
            buckets[s]["Ikv"].append(float(d.get("image_kv_tokens_est", 0)))
            buckets[s]["Vkv"].append(float(d.get("video_kv_tokens_est", 0)))
            buckets[s]["As"].append(float(d.get("audio_secs", 0.0)))
        payload = {}
        for s, b in buckets.items():
            payload.update({
                f"budget_by_sig/{s}/n": len(b["L"]),
                f"budget_by_sig/{s}/input_ids_p50": p50(b["L"]),
                f"budget_by_sig/{s}/text_p50":      p50(b["T"]),
                f"budget_by_sig/{s}/img_kv_p50":    p50(b["Ikv"]),
                f"budget_by_sig/{s}/vid_kv_p50":    p50(b["Vkv"]),
                f"budget_by_sig/{s}/audio_secs_p50":p50(b["As"]),
            })
        if payload:
            wandb.log(payload, step=step)

    # Optional histograms
    try:
        wandb.log({
            "budget/hist_input_ids": wandb.Histogram(input_ids_len),
            "budget/hist_img_kv":    wandb.Histogram(img_kv),
            "budget/hist_vid_kv":    wandb.Histogram(vid_kv),
        }, step=step)
    except Exception:
        pass
# ---- end utils ----
