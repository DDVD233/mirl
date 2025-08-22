# ---- utils: log modality token budgets from batch_dict (mean/min/max) ----
import numpy as np
import wandb
from collections import defaultdict

def _sanitize_tag(s):
    # keep W&B keys clean and avoid accidental nesting
    return str(s).replace("/", "_").replace(" ", "_")

def _normalize_bd_store_to_list_of_dicts(bd_store, N):
    """Return list[dict] of length N (handles list, np.array(dtype=object), dict-of-arrays)."""
    if isinstance(bd_store, np.ndarray):
        try:
            bd_store = bd_store.tolist()
        except Exception:
            pass
    if isinstance(bd_store, list):
        return list(bd_store)[:N]
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
                if isinstance(v, dict):  # e.g., placeholders dict-of-arrays
                    d[k] = {kk: _ith(vv, i) for kk, vv in v.items()}
                else:
                    d[k] = _ith(v, i)
            out.append(d)
        return out
    return []

def _infer_batch_size_from_batch_dict(batch_dict):
    bd = batch_dict.get("modality_token_breakdown", None)
    if bd is None:
        sigs = batch_dict.get("modality_signature", None)
        if sigs is not None:
            return len(sigs)
        for v in batch_dict.values():
            try:
                return len(v)
            except Exception:
                continue
        return 0
    if isinstance(bd, (list, np.ndarray)):
        return len(bd)
    if isinstance(bd, dict):
        for v in bd.values():
            if isinstance(v, dict):
                for vv in v.values():
                    try: return len(vv)
                    except Exception: pass
            else:
                try: return len(v)
                except Exception: pass
    return 0

def _vec(bd_list, key, sub=None):
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

def _mean(xs): return float(np.mean(xs)) if xs else 0.0
def _min(xs):  return float(np.min(xs))  if xs else 0.0
def _max(xs):  return float(np.max(xs))  if xs else 0.0

def log_modality_budgets(batch_dict, step):
    """
    Logs mean/min/max for:
      - input_ids_len (pre-truncation), text_tokens_est (pre-truncation)
      - image_kv_tokens_est, video_kv_tokens_est
      - audio_secs
      - placeholders (image/video/audio)
    Plus per-signature (modality_signature) aggregates.
    """
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

    # vectors
    input_ids_len = _vec(bd_list, "input_ids_len")          # pre-truncation
    text_tokens   = _vec(bd_list, "text_tokens_est")        # pre-truncation
    img_kv        = _vec(bd_list, "image_kv_tokens_est")
    vid_kv        = _vec(bd_list, "video_kv_tokens_est")
    aud_secs      = _vec(bd_list, "audio_secs")
    ph_img        = _vec(bd_list, "placeholders", "image_ph")
    ph_vid        = _vec(bd_list, "placeholders", "video_ph")
    ph_aud        = _vec(bd_list, "placeholders", "audio_ph")

    # batch-level summaries (mean/min/max)
    wandb.log({
        "modality_token_budgets/input_ids/mean": _mean(input_ids_len),
        "modality_token_budgets/input_ids/min":  _min(input_ids_len),
        "modality_token_budgets/input_ids/max":  _max(input_ids_len),

        "modality_token_budgets/text/mean": _mean(text_tokens),
        "modality_token_budgets/text/min":  _min(text_tokens),
        "modality_token_budgets/text/max":  _max(text_tokens),

        "modality_token_budgets/img_kv/mean": _mean(img_kv),
        "modality_token_budgets/img_kv/min":  _min(img_kv),
        "modality_token_budgets/img_kv/max":  _max(img_kv),

        "modality_token_budgets/vid_kv/mean": _mean(vid_kv),
        "modality_token_budgets/vid_kv/min":  _min(vid_kv),
        "modality_token_budgets/vid_kv/max":  _max(vid_kv),

        "modality_token_budgets/audio_secs/mean": _mean(aud_secs),
        "modality_token_budgets/audio_secs/min":  _min(aud_secs),
        "modality_token_budgets/audio_secs/max":  _max(aud_secs),

        "modality_token_budgets/placeholders/image/mean": _mean(ph_img),
        "modality_token_budgets/placeholders/image/min":  _min(ph_img),
        "modality_token_budgets/placeholders/image/max":  _max(ph_img),

        "modality_token_budgets/placeholders/video/mean": _mean(ph_vid),
        "modality_token_budgets/placeholders/video/min":  _min(ph_vid),
        "modality_token_budgets/placeholders/video/max":  _max(ph_vid),

        "modality_token_budgets/placeholders/audio/mean": _mean(ph_aud),
        "modality_token_budgets/placeholders/audio/min":  _min(ph_aud),
        "modality_token_budgets/placeholders/audio/max":  _max(ph_aud),
    }, step=step)

    # per-signature aggregates (mean/min/max)
    if sigs is not None and len(sigs) == len(bd_list):
        buckets = defaultdict(lambda: {"L":[], "T":[], "Ikv":[], "Vkv":[], "As":[], "Phi":[], "Phv":[], "Pha":[]})
        for s, d in zip(sigs, bd_list):
            buckets[s]["L"].append(float(d.get("input_ids_len", 0)))
            buckets[s]["T"].append(float(d.get("text_tokens_est", 0)))
            buckets[s]["Ikv"].append(float(d.get("image_kv_tokens_est", 0)))
            buckets[s]["Vkv"].append(float(d.get("video_kv_tokens_est", 0)))
            buckets[s]["As"].append(float(d.get("audio_secs", 0.0)))
            ph = d.get("placeholders", {}) or {}
            buckets[s]["Phi"].append(float(ph.get("image_ph", 0)))
            buckets[s]["Phv"].append(float(ph.get("video_ph", 0)))
            buckets[s]["Pha"].append(float(ph.get("audio_ph", 0)))

        payload = {}
        for s, b in buckets.items():
            tag = _sanitize_tag(s)
            payload.update({
                f"modality_token_budgets_by_sig/{tag}/n": len(b["L"]),

                f"modality_token_budgets_by_sig/{tag}/input_ids/mean": _mean(b["L"]),
                f"modality_token_budgets_by_sig/{tag}/input_ids/min":  _min(b["L"]),
                f"modality_token_budgets_by_sig/{tag}/input_ids/max":  _max(b["L"]),

                f"modality_token_budgets_by_sig/{tag}/text/mean": _mean(b["T"]),
                f"modality_token_budgets_by_sig/{tag}/text/min":  _min(b["T"]),
                f"modality_token_budgets_by_sig/{tag}/text/max":  _max(b["T"]),

                f"modality_token_budgets_by_sig/{tag}/img_kv/mean": _mean(b["Ikv"]),
                f"modality_token_budgets_by_sig/{tag}/img_kv/min":  _min(b["Ikv"]),
                f"modality_token_budgets_by_sig/{tag}/img_kv/max":  _max(b["Ikv"]),

                f"modality_token_budgets_by_sig/{tag}/vid_kv/mean": _mean(b["Vkv"]),
                f"modality_token_budgets_by_sig/{tag}/vid_kv/min":  _min(b["Vkv"]),
                f"modality_token_budgets_by_sig/{tag}/vid_kv/max":  _max(b["Vkv"]),

                f"modality_token_budgets_by_sig/{tag}/audio_secs/mean": _mean(b["As"]),
                f"modality_token_budgets_by_sig/{tag}/audio_secs/min":  _min(b["As"]),
                f"modality_token_budgets_by_sig/{tag}/audio_secs/max":  _max(b["As"]),

                f"modality_token_budgets_by_sig/{tag}/placeholders/image/mean": _mean(b["Phi"]),
                f"modality_token_budgets_by_sig/{tag}/placeholders/image/min":  _min(b["Phi"]),
                f"modality_token_budgets_by_sig/{tag}/placeholders/image/max":  _max(b["Phi"]),

                f"modality_token_budgets_by_sig/{tag}/placeholders/video/mean": _mean(b["Phv"]),
                f"modality_token_budgets_by_sig/{tag}/placeholders/video/min":  _min(b["Phv"]),
                f"modality_token_budgets_by_sig/{tag}/placeholders/video/max":  _max(b["Phv"]),

                f"modality_token_budgets_by_sig/{tag}/placeholders/audio/mean": _mean(b["Pha"]),
                f"modality_token_budgets_by_sig/{tag}/placeholders/audio/min":  _min(b["Pha"]),
                f"modality_token_budgets_by_sig/{tag}/placeholders/audio/max":  _max(b["Pha"]),
            })
        if payload:
            wandb.log(payload, step=step)

# ---- end utils ----
