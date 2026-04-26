# helper_senti_extra.py
# Adds macro_f1 / micro_f1 per sentiment collapse level (2/3/5/7).
# Intentionally self-contained: duplicates the small pure helpers from
# helper_senti.py so this file has no imports from that module.
from typing import Dict, List, Tuple


def _safe_div(num: float, den: float) -> float:
    return num / den if den > 0 else 0.0


def _order_sentiment_labels(meta: Dict) -> Tuple[List[str], Dict[str, int], Dict[int, int]]:
    canonical = [
        "highly negative", "negative", "weakly negative", "neutral",
        "weakly positive", "positive", "highly positive",
    ]
    gc = meta["global_classes"]["sentiment_intensity"]
    label_to_global = {entry["label"]: entry["index"] for entry in gc}
    missing = [lab for lab in canonical if lab not in label_to_global]
    if missing:
        raise ValueError(f"Sentiment labels missing in meta: {missing}")
    label2pos = {lab: i for i, lab in enumerate(canonical)}
    global2pos = {label_to_global[lab]: label2pos[lab] for lab in canonical}
    return canonical, label2pos, global2pos


def _sentiment_sets(meta: Dict) -> Dict[str, set]:
    _, _, g2p = _order_sentiment_labels(meta)
    gc = meta["global_classes"]["sentiment_intensity"]
    label2global = {e["label"]: e["index"] for e in gc}
    neg_labels = {"highly negative", "negative", "weakly negative"}
    pos_labels = {"weakly positive", "positive", "highly positive"}
    NEG = {g2p[label2global[l]] for l in neg_labels}
    POS = {g2p[label2global[l]] for l in pos_labels}
    NEU = {g2p[label2global["neutral"]]}
    return {"NEG": NEG, "POS": POS, "NEU": NEU}


def _translate_global_to_pos(seq: List[int], meta: Dict) -> List[int]:
    _, _, g2p = _order_sentiment_labels(meta)
    return [g2p[v] if v in g2p else -1 for v in seq]


def _collapse_positions(seq_pos: List[int], mode: int, meta: Dict) -> List[int]:
    sets = _sentiment_sets(meta)
    NEG, POS, NEU = sets["NEG"], sets["POS"], sets["NEU"]
    out = []
    for v in seq_pos:
        if mode == 7:
            out.append(v)
        elif mode == 5:
            if v == min(NEG):
                out.append(0)
            elif v in (NEG - {min(NEG)}):
                out.append(1)
            elif v in NEU:
                out.append(2)
            elif v in (POS - {max(POS)}):
                out.append(3)
            elif v == max(POS):
                out.append(4)
            else:
                out.append(-1)
        elif mode == 3:
            if v in NEG:
                out.append(0)
            elif v in NEU:
                out.append(1)
            elif v in POS:
                out.append(2)
            else:
                out.append(-1)
        elif mode == 2:
            if v in POS:
                out.append(1)
            elif v in NEG:
                out.append(0)
            else:
                out.append(-1)
        else:
            raise ValueError(f"Unsupported sentiment collapse mode: {mode}")
    return out


def _full_collapse_metrics(
    y_pred: List[int],
    y_true: List[int],
    compute_set_metrics_fn,
) -> Dict[str, float]:
    """Extract weighted, macro, and micro F1 plus accuracy from compute_set_metrics_fn."""
    ds = compute_set_metrics_fn(y_pred, y_true)["dataset_metrics"]
    return {
        "weighted_f1": ds.get("weighted_f1", 0.0),
        "macro_f1":    ds.get("macro_f1", 0.0),
        "micro_f1":    ds.get("micro_f1", 0.0),
        "accuracy":    ds.get("micro_accuracy", 0.0),
    }


def compute_sentiment_extra_metrics(
    y_pred_global: List[int],
    y_true_global: List[int],
    meta: Dict,
    compute_set_metrics_fn,
) -> Dict[str, float]:
    """
    Compute the *additional* per-collapse-level macro and micro F1 metrics that
    compute_sentiment_collapsed_metrics (helper_senti.py) does not emit.

    Returns keys: F1macro7, F1micro7, F1macro5, F1micro5,
                  F1macro3, F1micro3, F1macro2, F1micro2
    """
    senti_globals = {e["index"] for e in meta["global_classes"]["sentiment_intensity"]}
    idxs = [i for i, t in enumerate(y_true_global) if t in senti_globals]
    if not idxs:
        return {}

    y_true_sent = [y_true_global[i] for i in idxs]
    y_pred_sent = [y_pred_global[i] for i in idxs]

    y_true_pos = _translate_global_to_pos(y_true_sent, meta)
    y_pred_pos = _translate_global_to_pos(y_pred_sent, meta)

    metrics: Dict[str, float] = {}

    for mode in (7, 5, 3):
        y_t = _collapse_positions(y_true_pos, mode, meta)
        y_p = _collapse_positions(y_pred_pos, mode, meta)
        m = _full_collapse_metrics(y_p, y_t, compute_set_metrics_fn)
        metrics[f"F1macro{mode}"] = m["macro_f1"]
        metrics[f"F1micro{mode}"] = m["micro_f1"]

    # mode 2: exclude neutral GT (mirrors helper_senti.py's binary collapse)
    sets = _sentiment_sets(meta)
    NEU = sets["NEU"]
    y_t7 = _collapse_positions(y_true_pos, 7, meta)
    keep = [i for i, t in enumerate(y_t7) if t not in NEU]
    if keep:
        y_t2_src = [y_true_pos[i] for i in keep]
        y_p2_src = [y_pred_pos[i] for i in keep]
        y_t2 = _collapse_positions(y_t2_src, 2, meta)
        y_p2 = _collapse_positions(y_p2_src, 2, meta)
        m2 = _full_collapse_metrics(y_p2, y_t2, compute_set_metrics_fn)
        metrics["F1macro2"] = m2["macro_f1"]
        metrics["F1micro2"] = m2["micro_f1"]
    else:
        metrics["F1macro2"] = 0.0
        metrics["F1micro2"] = 0.0

    return metrics
