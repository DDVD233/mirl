# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Per-modality CLIMB validation metrics.

verl's standard validation reduction (``process_validation_metrics``) groups by
``data_source`` and takes the per-sample mean — perfect for accuracy (exact
label-set match), but it cannot compute class-macro F1, which needs the full
confusion matrix over all samples in a modality. This module adds that
corpus-level reduction:

  * per clinical modality (``data_source == "climb_<modality>"``): exact-match
    accuracy and class-macro F1,
  * overall: the macro-average across modalities (so the heavily over-
    represented chest_xray split doesn't dominate the headline number).

Generated training rows (``data_source == "climb_gen"``) are excluded — they are
not part of the held-out evaluation modalities.
"""

from collections import defaultdict

from verl.utils.climb import class_macro_f1, exact_match, parse_label_set

CLIMB_PREFIX = "climb_"


def compute_climb_modality_metrics(data_sources, preds, gts) -> dict:
    """Compute per-modality accuracy + class-macro F1 and their macro-averages.

    Args:
        data_sources: per-sample data source strings (e.g. ``climb_chest_xray``).
        preds: per-sample extracted model answers (strings).
        gts: per-sample ground-truth answers (strings).

    Returns:
        Flat ``{metric_name: value}`` dict, or ``{}`` if no CLIMB eval rows are
        present. Per-modality numbers land under ``val-aux/``; the macro-average
        headline numbers under ``val-core/``.
    """
    n = min(len(data_sources), len(preds), len(gts))
    by_mod = defaultdict(list)  # modality -> list[(pred_set, gt_set)]
    for i in range(n):
        ds = data_sources[i]
        if not isinstance(ds, str) or not ds.startswith(CLIMB_PREFIX) or ds == "climb_gen":
            continue
        modality = ds[len(CLIMB_PREFIX):]
        pred_set = parse_label_set(preds[i] or "")
        gt_set = parse_label_set(gts[i] or "")
        by_mod[modality].append((pred_set, gt_set))

    if not by_mod:
        return {}

    metrics = {}
    accs, f1s = [], []
    for modality, pairs in sorted(by_mod.items()):
        acc = sum(exact_match(p, g) for p, g in pairs) / len(pairs)
        f1 = class_macro_f1(pairs)
        metrics[f"val-aux/climb_{modality}/accuracy"] = acc
        metrics[f"val-aux/climb_{modality}/f1"] = f1
        metrics[f"val-aux/climb_{modality}/n"] = float(len(pairs))
        accs.append(acc)
        f1s.append(f1)

    metrics["val-core/climb/macro_acc"] = sum(accs) / len(accs)
    metrics["val-core/climb/macro_f1"] = sum(f1s) / len(f1s)
    metrics["val-core/climb/n_modalities"] = float(len(accs))
    return metrics
