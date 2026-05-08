"""Patch sglang's gemma3_causal.py for transformers 5.x compatibility.

In transformers 5.x, ROPE_INIT_FUNCTIONS no longer has a 'default' key
(it was removed in favor of inlining the default RoPE computation).
sglang 0.5.9 still does ``ROPE_INIT_FUNCTIONS[self.rope_type]`` with
``self.rope_type = 'default'`` for Gemma3, which crashes with KeyError.

This script rewrites the Gemma3RotaryEmbedding init to handle 'default'
manually using the same formula transformers 4.x used.

Run inside the apptainer container (which gives a writable-tmpfs overlay):
    python3 scripts/sglang_overrides/patch_gemma3_causal.py
"""

import sys
from pathlib import Path

CANDIDATES = [
    "/sgl-workspace/sglang/python/sglang/srt/models/gemma3_causal.py",
    "/usr/local/lib/python3.12/dist-packages/sglang/srt/models/gemma3_causal.py",
]

# Patch 1: ROPE_INIT_FUNCTIONS no longer has 'default' in transformers 5.x.
ROPE_OLD = "        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]"
ROPE_NEW = """        if self.rope_type in ROPE_INIT_FUNCTIONS:
            self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        else:
            # transformers 5.x removed the 'default' entry; reproduce the
            # original compute_default_rope_parameters here.
            def _default_rope_init_fn(config, device=None, seq_len=None, **kwargs):
                import torch
                base = getattr(config, "rope_theta", 10000.0)
                partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
                head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
                dim = int(head_dim * partial_rotary_factor)
                inv_freq = 1.0 / (
                    base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device).float() / dim)
                )
                return inv_freq, 1.0
            self.rope_init_fn = _default_rope_init_fn"""

# Patch 2: rope_local_base_freq was renamed to rope_parameters in transformers
# 5.x (a dict keyed by 'sliding_attention'/'full_attention'). Fall back to the
# global rope_theta if neither is present. Two variants exist in the file:
#   - module-level ``        config.rope_theta = config.rope_local_base_freq``
#   - inside ``if self.is_sliding:`` ``            self.rope_theta = config.rope_local_base_freq``
LOCAL_OLD_VARIANTS = [
    (
        "        config.rope_theta = config.rope_local_base_freq",
        """        if hasattr(config, "rope_local_base_freq"):
            config.rope_theta = config.rope_local_base_freq
        else:
            _rp = getattr(config, "rope_parameters", None) or {}
            _local = _rp.get("sliding_attention") if isinstance(_rp, dict) else None
            if isinstance(_local, dict) and "rope_theta" in _local:
                config.rope_theta = _local["rope_theta"]
            elif _local is not None and hasattr(_local, "rope_theta"):
                config.rope_theta = _local.rope_theta""",
    ),
    (
        "            self.rope_theta = config.rope_local_base_freq",
        """            if hasattr(config, "rope_local_base_freq"):
                self.rope_theta = config.rope_local_base_freq
            else:
                _rp = getattr(config, "rope_parameters", None) or {}
                _local = _rp.get("sliding_attention") if isinstance(_rp, dict) else None
                if isinstance(_local, dict) and "rope_theta" in _local:
                    self.rope_theta = _local["rope_theta"]
                elif _local is not None and hasattr(_local, "rope_theta"):
                    self.rope_theta = _local.rope_theta
                else:
                    self.rope_theta = getattr(config, "rope_theta", 10000.0)""",
    ),
]


def apply_patch(s, old, new, name):
    if new in s:
        print(f"  already applied: {name}")
        return s, False
    if old not in s:
        print(f"  WARN: {name} marker not found", file=sys.stderr)
        return s, False
    print(f"  applied: {name}")
    return s.replace(old, new), True


def main():
    for p in CANDIDATES:
        path = Path(p)
        if not path.is_file():
            continue
        print(f"patching {path}")
        s = path.read_text()
        s, c1 = apply_patch(s, ROPE_OLD, ROPE_NEW, "rope_init_fn fallback")
        any_changed = c1
        for old, new in LOCAL_OLD_VARIANTS:
            s, changed = apply_patch(s, old, new, f"rope_local_base_freq @ {old.lstrip()[:30]}...")
            any_changed = any_changed or changed
        if any_changed:
            path.write_text(s)
        return
    sys.exit("ERROR: no candidate gemma3_causal.py found")


if __name__ == "__main__":
    main()
