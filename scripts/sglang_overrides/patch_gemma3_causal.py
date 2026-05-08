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

OLD = "        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]"
NEW = """        if self.rope_type in ROPE_INIT_FUNCTIONS:
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


def main():
    for p in CANDIDATES:
        path = Path(p)
        if not path.is_file():
            continue
        s = path.read_text()
        if NEW in s:
            print(f"already patched: {path}")
            return
        if OLD not in s:
            print(f"old line not found in {path}; skipping", file=sys.stderr)
            continue
        path.write_text(s.replace(OLD, NEW, 1))
        print(f"patched: {path}")
        return
    sys.exit("ERROR: no candidate gemma3_causal.py found")


if __name__ == "__main__":
    main()
