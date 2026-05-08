"""One-off: patch sglang's gemma3_mm.py load_weights to accept transformers 5.x keys.

transformers 5.x Gemma3 state_dict naming:
  model.language_model.X
  model.vision_tower.X            (no inner vision_model wrap)
  model.multi_modal_projector.X
  lm_head.weight

sglang Gemma3ForConditionalGeneration internal naming:
  language_model.X                (Gemma3ForCausalLM.load_weights handles this)
  vision_tower.vision_model.X     (SiglipVisionModel wraps SiglipVisionTransformer)
  multi_modal_projector.X
  lm_head.weight
"""

import sglang


def main():
    import sglang.srt.models.gemma3_mm as mod
    fp = mod.__file__
    with open(fp) as f:
        s = f.read()

    original = (
        "        for name, loaded_weight in weights:\n"
        '            if "language_model" in name:'
    )
    patched_old = (
        "        for name, loaded_weight in weights:\n"
        "            # Strip transformers 5.x 'model.' prefix on multimodal weight keys\n"
        '            if name.startswith("model.") and not name.startswith("model.language_model"):\n'
        '                name = name[len("model."):]\n'
        '            if "language_model" in name:'
    )
    new_patch = (
        "        for name, loaded_weight in weights:\n"
        "            # transformers 5.x Gemma3 -> sglang key remap:\n"
        "            # model.language_model.X -> language_model.model.X\n"
        "            # model.vision_tower.X   -> vision_tower.vision_model.X\n"
        "            # model.multi_modal_projector.X -> multi_modal_projector.X\n"
        '            if name.startswith("model."):\n'
        '                name = name[len("model."):]\n'
        '                if name.startswith("vision_tower.") and not name.startswith("vision_tower.vision_model."):\n'
        '                    name = "vision_tower.vision_model." + name[len("vision_tower."):]\n'
        '                elif name.startswith("language_model.") and not name.startswith("language_model.model."):\n'
        '                    name = "language_model.model." + name[len("language_model."):]\n'
        '            if "language_model" in name:'
    )

    if new_patch in s:
        print(f"already up to date: {fp}")
        return

    if patched_old in s:
        s2 = s.replace(patched_old, new_patch, 1)
    elif original in s:
        s2 = s.replace(original, new_patch, 1)
    else:
        raise SystemExit(f"could not find target block in {fp}; manual inspection required")

    with open(fp, "w") as f:
        f.write(s2)
    print(f"patched: {fp}")


if __name__ == "__main__":
    main()
