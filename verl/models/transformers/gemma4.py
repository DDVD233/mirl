# Gemma4 mixed multimodal/text-only training fix for FSDP/DeepSpeed.
#
# Problem: with a sharded vision tower, the vision forward (and its collectives)
# run only when a micro-batch has an image. verl distributes samples by sequence
# length, not modality, so in a mixed batch some ranks run the vision tower and
# others do not -> mismatched NCCL collectives -> backward deadlock (the
# "1 rank N collectives ahead in ReduceScatter" hang).
#
# Fix (adapted from ms-swift PR modelscope/ms-swift#9180 "support gemma4 mixed
# data"): all_reduce a flag so every rank agrees whether ANY rank has an image,
# and on text-only ranks run the vision tower on a tiny dummy image with a
# zero-scaled contribution -> the vision-tower forward/backward (and collectives)
# run uniformly across ranks, with no effect on the output.
import torch
import torch.distributed as dist
from types import MethodType

from PIL import Image

_DUMMY_PIXELS = {}


def _get_dummy_pixels(model_name, device, dtype):
    if model_name not in _DUMMY_PIXELS:
        from transformers import AutoProcessor

        proc = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        img = Image.new("RGB", (32, 32), (0, 0, 0))
        out = proc.image_processor(images=[img], return_tensors="pt")
        _DUMMY_PIXELS[model_name] = (out["pixel_values"], out.get("image_position_ids"))
    pv, pos = _DUMMY_PIXELS[model_name]
    pv = pv.to(device=device, dtype=dtype)
    pos = pos.to(device) if pos is not None else None
    return pv, pos


def apply_gemma4_mixed_data_patch(model):
    """Patch Gemma4Model.forward so the vision tower runs uniformly across ranks."""
    inner = getattr(model, "model", model)  # Gemma4ForConditionalGeneration.model -> Gemma4Model
    if getattr(inner, "_mixed_data_patched", False):
        return False
    if not (hasattr(inner, "get_image_features") and hasattr(inner, "vision_tower") and inner.vision_tower is not None):
        return False
    model_name = getattr(getattr(model, "config", None), "_name_or_path", None) or getattr(
        getattr(inner, "config", None), "_name_or_path", None
    )
    if not model_name:
        return False
    orig_forward = inner.forward

    def _dummy_image_embeds(self, inputs_embeds):
        pv, pos = _get_dummy_pixels(model_name, inputs_embeds.device, self.vision_tower.dtype)
        feats = self.get_image_features(pv, pos, return_dict=True).pooler_output
        return inputs_embeds + feats.mean() * 0.0

    def forward(self, input_ids=None, pixel_values=None, pixel_values_videos=None,
                inputs_embeds=None, per_layer_inputs=None, **kwargs):
        has_local = (pixel_values is not None) or (pixel_values_videos is not None)
        any_image = has_local
        if dist.is_initialized():
            ref = input_ids if input_ids is not None else inputs_embeds
            flag = torch.tensor([1 if has_local else 0], device=ref.device, dtype=torch.long)
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            any_image = flag.item() > 0
        if any_image and (not has_local) and input_ids is not None and inputs_embeds is None:
            ie = self.get_input_embeddings()(input_ids)
            if per_layer_inputs is None and self.config.get_text_config().hidden_size_per_layer_input:
                per_layer_inputs = self.language_model.get_per_layer_inputs(input_ids, ie)
            ie = _dummy_image_embeds(self, ie)
            return orig_forward(input_ids=None, pixel_values=None, pixel_values_videos=None,
                                inputs_embeds=ie, per_layer_inputs=per_layer_inputs, **kwargs)
        return orig_forward(input_ids=input_ids, pixel_values=pixel_values,
                            pixel_values_videos=pixel_values_videos, inputs_embeds=inputs_embeds,
                            per_layer_inputs=per_layer_inputs, **kwargs)

    inner.forward = MethodType(forward, inner)
    inner._mixed_data_patched = True
    return True
