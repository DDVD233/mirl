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

from io import BytesIO
from typing import Optional

import torch
from PIL import Image
from qwen_vl_utils import fetch_image, fetch_video


def process_image(image: dict | Image.Image | str) -> Image.Image:
    if isinstance(image, str):
        image = {"type": "image", "image": image, "min_pixels": 65536, "max_pixels": 524288}

    if isinstance(image, Image.Image):
        return image.convert("RGB")

    if "bytes" in image:
        assert "image" not in image, "Cannot have both `bytes` and `image`"
        image["image"] = Image.open(BytesIO(image["bytes"]))

    try:
        return fetch_image(image)
    except Exception as e:
        print(e)
        dummy_image = Image.new("RGB", (224, 224))
        return process_image(dummy_image)


VIDEO_FORMAT_HELP = """Currently, we only support the video formats introduced in qwen2-vl.
Refer to https://github.com/QwenLM/Qwen2.5-VL?tab=readme-ov-file#using---transformers-to-chat.

eg.
{
    "type": "video",
    "video": [
        "file:///path/to/frame1.jpg",
        "file:///path/to/frame2.jpg"
    ]
}

{
    "type": "video",
    "video": "file:///path/to/video.mp4"
}
# Defaults to fps=2, min_frames=4, max_frames=768

{
    "type": "video",
    "video": "file:///path/to/video.mp4",
    "fps": 2,
    "min_frames": 1,
    "max_frames": 32
}
"""

from typing import Optional, Dict
import os, time, traceback
import torch

def process_video(
    video: Dict,
    nframes: Optional[int] = None,
    fps: Optional[float] = None,
    fps_min_frames: Optional[int] = None,
    fps_max_frames: Optional[int] = None,
    *,
    debug: bool = False,           # <-- turn on diagnostics
    name_hint: Optional[str] = None
) -> torch.Tensor:
    """Converts a video dict into a [n_frames, 3, H, W] uint8 tensor.

    Set debug=True for per-call diagnostics to help track OOM spikes.
    """
    start_t = time.perf_counter()

    # Normalize string input → dict
    if isinstance(video, str):
        # Your current defaults (tiny visual budget)
        # video = {"type": "video", "video": video,
        #          "min_pixels": 32768, "max_pixels": 32768, "nframes": 2}

    # Moderate budget
        # video = {"type": "video", "video": video,
        #         "min_pixels": 49152, "max_pixels": 262144, "nframes": 4}
        video= {
            "type": "video", "video": video,
            "min_pixels": 147456, "max_pixels": 147456, "nframes": 4
        }        
        
    # # Most expensive budget
    #     video = {"type": "video", "video": video, "min_pixels": 65536, "max_pixels": 524288,
    #              "nframes": 4}

    if not isinstance(video, dict) or "video" not in video:
        raise NotImplementedError("Video format must be dict with key 'video'.")

    # Shallow copy; we may add keys
    video = dict(video)

    # Compose sampling rules
    assert nframes is None or fps is None, "Can't use both `nframes` and `fps`."
    contains_sampling_rules = ("nframes" in video) or ("fps" in video)
    if not contains_sampling_rules:
        if nframes is not None:
            video["nframes"] = nframes
        elif fps is not None:
            video["fps"] = fps
            if fps_min_frames is not None:
                video["min_frames"] = fps_min_frames
            if fps_max_frames is not None:
                video["max_frames"] = fps_max_frames

    # --- DIAGNOSTICS (before decode/resize) ---
    if debug:
        vpath = video.get("video")
        min_px = video.get("min_pixels", None)
        max_px = video.get("max_pixels", None)
        nf_req = video.get("nframes", None)
        fps_req = video.get("fps", None)
        size_on_disk = None
        try:
            if isinstance(vpath, str) and os.path.exists(vpath):
                size_on_disk = os.path.getsize(vpath)
        except Exception:
            pass
        print(f"[process_video][pre] name={name_hint or ''} path={vpath} exists={os.path.exists(vpath) if isinstance(vpath,str) else 'N/A'} "
              f"size={size_on_disk}B min_px={min_px} max_px={max_px} "
              f"nframes_req={nf_req} fps_req={fps_req}")

    # Decode + resize according to your fetcher
    try:
        frames = fetch_video(video)  # expected [T, 3, H, W], dtype=uint8
    except Exception as e:
        if debug:
            print(f"[process_video][error] {e}\n{traceback.format_exc()}")
        # Return a small dummy to keep pipeline alive
        dummy = torch.zeros((1, 3, 224, 224), dtype=torch.uint8)
        return dummy

    # --- DIAGNOSTICS (after decode/resize) ---
    if debug:
        try:
            T, C, H, W = frames.shape
        except Exception:
            T, C, H, W = (None, None, None, None)

        # Estimate visual tokens via 28x28 rule
        def tok_per_frame(h, w):
            if not (isinstance(h, int) and isinstance(w, int)):
                return None
            # tokens ≈ ceil(H/28)*ceil(W/28)
            import math
            return math.ceil(h / 28) * math.ceil(w / 28)

        vtok_pf = tok_per_frame(H, W)
        vtok_total = (vtok_pf * T) if (vtok_pf is not None and isinstance(T, int)) else None

        # Optional: quick CUDA mem snapshot (safe even on CPU)
        if torch.cuda.is_available():
            cur = round(torch.cuda.memory_allocated() / 1e9, 2)
            peak = round(torch.cuda.max_memory_allocated() / 1e9, 2)
            mem_str = f"cuda_cur={cur}GB cuda_peak={peak}GB"
        else:
            mem_str = "cuda=N/A"

        dt = (time.perf_counter() - start_t) * 1000
        print(f"[process_video][post] name={name_hint or ''} shape={frames.shape} "
              f"HxW={H}x{W} T={T} tok/frame≈{vtok_pf} tok_total≈{vtok_total} "
              f"elapsed={dt:.1f}ms {mem_str}")

        # Warn if frame count or token budget larger than expected
        if T is not None and (("nframes" in video and T != video["nframes"]) or (T > 8)):
            print(f"[process_video][warn] unexpected T={T} (requested {video.get('nframes')}).")
        if vtok_total is not None and vtok_total > 4000:
            print(f"[process_video][warn] large visual token count: ~{vtok_total}. Consider lowering pixels/frames.")

    return frames


# def process_video(
#     video: dict,
#     nframes: Optional[int] = None,
#     fps: Optional[float] = None,
#     fps_min_frames: Optional[int] = None,
#     fps_max_frames: Optional[int] = None,
# ) -> torch.Tensor:
#     """Converts a video dict into a [n_frames, 3, H, W] tensor

#     Add video sample FPS in a future MR
#     """
#     if isinstance(video, str):
#         # This is the original form
#         # video = {"type": "video", "video": video, "min_pixels": 65536, "max_pixels": 524288,
#         #          "nframes": 4}
#         video = {"type": "video", "video": video, "min_pixels": 32768, "max_pixels": 32768,
#             "nframes": 2}

#     if not isinstance(video, dict) or "video" not in video:
#         raise NotImplementedError(VIDEO_FORMAT_HELP)
#     assert nframes is None or fps is None, "Can't use both `nframes` or `fps`"

#     # Shallow copy... since we might want to add some keys
#     video = dict(video)

#     contains_sampling_rules = "nframes" in video or "fps" in video
#     if not contains_sampling_rules:
#         if nframes is not None:
#             video["nframes"] = nframes
#         elif fps is not None:
#             video["fps"] = fps
#             if fps_min_frames is not None:
#                 video["min_frames"] = fps_min_frames
#             if fps_max_frames is not None:
#                 video["max_frames"] = fps_max_frames
#     try:
#         return fetch_video(video)
#     except Exception as e:
#         print(e)
#         dummy_video = torch.zeros((1, 3, 224, 224), dtype=torch.uint8)
#         return dummy_video


def process_multi_modal_inputs_for_minicpmo(input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs):
    # Adjust image bounds based on left padding and cumulative sequence lengths
    # This is necessary for MiniCPM-o's vision-language alignment
    left_padding_length = torch.argmax(attention_mask, dim=1)
    image_bounds = []
    for i in range(len(multi_modal_inputs["image_bound"])):
        image_bound = (
            multi_modal_inputs["image_bound"][i].to(left_padding_length.device) - left_padding_length[i] + cu_seqlens[i]
        )
        image_bounds.append(image_bound)

    # Flatten pixel values list for MiniCPM-o processing
    pixel_values = []
    for i in range(len(multi_modal_inputs["pixel_values"])):
        pixel_values.extend([p for p in multi_modal_inputs["pixel_values"][i]])

    multi_modal_inputs["pixel_values"] = [pixel_values]
    multi_modal_inputs["image_bound"] = [torch.vstack(image_bounds)]
    multi_modal_inputs["tgt_sizes"] = [torch.vstack(multi_modal_inputs["tgt_sizes"])]
    multi_modal_inputs["input_ids"] = input_ids
    multi_modal_inputs["attention_mask"] = attention_mask
    multi_modal_inputs["position_ids"] = position_ids
    return {"data": multi_modal_inputs}
