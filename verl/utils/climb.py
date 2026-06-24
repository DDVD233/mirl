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

"""Shared helpers for the multimodal CLIMB self-evolving pipeline.

This module is the single importable home for the two things that several
otherwise-unrelated components must agree on:

1. **Media resolution.** CLIMB image/video files live only on the local node
   (``mib``) under ``/scratch/high_modality`` and are served over HTTP by
   ``scripts/self_evolving/climb_file_server.py`` (Bearer-auth). The trainer
   runs on remote B200 nodes that cannot see those files on disk, so dataset
   rows carry an opaque reference ``"climb://<relpath>"`` (or the dict form
   ``{"climb_path": "<relpath>"}``) instead of a path. ``resolve_climb_media``
   downloads each referenced file once into a local content cache and rewrites
   the reference to an absolute local path, after which verl's normal
   multimodal path (``RLHFDataset._build_messages`` →
   ``vision_utils.process_image`` / ``process_video``) handles it unchanged.

2. **Label parsing + classification metrics.** The CLIMB reward scorer
   (``verl/utils/reward_score/climb.py``) and the validation metric reducer
   (``verl/trainer/ppo/climb_metrics.py``) must parse model answers and ground
   truths into label *sets* identically, and compute per-modality class-macro
   F1 the same way. Those helpers live here so both sides import one source of
   truth.

The auth token is **always** read from the environment (default env var
``CLIMB_FILE_TOKEN``); it is never hardcoded or persisted in any file.
"""

import logging
import os
import re
import tempfile
from typing import Optional

logger = logging.getLogger(__name__)

CLIMB_SCHEME = "climb://"
CLIMB_PATH_KEY = "climb_path"
DEFAULT_TOKEN_ENV = "CLIMB_FILE_TOKEN"


# ======================================================================
# Reference helpers
# ======================================================================
def _ref_relpath(ref) -> Optional[str]:
    """Return the CLIMB relpath for a media reference, or None if `ref` is not
    a CLIMB reference (a plain path / PIL image / already-resolved dict)."""
    if isinstance(ref, str):
        if ref.startswith(CLIMB_SCHEME):
            return ref[len(CLIMB_SCHEME):]
        return None
    if isinstance(ref, dict):
        rel = ref.get(CLIMB_PATH_KEY)
        if isinstance(rel, str) and rel:
            return rel
    return None


def climb_modality_of(rel: str) -> str:
    """Clinical modality = the top subfolder of a CLIMB relative path.

    e.g. ``chest_xray/chexpert_full/.../x.jpg`` → ``chest_xray``. The leading
    ``high_modality/`` prefix that Milvus stores in ``image_path`` is stripped
    first so both forms resolve to the same modality.
    """
    rel = (rel or "").lstrip("/")
    if rel.startswith("high_modality/"):
        rel = rel[len("high_modality/"):]
    parts = rel.split("/")
    return parts[0] if parts and parts[0] else "unknown"


def strip_high_modality(rel: str) -> str:
    """Normalize a Milvus ``image_path`` (``high_modality/<m>/..``) or a bare
    relpath to the path the file server expects (rooted at high_modality)."""
    rel = (rel or "").lstrip("/")
    if rel.startswith("high_modality/"):
        rel = rel[len("high_modality/"):]
    return rel


# ======================================================================
# Runtime config + media resolution (trainer side)
# ======================================================================
class ClimbMediaConfig:
    """Resolved runtime config for fetching CLIMB media. Built from the verl
    ``data.climb`` sub-config; ``None`` (via :meth:`from_data_config`) when the
    file server is not configured, so resolution is a no-op for non-CLIMB runs.
    """

    def __init__(self, file_base: str, token: str, cache_dir: str,
                 max_pixels: Optional[int] = None, timeout: float = 60.0,
                 video_frames: int = 8, force_multimodal: bool = False,
                 video_max_pixels: Optional[int] = 200704):
        self.file_base = file_base.rstrip("/")
        self.token = token
        self.cache_dir = os.path.expanduser(cache_dir)
        self.max_pixels = max_pixels
        self.timeout = timeout
        self.video_frames = int(video_frames)
        # Per-frame pixel cap when a video is flattened to image frames. The
        # dataset converts every <video> to N=video_frames <image> tokens; a
        # small cap keeps a whole clip at ~<2k vision tokens (e.g. 200704 px =
        # 448x448 -> 256 tokens/frame, x6 frames = 1536). See _build_messages.
        self.video_max_pixels = int(video_max_pixels) if video_max_pixels else None
        # When True, the dataset attaches a dummy image to text-only rows so
        # every sample exercises the vision tower — see dummy_image().
        self.force_multimodal = bool(force_multimodal)

    @classmethod
    def from_data_config(cls, config) -> Optional["ClimbMediaConfig"]:
        """Build from a verl data DictConfig. Returns None when
        ``data.climb.file_base`` is unset (the common, non-CLIMB case)."""
        if config is None:
            return None
        try:
            climb = config.get("climb", None)
        except Exception:
            climb = None
        if not climb:
            return None
        file_base = climb.get("file_base", "") if hasattr(climb, "get") else getattr(climb, "file_base", "")
        if not file_base:
            return None
        token_env = (climb.get("file_token_env", DEFAULT_TOKEN_ENV)
                     if hasattr(climb, "get") else DEFAULT_TOKEN_ENV)
        token = os.environ.get(token_env, "")
        if not token:
            logger.warning(
                "CLIMB file server configured (file_base=%s) but env %s is empty — "
                "media fetches will be unauthenticated and likely 401.",
                file_base, token_env,
            )
        cache_dir = (climb.get("cache_dir", "~/.cache/verl/climb_media")
                     if hasattr(climb, "get") else "~/.cache/verl/climb_media")
        max_pixels = climb.get("max_pixels", None) if hasattr(climb, "get") else None
        timeout = float(climb.get("timeout", 60.0)) if hasattr(climb, "get") else 60.0
        video_frames = int(climb.get("video_frames", 8)) if hasattr(climb, "get") else 8
        force_multimodal = bool(climb.get("force_multimodal", False)) if hasattr(climb, "get") else False
        video_max_pixels = (climb.get("video_max_pixels", 200704) if hasattr(climb, "get") else 200704)
        video_max_pixels = int(video_max_pixels) if video_max_pixels else None
        return cls(file_base, token, cache_dir, max_pixels, timeout, video_frames,
                   force_multimodal, video_max_pixels)


def _download_to_cache(rel: str, cfg: ClimbMediaConfig, want_max_pixels: bool) -> Optional[str]:
    """Fetch one CLIMB file into the local content cache (download-once) and
    return its absolute path, or None on failure. Atomic via temp+rename so
    concurrent dataloader workers never observe a partial file."""
    import requests

    rel = strip_high_modality(rel)
    cache_path = os.path.join(cfg.cache_dir, rel)
    if os.path.isfile(cache_path) and os.path.getsize(cache_path) > 0:
        return cache_path

    url = f"{cfg.file_base}/file/{rel}"
    params = {}
    if want_max_pixels and cfg.max_pixels:
        params["max_pixels"] = int(cfg.max_pixels)
    headers = {"Authorization": f"Bearer {cfg.token}"} if cfg.token else {}
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with requests.get(url, params=params, headers=headers, timeout=cfg.timeout, stream=True) as r:
            r.raise_for_status()
            fd, tmp = tempfile.mkstemp(dir=os.path.dirname(cache_path), suffix=".part")
            try:
                with os.fdopen(fd, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        if chunk:
                            f.write(chunk)
                os.replace(tmp, cache_path)
            finally:
                if os.path.exists(tmp):
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass
        return cache_path
    except Exception as e:
        logger.warning("CLIMB media fetch failed for %s: %s: %s", rel, type(e).__name__, e)
        return None


def _extract_video_frames(mp4_path: str, n_frames: int) -> list:
    """Extract up to `n_frames` evenly spaced frames from a video to JPGs next
    to it, returning their absolute paths. Uses OpenCV (decord/torchcodec are
    often unavailable and torchvision.io.read_video is gone in recent builds),
    so qwen_vl_utils receives a frame LIST and never calls a video backend.
    Returns [] on failure."""
    try:
        import cv2  # type: ignore

        base = os.path.splitext(mp4_path)[0]
        cap = cv2.VideoCapture(mp4_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        if total <= 0:
            cap.release()
            return []
        n = max(1, n_frames)
        idxs = [min(int(total * (k + 0.5) / n), total - 1) for k in range(n)]
        paths = []
        for j, fi in enumerate(idxs):
            fp = f"{base}.frame{j:03d}.jpg"
            if not (os.path.isfile(fp) and os.path.getsize(fp) > 0):
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                ok, frame = cap.read()
                if not ok:
                    continue
                cv2.imwrite(fp, frame)
            if os.path.isfile(fp):
                paths.append(fp)
        cap.release()
        return paths
    except Exception as e:
        logger.warning("CLIMB video frame extraction failed for %s: %s: %s",
                       mp4_path, type(e).__name__, e)
        return []


def _resolve_one(ref, cfg: ClimbMediaConfig, is_video: bool):
    """Resolve a single image/video reference. CLIMB image refs become an
    absolute local path; CLIMB video refs become a LIST of extracted frame
    paths (so qwen_vl_utils takes its frame-list path and skips video decoding).
    Non-CLIMB refs are returned unchanged."""
    rel = _ref_relpath(ref)
    if rel is None:
        return ref  # not a CLIMB ref — leave the existing path/PIL/dict alone
    local = _download_to_cache(rel, cfg, want_max_pixels=not is_video)
    if local is None:
        # Keep a deterministic missing-file marker; the dataset's image
        # sanitizer / fetch will substitute a placeholder rather than crash.
        local = os.path.join(cfg.cache_dir, strip_high_modality(rel))

    if is_video:
        frames = _extract_video_frames(local, cfg.video_frames)
        # A frame list is consumed by RLHFDataset's video list-branch and by
        # qwen_vl_utils as pre-sampled frames (no read_video). Fall back to the
        # raw path only if extraction produced nothing.
        return frames if frames else local

    extra = {k: v for k, v in (ref.items() if isinstance(ref, dict) else {}).items()
             if k != CLIMB_PATH_KEY}
    if extra:
        return {"image": local, **extra}
    return local


def resolve_climb_media(images, videos, cfg: ClimbMediaConfig):
    """Rewrite any CLIMB references in ``images``/``videos`` to local cached
    paths. Returns new lists; non-CLIMB entries pass through untouched. Safe to
    call on lists with no CLIMB refs (returns them effectively unchanged).
    """
    if cfg is None:
        return images, videos
    new_images = [_resolve_one(im, cfg, is_video=False) for im in (images or [])]
    new_videos = [_resolve_one(vid, cfg, is_video=True) for vid in (videos or [])]
    return new_images, new_videos


def has_climb_refs(images, videos) -> bool:
    """True if any element of either list is a CLIMB reference."""
    for ref in list(images or []) + list(videos or []):
        if _ref_relpath(ref) is not None:
            return True
    return False


def resolve_and_flatten_media(messages, images, videos, cfg):
    """Resolve any ``climb://`` refs to local files and flatten every ``<video>``
    into ``video_frames`` ``<image>`` frames, rewriting message content in place.

    Returns ``(new_images, new_videos)`` where ``new_videos`` is ``[]`` once any
    videos have been folded into the image stream. Mirrors the inline logic in
    ``RLHFDataset._build_messages`` so the SFT dataset (which builds its own
    chat messages with ``<image>``/``<video>`` placeholders) gets the same
    image-only treatment. ``messages`` is a list of ``{"role","content"}`` dicts
    with string content. No-op (returns inputs) when ``cfg`` is None.
    """
    import re as _re

    images = list(images or [])
    videos = list(videos or [])
    if cfg is None:
        return images, videos
    if has_climb_refs(images, videos):
        images, videos = resolve_climb_media(images, videos, cfg)
    if not videos:
        return images, videos

    vmp = getattr(cfg, "video_max_pixels", None)

    def _frames_of(v):
        if isinstance(v, dict):
            v = v.get("video", v)
        frames = v if isinstance(v, list) else [v]
        out = []
        for fr in frames:
            fr = os.fspath(fr) if isinstance(fr, os.PathLike) else fr
            out.append({"image": fr, "max_pixels": vmp} if vmp else {"image": fr})
        return out

    per_video_frames = [_frames_of(v) for v in videos]
    new_images: list = []
    ii = vi = 0
    for message in messages:
        content = message.get("content")
        if not isinstance(content, str):
            continue
        rebuilt = []
        for seg in _re.split("(<image>|<video>|<audio>)", content):
            if seg == "<image>":
                if ii < len(images):
                    new_images.append(images[ii])
                    ii += 1
                rebuilt.append("<image>")
            elif seg == "<video>":
                frames = per_video_frames[vi] if vi < len(per_video_frames) else []
                vi += 1
                new_images.extend(frames)
                rebuilt.append("<image>" * len(frames))
            else:
                rebuilt.append(seg)
        message["content"] = "".join(rebuilt)
    if ii < len(images):
        new_images.extend(images[ii:])
    return new_images, []


def dummy_image(size: int = 224):
    """A blank RGB image attached to text-only rows under force_multimodal.

    Mixing text-only and multimodal samples in one FSDP batch makes the vision
    tower run on some ranks/micro-batches but not others; the vision-tower
    all-gather then desyncs and NCCL times out (the classic Gemma text/image
    mixing hang). Giving every text-only row a blank image makes the vision
    path run uniformly on every rank. 224x224 stays above Qwen's min_pixels.
    """
    from PIL import Image

    return Image.new("RGB", (size, size), color=(0, 0, 0))


# ======================================================================
# Label parsing + classification metrics (reward + eval side)
# ======================================================================
def normalize_label(s: str) -> str:
    """Canonicalize one label phrase for set comparison: lowercase, collapse
    whitespace, drop surrounding quotes/brackets and trailing punctuation."""
    s = (s or "").strip().strip("\"'`[](){}").strip()
    s = re.sub(r"\s+", " ", s)
    s = s.rstrip(".;:").strip()
    return s.lower()


# Split a CLIMB answer string into its constituent labels. CLIMB answers are
# either a single phrase ("No PE") or several comma/semicolon/newline-separated
# phrases ("Pleural Effusion, Support Devices").
_LABEL_SPLIT_RE = re.compile(r"[,;\n]| and (?=[A-Z])")


def parse_label_set(answer: str, options: Optional[list] = None) -> set:
    """Parse a CLIMB answer (gt or model output) into a normalized label set.

    If ``options`` (the modality's allowed phrases) is provided, the answer is
    matched against them so that a model that paraphrases or lists labels in a
    different order still maps onto the canonical option set. Otherwise we fall
    back to splitting on separators.
    """
    if answer is None:
        return set()
    raw = str(answer)
    if options:
        norm_opts = {normalize_label(o): normalize_label(o) for o in options}
        hay = normalize_label(raw)
        found = {canon for nopt, canon in norm_opts.items() if nopt and nopt in hay}
        if found:
            return found
    parts = [normalize_label(p) for p in _LABEL_SPLIT_RE.split(raw)]
    return {p for p in parts if p}


def f1_set(pred: set, gt: set) -> float:
    """Sample-level F1 between a predicted and a ground-truth label set.
    Both empty → 1.0 (vacuously correct); one empty → 0.0."""
    if not gt and not pred:
        return 1.0
    if not gt or not pred:
        return 0.0
    tp = len(pred & gt)
    if tp == 0:
        return 0.0
    precision = tp / len(pred)
    recall = tp / len(gt)
    return 2 * precision * recall / (precision + recall)


def exact_match(pred: set, gt: set) -> float:
    """Perfect-set-match accuracy (1.0 if the label sets are identical)."""
    return 1.0 if pred == gt else 0.0


def class_macro_f1(pairs: list) -> float:
    """Class-macro F1 over a list of (pred_set, gt_set) pairs.

    The class universe is the union of all ground-truth labels (the standard
    CLIMB convention — only classes that actually occur in the references are
    scored). For each class we accumulate tp/fp/fn across all samples, compute
    its F1, then average over classes. Returns 0.0 if no gt labels exist.
    """
    classes = set()
    for _, gt in pairs:
        classes |= gt
    if not classes:
        return 0.0
    f1s = []
    for c in classes:
        tp = fp = fn = 0
        for pred, gt in pairs:
            in_p, in_g = (c in pred), (c in gt)
            if in_p and in_g:
                tp += 1
            elif in_p and not in_g:
                fp += 1
            elif in_g and not in_p:
                fn += 1
        denom = 2 * tp + fp + fn
        f1s.append((2 * tp / denom) if denom else 0.0)
    return sum(f1s) / len(f1s)
