"""
CLIMB media file server.

A tiny authenticated HTTP file server for the multimodal CLIMB dataset. It runs
on the **local node** (``mib``) where ``/scratch/high_modality`` physically
lives, and serves the image/video files to the two remote consumers that cannot
see that disk:

  * the generation server's teacher (fetches images to synthesize multimodal
    questions), and
  * the trainer's dataset (fetches images/videos for the training/eval rows).

Both authenticate with the SAME key as the chat/teacher API, supplied via the
``Authorization: Bearer <token>`` header and checked against the environment
variable ``CLIMB_FILE_TOKEN``. The token is NEVER read from a file or baked into
source — only from the environment.

Endpoints
---------
GET /healthz                       liveness (no auth)
GET /file/{relpath}?max_pixels=N   stream a file under $CLIMB_DATA_ROOT.
                                   For images, optional server-side downscale to
                                   `max_pixels` (saves bandwidth and matches the
                                   resolution the student trains at).

Run with `serve/start_climb_file_server.sh`.
"""

import argparse
import io
import mimetypes
import os

import uvicorn
from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse, Response

DATA_ROOT = os.path.realpath(os.environ.get("CLIMB_DATA_ROOT", "/scratch/high_modality"))
TOKEN_ENV = os.environ.get("CLIMB_FILE_TOKEN_ENV", "CLIMB_FILE_TOKEN")

app = FastAPI()


def _expected_token() -> str:
    # Read on every request so the operator can rotate the key without a
    # restart, and so importing this module never captures a secret.
    return os.environ.get(TOKEN_ENV, "")


def _check_auth(authorization: str | None) -> None:
    expected = _expected_token()
    if not expected:
        # Fail closed: if the server has no token configured, refuse everything
        # rather than silently serving the dataset unauthenticated.
        raise HTTPException(status_code=503, detail="server token not configured")
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    if authorization[len("Bearer "):].strip() != expected:
        raise HTTPException(status_code=403, detail="invalid token")


def _safe_path(relpath: str) -> str:
    """Resolve `relpath` under DATA_ROOT, rejecting any traversal that escapes
    the root (symlinks included, via realpath)."""
    rel = relpath.lstrip("/")
    full = os.path.realpath(os.path.join(DATA_ROOT, rel))
    if full != DATA_ROOT and not full.startswith(DATA_ROOT + os.sep):
        raise HTTPException(status_code=403, detail="path outside data root")
    if not os.path.isfile(full):
        raise HTTPException(status_code=404, detail="not found")
    return full


@app.get("/healthz")
async def healthz():
    return {"ok": True, "data_root": DATA_ROOT}


@app.get("/file/{relpath:path}")
async def get_file(
    relpath: str,
    max_pixels: int | None = Query(default=None, ge=1),
    authorization: str | None = Header(default=None),
):
    _check_auth(authorization)
    full = _safe_path(relpath)
    mime = mimetypes.guess_type(full)[0] or "application/octet-stream"

    # Image + downscale request: re-encode to fit within max_pixels (PIL), so
    # the wire payload and the resolution the teacher/student see match the
    # dataset's max_pixels. Mirrors `_image_to_data_uri` in generation_server.py.
    if max_pixels and mime.startswith("image/"):
        try:
            from PIL import Image

            im = Image.open(full).convert("RGB")
            if im.width * im.height > max_pixels:
                scale = (max_pixels / float(im.width * im.height)) ** 0.5
                im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))))
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            return Response(content=buf.getvalue(), media_type="image/png")
        except Exception:
            # Fall back to the raw file rather than failing the request.
            pass

    return FileResponse(full, media_type=mime)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18080)
    args = parser.parse_args()
    if not os.path.isdir(DATA_ROOT):
        raise SystemExit(f"CLIMB_DATA_ROOT does not exist: {DATA_ROOT}")
    if not _expected_token():
        raise SystemExit(
            f"environment variable {TOKEN_ENV} is empty — refusing to start "
            f"(set it to the teacher API key value)."
        )
    print(f"CLIMB file server: root={DATA_ROOT} port={args.port} token_env={TOKEN_ENV}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
