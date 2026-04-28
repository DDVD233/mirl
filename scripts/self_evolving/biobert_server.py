"""BioBERT embedding-similarity server.

A small FastAPI service that loads a BioBERT-based sentence-transformer
model once and serves cosine-similarity queries. Used by the self-evolving
reward function to add a smooth semantic-similarity signal between the
model's extracted answer and the ground truth, complementing the discrete
exact-match accuracy.

Endpoints
---------
GET  /healthz                  -> {"ok": true, "model": <name>}
POST /v1/similarity            -> {"similarity": float}    body: {text1, text2}
POST /v1/similarity_batch      -> {"similarities": [...]}  body: {pairs: [[a,b], ...]}

Run with::

    BIOBERT_MODEL=pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb \\
    BIOBERT_PORT=8003 \\
    BIOBERT_DEVICE=cuda \\
    python scripts/self_evolving/biobert_server.py
"""

from __future__ import annotations

import os
from typing import List

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import torch
import uvicorn


MODEL_NAME = os.environ.get(
    "BIOBERT_MODEL",
    "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
)
DEVICE = os.environ.get("BIOBERT_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
PORT = int(os.environ.get("BIOBERT_PORT", "8003"))
HOST = os.environ.get("BIOBERT_HOST", "0.0.0.0")


print(f"[biobert] loading {MODEL_NAME} on {DEVICE} ...", flush=True)
model = SentenceTransformer(MODEL_NAME, device=DEVICE)
print("[biobert] ready", flush=True)


app = FastAPI(title="BioBERT similarity")


class SimRequest(BaseModel):
    text1: str
    text2: str


class BatchSimRequest(BaseModel):
    pairs: List[List[str]]


def _cosine_pair(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    # embeddings already L2-normalized by SentenceTransformer
    return float(np.dot(emb_a, emb_b))


@app.get("/healthz")
def healthz():
    return {"ok": True, "model": MODEL_NAME, "device": DEVICE}


@app.post("/v1/similarity")
def similarity(req: SimRequest):
    if not req.text1 or not req.text2:
        return {"similarity": 0.0}
    embs = model.encode(
        [req.text1, req.text2],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return {"similarity": _cosine_pair(embs[0], embs[1])}


@app.post("/v1/similarity_batch")
def similarity_batch(req: BatchSimRequest):
    pairs = req.pairs or []
    if not pairs:
        return {"similarities": []}
    flat: list[str] = []
    for p in pairs:
        if len(p) != 2:
            return {"error": "each pair must have exactly two strings"}
        flat.append(p[0])
        flat.append(p[1])
    embs = model.encode(
        flat,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
        batch_size=64,
    )
    sims = [
        _cosine_pair(embs[2 * i], embs[2 * i + 1]) if pairs[i][0] and pairs[i][1] else 0.0
        for i in range(len(pairs))
    ]
    return {"similarities": sims}


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
