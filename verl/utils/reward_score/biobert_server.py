"""Standalone BioBERT embedding similarity server.

Usage:
    python -m verl.utils.reward_score.biobert_server [--port 5100] [--device cuda]

The reward function in mimic.py queries this server instead of loading
BioBERT inside the Ray worker, avoiding torch dispatch mode conflicts.

Uses FastAPI + uvicorn. Concurrent requests are handled by the async
event loop but GPU inference is serialized via a lock to avoid conflicts.
"""

import argparse
import asyncio
import time
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer

_model = None
_tokenizer = None
_device = None
_lock = asyncio.Lock()


def _load_model(device: str):
    global _model, _tokenizer, _device
    _device = torch.device(device)
    model_name = "medicalai/ClinicalBERT"
    _tokenizer = AutoTokenizer.from_pretrained(model_name)
    _model = AutoModel.from_pretrained(model_name).to(_device)
    _model.eval()
    print(f"BioBERT loaded on {_device}")


def _embed(text: str):
    inputs = _tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(_device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = _model(**inputs)
    mask = inputs["attention_mask"].unsqueeze(-1).float()
    embedding = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
    return embedding.squeeze(0)


def _compute_similarity(pred_text: str, gt_text: str) -> float:
    pred_emb = _embed(pred_text)
    gt_emb = _embed(gt_text)
    cos_sim = torch.nn.functional.cosine_similarity(
        pred_emb.unsqueeze(0), gt_emb.unsqueeze(0)
    ).item()
    return max(0.0, cos_sim)


class SimilarityRequest(BaseModel):
    pred: str
    gt: str


app = FastAPI()


@app.post("/similarity")
async def similarity(req: SimilarityRequest):
    if not req.pred or not req.gt:
        return {"similarity": 0.0, "time": 0.0}

    async with _lock:
        t0 = time.time()
        result = _compute_similarity(req.pred, req.gt)
        elapsed = time.time() - t0

    print(f"[biobert_server] sim={result:.4f}, time={elapsed:.3f}s")
    return {"similarity": result, "time": elapsed}


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5100)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    _load_model(args.device)
    # Single worker to keep one model on GPU
    uvicorn.run(app, host="0.0.0.0", port=args.port, workers=1)
