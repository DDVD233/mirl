"""Standalone BioBERT embedding similarity server.

Usage:
    python -m verl.utils.reward_score.biobert_server [--port 5100] [--device cuda]

The reward function in mimic.py queries this server instead of loading
BioBERT inside the Ray worker, avoiding torch dispatch mode conflicts.
"""

import argparse
import time

import torch
from flask import Flask, jsonify, request
from transformers import AutoModel, AutoTokenizer

app = Flask(__name__)

_model = None
_tokenizer = None
_device = None


def _load_model(device: str):
    global _model, _tokenizer, _device
    _device = torch.device(device)
    model_name = "dmis-lab/biobert-base-cased-v1.2"
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


@app.route("/similarity", methods=["POST"])
def similarity():
    data = request.get_json()
    pred_text = data.get("pred", "")
    gt_text = data.get("gt", "")

    if not pred_text or not gt_text:
        return jsonify({"similarity": 0.0, "time": 0.0})

    t0 = time.time()
    pred_emb = _embed(pred_text)
    gt_emb = _embed(gt_text)
    cos_sim = torch.nn.functional.cosine_similarity(
        pred_emb.unsqueeze(0), gt_emb.unsqueeze(0)
    ).item()
    result = max(0.0, cos_sim)
    elapsed = time.time() - t0

    return jsonify({"similarity": result, "time": elapsed})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5100)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    _load_model(args.device)
    app.run(host="0.0.0.0", port=args.port)
