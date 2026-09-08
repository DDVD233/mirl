"""Reproduce an unscored failed case and retain retrieval/planner diagnostics."""

import argparse
import asyncio
import json
import math
from collections import Counter
from pathlib import Path

import aiohttp
import mimic_method_baselines as baseline


async def diagnose(args):
    root = Path(args.run)
    manifest = json.loads((root / "manifest.json").read_text())
    config = argparse.Namespace(**manifest["config"])
    config.retries = args.retries
    entries = baseline.read_rows(Path(config.val_file))
    entry = next(row for row in entries if str(row["extra_info"]["hadm_id"]) == args.case)
    helper = baseline.load_module("diagnostic_eval", config.eval_module)
    reward = baseline.load_module("diagnostic_reward", config.reward_file)
    body, _ = baseline.build_request(entry, helper, config)
    trace, searches = [], []

    class Chat(baseline.Chat):
        async def call(self, messages, internal_trace, label, **kwargs):
            result = await super().call(messages, internal_trace, label, **kwargs)
            trace.append(internal_trace[-1])
            return result

    class Retriever(baseline.Retriever):
        async def search(self, query, internal_trace):
            try:
                return await super().search(query, internal_trace)
            except ValueError:
                async with self.session.post(
                    config.embed_base + "/embeddings", json={"model": config.embed_model, "input": [query]}
                ) as response:
                    response.raise_for_status()
                    vec = (await response.json())["data"][0]["embedding"]
                audit = {"query": query, "vector_norm": math.sqrt(sum(x * x for x in vec)), "searches": []}
                for limit in (32, 128, 512):
                    hits = await asyncio.to_thread(
                        self.client.search,
                        collection_name=config.collection,
                        data=[vec],
                        limit=limit,
                        filter=self.filter,
                        output_fields=["entry_id", "source_dataset", "text_content", "question", "answer"],
                        search_params={"metric_type": "COSINE", "params": {"ef": max(128, limit)}},
                    )
                    entities = [hit["entity"] for hit in hits[0]]
                    texts = [
                        entity.get("text_content")
                        or "\n".join(str(entity.get(key) or "") for key in ("question", "answer"))
                        for entity in entities
                    ]
                    audit["searches"].append(
                        {
                            "limit": limit,
                            "raw_count": len(entities),
                            "source_counts": dict(Counter(e["source_dataset"] for e in entities)),
                            "lengths": [len(text.strip()) for text in texts],
                            "usable": sum(len(text.strip()) >= 80 for text in texts),
                            "examples": entities[:3],
                        }
                    )
                searches.append(audit)
                raise

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=240)) as session:
        chat = Chat(session, config)
        retriever = Retriever(session, config) if args.method != "cove" else None
        error = None
        try:
            await baseline.generate(body["messages"], args.method, chat, retriever, config, reward.extract_final_answer)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            if retriever:
                retriever.client.close()
        baseline.write_json(
            Path(args.output),
            {"case": args.case, "method": args.method, "error": error, "trace": trace, "retrieval_failures": searches},
        )
        print(
            json.dumps(
                {
                    "method": args.method,
                    "error": error,
                    "calls": [
                        {
                            "label": call["label"],
                            "chars": len(call["response"]),
                            "finish_reason": call["finish_reason"],
                            "usage": call["usage"],
                        }
                        for call in trace
                    ],
                    "retrieval": [
                        {
                            "norm": s["vector_norm"],
                            "searches": [
                                {k: v for k, v in x.items() if k not in ("examples", "lengths")} for x in s["searches"]
                            ],
                        }
                        for s in searches
                    ],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--retries", type=int, default=4)
    asyncio.run(diagnose(parser.parse_args()))
