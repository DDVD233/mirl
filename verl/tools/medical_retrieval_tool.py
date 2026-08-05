# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""Medical knowledge retrieval tool for the self-evolving HealthBench RL run.

The rollout policy calls ``search_medical_kb`` with a QUERY PLAN — 1..4 sub-queries
covering different facets of the request — which this tool forwards to the
generation server's ``POST /retrieve``. That endpoint batch-embeds the plan,
searches the read-only ``medical_knowledge_v2`` Milvus collection once per
sub-query, merges round-robin with dedup, and compresses the result into a
question-conditioned evidence brief.

Why a plan and not one query: measured on 93 rubric criteria whose supporting fact
IS in the KB, a single query supplies it 10.8% of the time vs 35.9% for two
(McNemar p<1e-4). A HealthBench task grades 2-5 distinct facts and one query can
only fetch one of them.

The endpoint lives in ``scripts/self_evolving/generation_server.py`` and is
colocated with the trainer (default ``http://localhost:8006/retrieve``), so the
tool holds no Milvus/embedding connections of its own.
"""

import json
import logging
import os
from typing import Any, Optional
from uuid import uuid4

import httpx

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class MedicalRetrievalTool(BaseTool):
    """Stateless retrieval tool backed by the gen server's ``/retrieve`` route.

    Config keys (all optional):
        retrieval_url: full URL of the /retrieve endpoint. Falls back to the
            ``RETRIEVAL_URL`` env var, then ``http://localhost:8006/retrieve``.
        topk: passages fetched PER sub-query. Omit to let the server decide —
            retrieval depth is a server-side policy knob and pinning it here is how
            it previously diverged from the tuned value.
        total: merged passage budget across the whole plan (server default 16).
        max_queries: hard cap on sub-queries per call (default 4).
        timeout: per-request timeout in seconds (default 60 — must exceed the
            server's summarizer budget or every summarized call times out here).
    """

    MAX_QUERIES_DEFAULT = 4

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.retrieval_url = (
            config.get("retrieval_url")
            or os.getenv("RETRIEVAL_URL")
            or "http://localhost:8006/retrieve"
        )
        self.topk = config.get("topk")  # None => server policy
        self.total = config.get("total")
        self.max_queries = int(config.get("max_queries", self.MAX_QUERIES_DEFAULT))
        self.timeout = float(config.get("timeout", 60))

    def _coerce_queries(self, parameters: dict[str, Any]) -> list[str]:
        """Accept a query plan in every shape the model might emit.

        The qwen3_coder XML parser hands every ``<parameter=...>`` body over as a
        string and only reaches ``ast.literal_eval`` for non-scalar declared types,
        so a plan can arrive as a real list, a JSON string, or newline-separated
        text. All three are normal; none of them should cost a rollout.
        """
        raw = parameters.get("queries")
        if raw is None:
            raw = parameters.get("query")
        if raw is None:
            return []
        if isinstance(raw, str):
            s = raw.strip()
            if s.startswith("["):
                try:
                    parsed = json.loads(s)
                    raw = parsed if isinstance(parsed, list) else [s]
                except Exception:
                    raw = s.splitlines()
            else:
                raw = s.splitlines()
        if not isinstance(raw, (list, tuple)):
            raw = [raw]
        out, seen = [], set()
        for q in raw:
            q = str(q).strip().lstrip("-*0123456789. ").strip('"').strip()
            if len(q) >= 4 and q.lower() not in seen:
                seen.add(q.lower())
                out.append(q)
        return out[: self.max_queries]

    @staticmethod
    def _question_of(agent_data) -> str:
        """The clinician request, for question-conditioned summarization."""
        for msg in reversed(getattr(agent_data, "messages", None) or []):
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return " ".join(
                    p.get("text", "") for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                ).strip()
        return ""

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        return instance_id or str(uuid4()), ToolResponse()

    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        # `metrics` (3rd element) is how retrieval telemetry and the evidence text
        # reach the agent loop -> extra_fields -> the reward's coverage grader. It
        # must be populated on EVERY path, including the failures.
        queries = self._coerce_queries(parameters)
        if not queries:
            return (
                ToolResponse(text="No search query provided. Provide 1-4 English sub-queries."),
                0.0,
                {"retrieval_hits": 0, "retrieval_error": 1, "queries": [], "retrieval_text": ""},
            )
        body: dict[str, Any] = {"queries": queries}
        question = self._question_of(kwargs.get("agent_data"))
        if question:
            body["question"] = question[:4000]
        if self.topk:
            body["top_k"] = int(self.topk)
        if self.total:
            body["total"] = int(self.total)
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(self.retrieval_url, json=body)
                resp.raise_for_status()
                data = resp.json()
        except Exception as e:  # retrieval must never crash a rollout
            logger.warning(
                f"medical retrieval failed for {queries[0][:60]!r} "
                f"({len(queries)} queries): {type(e).__name__}: {e!r}"
            )
            return (
                ToolResponse(text="Retrieval is temporarily unavailable; answer from your own knowledge."),
                0.0,
                {"retrieval_hits": 0, "retrieval_error": 1, "queries": queries, "retrieval_text": ""},
            )
        text = data.get("text") or "No relevant passages found in the medical knowledge base."
        return (
            ToolResponse(text=text),
            0.0,
            {
                "retrieval_hits": len(data.get("passages") or []),
                "retrieval_error": 0,
                "queries": data.get("queries") or queries,
                "retrieval_text": text,
                "summarized": 1 if data.get("summarized") else 0,
            },
        )
