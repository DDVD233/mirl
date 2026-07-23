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

The rollout policy calls ``search_medical_kb`` with a natural-language ``query``;
this tool forwards it to the generation server's ``POST /retrieve`` endpoint
(which embeds the query and searches the read-only ``medical_knowledge_v2``
Milvus collection) and returns the formatted passages. Grounding the solver's
final answer in retrieved evidence targets HealthBench-Professional failures
that stem from missing clinical knowledge rather than reasoning.

The endpoint lives in ``scripts/self_evolving/generation_server.py`` and is
colocated with the trainer (default ``http://localhost:8006/retrieve``), so the
tool holds no Milvus/embedding connections of its own.
"""

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
        topk: number of passages to request (default 5).
        timeout: per-request timeout in seconds (default 30).
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.retrieval_url = (
            config.get("retrieval_url")
            or os.getenv("RETRIEVAL_URL")
            or "http://localhost:8006/retrieve"
        )
        self.topk = int(config.get("topk", 5))
        self.timeout = float(config.get("timeout", 30))

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        return instance_id or str(uuid4()), ToolResponse()

    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        query = parameters.get("query")
        if not isinstance(query, str) or not query.strip():
            return ToolResponse(text="No search query provided."), 0.0, {"retrieval_hits": 0}
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    self.retrieval_url, json={"query": query, "top_k": self.topk}
                )
                resp.raise_for_status()
                data = resp.json()
        except Exception as e:  # retrieval must never crash a rollout
            logger.warning(f"medical retrieval failed for {query[:60]!r}: {type(e).__name__}: {e!r}")
            return (
                ToolResponse(text="Retrieval is temporarily unavailable; answer from your own knowledge."),
                0.0,
                {"retrieval_hits": 0, "retrieval_error": 1},
            )
        text = data.get("text") or "No relevant passages found in the medical knowledge base."
        n_hits = len(data.get("passages") or [])
        return ToolResponse(text=text), 0.0, {"retrieval_hits": n_hits}
