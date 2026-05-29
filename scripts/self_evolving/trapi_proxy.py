"""OpenAI-compatible reverse proxy that fronts TRAPI with refreshing AAD auth.

Why this exists
---------------
TRAPI authenticates with Microsoft Entra ID (AAD) tokens that expire ~hourly.
The verl OPD teacher client (`external_client.py`) and the reward/judge path
both send a *static* `Authorization: Bearer <api_key>` header set once at
startup, so they 401 mid-run against TRAPI.

This proxy runs on the host that holds the SC-ALT `az login` (point.dd.works)
and is the only thing that touches AAD: it resolves a fresh token per request
(cached + auto-refreshed near expiry) and injects it on the way upstream.
Downstream callers keep their existing workflow unchanged — they talk plain
OpenAI to the proxy with API_KEY=EMPTY, exactly like the local vLLM teacher:

    export TEACHER_URL=http://point.dd.works:18890/v1
    export API_BASE=http://point.dd.works:18890/v1
    export TEACHER_API_KEY=EMPTY            # ignored by the proxy

The proxy is a transparent pass-through: it forwards method, path (minus the
local /v1 prefix), query, and body to TRAPI and streams the response back as
is. TRAPI already returns standard OpenAI shape plus vLLM's `prompt_logprobs`
extension, so no format conversion is needed — token-id prompts and
prompt_logprobs flow through untouched.

Run
---
    pip install fastapi uvicorn httpx azure-identity     # all in `svl`
    az login --scope api://trapi/.default
    python scripts/self_evolving/trapi_proxy.py
    # or: uvicorn-style knobs via env (see below)

Env
---
    TRAPI_UPSTREAM   upstream base incl. instance + /openai/v1
                     (default redmond/interactive)
    TRAPI_SCOPE      AAD scope (default api://trapi/.default)
    PROXY_HOST       listen host (default 0.0.0.0)
    PROXY_PORT       listen port (default 18890)
    PROXY_LOG_LEVEL  uvicorn log level (default info)
"""

from __future__ import annotations

import asyncio
import hmac
import logging
import os
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

UPSTREAM = os.environ.get(
    "TRAPI_UPSTREAM",
    "https://trapi.research.microsoft.com/redmond/interactive/openai/v1",
).rstrip("/")
SCOPE = os.environ.get("TRAPI_SCOPE", "api://trapi/.default")
HOST = os.environ.get("PROXY_HOST", "0.0.0.0")
PORT = int(os.environ.get("PROXY_PORT", "18890"))
# Optional shared secret that clients must present (Authorization: Bearer <key>
# or api-key: <key>). Gates access to our TRAPI quota since the proxy listens
# on 0.0.0.0. If unset/empty, the proxy accepts any caller (open mode).
PROXY_API_KEY = os.environ.get("PROXY_API_KEY", "").strip()
# Refresh when the cached token is within this many seconds of expiry.
REFRESH_SKEW_S = int(os.environ.get("TRAPI_REFRESH_SKEW_S", "300"))
# Total per-request timeout to TRAPI. Long prefills (prompt_logprobs over 12k
# tokens) can take a while, so keep this generous — matches the teacher client.
UPSTREAM_TIMEOUT_S = float(os.environ.get("TRAPI_UPSTREAM_TIMEOUT", "600"))
# Cap concurrent in-flight upstream requests so a per-step reward burst
# (batch*n judge calls) doesn't slam TRAPI's APIM gateway all at once.
MAX_CONCURRENCY = int(os.environ.get("TRAPI_MAX_CONCURRENCY", "96"))
# Retry 429/5xx upstream responses with backoff so clients never see them.
# We decide to retry from the status line *before* streaming any body, so this
# is safe for both streamed (SSE) and buffered responses.
MAX_RETRIES = int(os.environ.get("TRAPI_MAX_RETRIES", "6"))
RETRY_BASE_S = float(os.environ.get("TRAPI_RETRY_BASE_S", "1.0"))
RETRY_STATUSES = {429, 500, 502, 503, 504}

# Hop-by-hop headers must not be forwarded (RFC 7230 §6.1). We also drop host
# (set by httpx), authorization (we inject our own), and content-length /
# transfer-encoding (recomputed by the respective layers).
_DROP_REQUEST_HEADERS = {
    "host", "authorization", "content-length", "connection",
    "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade",
}
_DROP_RESPONSE_HEADERS = {
    "content-length", "connection", "keep-alive", "transfer-encoding",
    "proxy-authenticate", "proxy-authorization", "te", "trailers", "upgrade",
}

logger = logging.getLogger("trapi_proxy")


def _client_key(request: Request) -> str:
    """Extract the client-presented key from Authorization or api-key header."""
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[len("bearer "):].strip()
    return request.headers.get("api-key", "").strip()


def _authorized(request: Request) -> bool:
    if not PROXY_API_KEY:  # open mode — no client key configured
        return True
    return hmac.compare_digest(_client_key(request), PROXY_API_KEY)


class TokenCache:
    """Caches an AAD token and refreshes it (under a lock) near expiry.

    `AzureCliCredential.get_token` is synchronous (it shells out to `az`), so
    we run it in a thread to avoid blocking the event loop. The refresh only
    actually fires ~once an hour; every other call returns the cached string.
    """

    def __init__(self, scope: str):
        from azure.identity import (
            AzureCliCredential,
            ChainedTokenCredential,
            ManagedIdentityCredential,
        )

        self._scope = scope
        self._credential = ChainedTokenCredential(
            AzureCliCredential(), ManagedIdentityCredential()
        )
        self._token = None  # azure.core.credentials.AccessToken
        self._lock = asyncio.Lock()

    def _fresh_enough(self) -> bool:
        return (
            self._token is not None
            and self._token.expires_on - time.time() > REFRESH_SKEW_S
        )

    async def token(self) -> str:
        if self._fresh_enough():
            return self._token.token
        async with self._lock:
            if self._fresh_enough():  # double-check after acquiring the lock
                return self._token.token
            loop = asyncio.get_running_loop()
            self._token = await loop.run_in_executor(
                None, self._credential.get_token, self._scope
            )
            ttl = int(self._token.expires_on - time.time())
            logger.info("refreshed TRAPI token (valid ~%ds)", ttl)
            return self._token.token


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.tokens = TokenCache(SCOPE)
    app.state.client = httpx.AsyncClient(
        timeout=httpx.Timeout(UPSTREAM_TIMEOUT_S, connect=30.0),
        # Generous pool: the trainer bursts batch_size*n teacher calls per step.
        limits=httpx.Limits(max_connections=256, max_keepalive_connections=64),
    )
    app.state.sem = asyncio.Semaphore(MAX_CONCURRENCY)
    # Fail fast & warm the cache so the first proxied request isn't slow.
    try:
        await app.state.tokens.token()
        auth_mode = "client key REQUIRED" if PROXY_API_KEY else "OPEN (no client key)"
        logger.info(
            "TRAPI proxy ready: %s  ->  %s  [auth: %s]",
            f"{HOST}:{PORT}", UPSTREAM, auth_mode,
        )
        if not PROXY_API_KEY:
            logger.warning(
                "PROXY_API_KEY is unset — anyone who can reach this port can "
                "spend our TRAPI quota. Set PROXY_API_KEY before exposing externally."
            )
    except Exception as e:  # noqa: BLE001
        logger.error(
            "Could not acquire an AAD token at startup (%s: %s). "
            "Run: az login --scope %s",
            type(e).__name__, e, SCOPE,
        )
    try:
        yield
    finally:
        await app.state.client.aclose()


app = FastAPI(title="TRAPI proxy", lifespan=lifespan)


@app.get("/health")
async def health() -> JSONResponse:
    try:
        await app.state.tokens.token()
        return JSONResponse({"status": "ok", "upstream": UPSTREAM})
    except Exception as e:  # noqa: BLE001
        return JSONResponse(
            {"status": "auth_error", "error": f"{type(e).__name__}: {e}"},
            status_code=503,
        )


@app.api_route(
    "/{full_path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
)
async def proxy(full_path: str, request: Request) -> Response:
    # Local clients hit /v1/<x>; TRAPI's UPSTREAM already ends in /openai/v1,
    # so strip the leading /v1 and append the remainder (/<x>).
    if not _authorized(request):
        return JSONResponse(
            {"error": {"message": "Invalid proxy API key.", "type": "invalid_request_error",
                       "code": "invalid_api_key"}},
            status_code=401,
        )

    path = "/" + full_path
    if path.startswith("/v1/"):
        path = path[len("/v1"):]
    elif path == "/v1":
        path = "/"
    upstream_url = f"{UPSTREAM}{path}"

    headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in _DROP_REQUEST_HEADERS
    }
    try:
        headers["Authorization"] = f"Bearer {await request.app.state.tokens.token()}"
    except Exception as e:  # noqa: BLE001
        return JSONResponse(
            {"error": {"message": f"TRAPI auth failed: {type(e).__name__}: {e}",
                       "type": "proxy_auth_error"}},
            status_code=503,
        )

    body = await request.body()
    client: httpx.AsyncClient = request.app.state.client
    sem: asyncio.Semaphore = request.app.state.sem

    # Stream the upstream response so SSE (stream=true chat) passes through and
    # large prompt_logprobs bodies don't have to be buffered whole. We inspect
    # the status line before reading the body, so retrying on 429/5xx is safe
    # for streamed responses too. A semaphore caps concurrent upstream calls.
    upstream = None
    backoff = RETRY_BASE_S
    for attempt in range(MAX_RETRIES + 1):
        req = client.build_request(
            request.method,
            upstream_url,
            params=request.query_params,
            headers=headers,
            content=body,
        )
        try:
            async with sem:
                upstream = await client.send(req, stream=True)
        except httpx.HTTPError as e:
            if attempt < MAX_RETRIES:
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
                continue
            return JSONResponse(
                {"error": {"message": f"upstream request failed: {type(e).__name__}: {e}",
                           "type": "proxy_upstream_error"}},
                status_code=502,
            )
        if upstream.status_code in RETRY_STATUSES and attempt < MAX_RETRIES:
            retry_after = upstream.headers.get("retry-after")
            delay = float(retry_after) if retry_after and retry_after.isdigit() else backoff
            await upstream.aclose()
            await asyncio.sleep(delay)
            backoff = min(backoff * 2, 30.0)
            continue
        break

    resp_headers = {
        k: v for k, v in upstream.headers.items()
        if k.lower() not in _DROP_RESPONSE_HEADERS
    }

    async def body_iter():
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await upstream.aclose()

    return StreamingResponse(
        body_iter(),
        status_code=upstream.status_code,
        headers=resp_headers,
        media_type=upstream.headers.get("content-type"),
    )


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level=os.environ.get("PROXY_LOG_LEVEL", "info"),
    )
