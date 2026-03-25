"""
Authenticated HTTP forward proxy for Google Cloud APIs.

Deployed on render.com (US/EU) so Russian-hosted servers can reach
Google APIs through a non-restricted IP.

Pipeline Google client libraries must use REST transport (not gRPC):
  SpeechClient(transport="rest")
  TextToSpeechClient(transport="rest")
Gemini SDK already uses REST.

Pipeline sets env vars:
  GOOGLE_PROXY_URL=https://your-proxy.onrender.com
  GOOGLE_PROXY_TOKEN=your-secret
"""

import logging
import os

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse, Response

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-12s %(levelname)-5s %(message)s",
)
logger = logging.getLogger("google-proxy")

PROXY_SECRET = os.environ.get("PROXY_SECRET", "change-me")

ALLOWED_HOSTS = {
    "speech.googleapis.com",
    "texttospeech.googleapis.com",
    "generativelanguage.googleapis.com",
    "oauth2.googleapis.com",
    "www.googleapis.com",
}

app = FastAPI(title="Google API Proxy", docs_url=None, redoc_url=None)


def _check_auth(request: Request) -> bool:
    token = request.headers.get("x-proxy-token", "")
    return token == PROXY_SECRET


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.api_route(
    "/proxy/{target_host}/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
)
async def forward_request(request: Request, target_host: str, path: str):
    """Forward request to target Google API host.

    URL format: /proxy/{host}/{original_path}
    Example:    /proxy/speech.googleapis.com/v2/projects/123/locations/global/recognizers/_:recognize
    """
    if not _check_auth(request):
        return PlainTextResponse("Unauthorized", status_code=401)

    if target_host not in ALLOWED_HOSTS:
        return PlainTextResponse(f"Host not allowed: {target_host}", status_code=403)

    url = f"https://{target_host}/{path}"
    if request.url.query:
        url += f"?{request.url.query}"

    # Forward headers, strip proxy-specific and hop-by-hop
    skip = {"host", "x-proxy-token", "transfer-encoding", "connection", "content-length"}
    fwd_headers = {
        k: v for k, v in request.headers.items() if k.lower() not in skip
    }

    body = await request.body()

    logger.info("%s %s (%d bytes)", request.method, url[:120], len(body))

    try:
        async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
            resp = await client.request(
                method=request.method,
                url=url,
                headers=fwd_headers,
                content=body,
            )

        resp_headers = {
            k: v for k, v in resp.headers.items()
            if k.lower() not in ("transfer-encoding", "connection", "content-encoding")
        }
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            headers=resp_headers,
        )
    except httpx.ConnectError as e:
        logger.error("Upstream connect error: %s", e)
        return PlainTextResponse(f"Upstream error: {e}", status_code=502)
    except httpx.TimeoutException:
        return PlainTextResponse("Upstream timeout", status_code=504)


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info")
