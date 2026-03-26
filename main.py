"""
Authenticated HTTP forward proxy for Google Cloud APIs.

Deployed on render.com (US/EU) so Russian-hosted servers can reach
Google APIs through a non-restricted IP.

Supports:
  1. REST forwarding — /proxy/{host}/{path}  (all Google REST APIs)
  2. WebSocket gRPC bridge — /ws/speech-stream  (STT v2 streaming recognition)
  3. Diagnostics — /diagnostics  (latency to each Google service)

Pipeline sets env vars:
  GOOGLE_PROXY_URL=https://your-proxy.onrender.com
  GOOGLE_PROXY_TOKEN=your-secret
"""

import asyncio
import json
import logging
import os
import re
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse, Response

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-12s %(levelname)-5s %(message)s",
)
logger = logging.getLogger("google-proxy")

PROXY_SECRET = os.environ.get("PROXY_SECRET", "change-me")

ALLOWED_HOSTS = {
    # Global endpoints
    "speech.googleapis.com",
    "texttospeech.googleapis.com",
    "generativelanguage.googleapis.com",
    "oauth2.googleapis.com",
    "www.googleapis.com",
    # EU regional endpoints
    "eu-speech.googleapis.com",
    "eu-texttospeech.googleapis.com",
    # Regional endpoints (europe-west3 = Frankfurt, europe-west4 = Netherlands)
    "europe-west3-speech.googleapis.com",
    "europe-west4-speech.googleapis.com",
    "europe-west3-texttospeech.googleapis.com",
    "europe-west4-texttospeech.googleapis.com",
    # Vertex AI regional (for Gemini via Vertex)
    "europe-west1-aiplatform.googleapis.com",
    "europe-west3-aiplatform.googleapis.com",
    "europe-west4-aiplatform.googleapis.com",
}

# Map: when the proxy is in EU, rewrite global endpoints to EU ones.
# Set PROXY_REGION=eu to enable. Clients can still use global endpoints —
# the proxy will transparently route to the nearest regional endpoint.
PROXY_REGION = os.environ.get("PROXY_REGION", "")  # "", "eu", "europe-west3", etc.

EU_ENDPOINT_REWRITES = {
    "speech.googleapis.com": "eu-speech.googleapis.com",
    "texttospeech.googleapis.com": "eu-texttospeech.googleapis.com",
}

REGIONAL_ENDPOINT_REWRITES = {
    "speech.googleapis.com": "{region}-speech.googleapis.com",
    "texttospeech.googleapis.com": "{region}-texttospeech.googleapis.com",
}

# Default Vertex AI region when PROXY_REGION is "eu" (confirmed working)
_VERTEX_EU_DEFAULT_REGION = "europe-west1"

# Regex to extract model name and action from generativelanguage.googleapis.com paths
# e.g. "v1beta/models/gemini-2.5-flash:streamGenerateContent" -> ("gemini-2.5-flash", ":streamGenerateContent")
_GEMINI_MODEL_RE = re.compile(r"v1(?:beta)?/models/([^/?]+)")


def _resolve_host(target_host: str) -> str:
    """Rewrite global host to regional if PROXY_REGION is set."""
    if not PROXY_REGION or target_host not in EU_ENDPOINT_REWRITES:
        return target_host

    if PROXY_REGION == "eu":
        return EU_ENDPOINT_REWRITES.get(target_host, target_host)

    # Specific region like "europe-west3"
    template = REGIONAL_ENDPOINT_REWRITES.get(target_host)
    if template:
        return template.format(region=PROXY_REGION)

    return target_host


def _vertex_region() -> str:
    """Return the Vertex AI region based on PROXY_REGION."""
    if PROXY_REGION == "eu":
        return _VERTEX_EU_DEFAULT_REGION
    return PROXY_REGION


def _rewrite_gemini_to_vertex(path: str, query: str, project_id: str) -> str | None:
    """Rewrite a generativelanguage.googleapis.com path to Vertex AI format.

    Returns the full URL if rewriting is possible, None otherwise.

    Input path example:
        v1beta/models/gemini-2.5-flash:streamGenerateContent
    Output URL:
        https://{region}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{region}/publishers/google/models/gemini-2.5-flash:streamGenerateContent
    """
    if not PROXY_REGION or not project_id:
        return None

    match = _GEMINI_MODEL_RE.search(path)
    if not match:
        return None

    # model_and_action includes the model name and any :action suffix
    # e.g. "gemini-2.5-flash:streamGenerateContent" or "gemini-2.5-flash"
    model_and_action = match.group(1)

    region = _vertex_region()
    vertex_host = f"{region}-aiplatform.googleapis.com"
    vertex_path = (
        f"v1/projects/{project_id}/locations/{region}"
        f"/publishers/google/models/{model_and_action}"
    )
    url = f"https://{vertex_host}/{vertex_path}"
    if query:
        url += f"?{query}"

    return url


# ---------------------------------------------------------------------------
# Persistent HTTP client (connection pool with HTTP/2)
# ---------------------------------------------------------------------------

_http_client: httpx.AsyncClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage persistent HTTP client lifecycle."""
    global _http_client

    _http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(120.0, connect=10.0),
        follow_redirects=True,
        limits=httpx.Limits(
            max_connections=50,
            max_keepalive_connections=20,
            keepalive_expiry=120,
        ),
    )
    logger.info("HTTP client pool started (max_conn=50, keepalive=20)")
    yield
    await _http_client.aclose()
    _http_client = None
    logger.info("HTTP client pool closed")


app = FastAPI(title="Google API Proxy", docs_url=None, redoc_url=None, lifespan=lifespan)

# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def _check_auth(request: Request) -> bool:
    token = request.headers.get("x-proxy-token", "")
    return token == PROXY_SECRET


def _check_auth_value(token: str) -> bool:
    return token == PROXY_SECRET


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    pool_info = None
    if _http_client:
        pool_info = {"alive": True}
    return {
        "status": "ok",
        "grpc_streaming": True,
        "connection_pool": pool_info,
        "proxy_region": PROXY_REGION or "global",
        "render_region": os.environ.get("RENDER_REGION", "unknown"),
    }


# ---------------------------------------------------------------------------
# Diagnostics — measure latency to each Google Cloud service
# ---------------------------------------------------------------------------

@app.get("/diagnostics")
async def diagnostics(request: Request):
    """Measure proxy-to-Google latency for each allowed host."""
    if not _check_auth(request):
        return PlainTextResponse("Unauthorized", status_code=401)

    client = _http_client or httpx.AsyncClient(timeout=5.0)
    results = {}

    # Probe key hosts (global + EU regional for comparison)
    probe_hosts = [
        "speech.googleapis.com",
        "eu-speech.googleapis.com",
        "texttospeech.googleapis.com",
        "eu-texttospeech.googleapis.com",
        "generativelanguage.googleapis.com",
        "oauth2.googleapis.com",
    ]

    # Add Vertex AI EU endpoints when proxy is in EU
    if PROXY_REGION:
        region = _vertex_region()
        vertex_host = f"{region}-aiplatform.googleapis.com"
        if vertex_host not in probe_hosts:
            probe_hosts.append(vertex_host)

    for host in probe_hosts:
        t0 = time.perf_counter()
        try:
            resp = await client.head(f"https://{host}/", timeout=5.0)
            ms = (time.perf_counter() - t0) * 1000
            resolved = _resolve_host(host)
            results[host] = {
                "latency_ms": round(ms, 1),
                "status": resp.status_code,
                "resolved_to": resolved if resolved != host else None,
            }
        except Exception as e:
            ms = (time.perf_counter() - t0) * 1000
            results[host] = {"latency_ms": round(ms, 1), "error": str(e)[:120]}

    return {
        "proxy_region": PROXY_REGION or "global",
        "render_service": os.environ.get("RENDER_SERVICE_NAME", "unknown"),
        "pool_active": _http_client is not None,
        "endpoint_rewrites_active": bool(PROXY_REGION),
        "targets": results,
    }


# ---------------------------------------------------------------------------
# REST forward proxy — uses persistent connection pool
# ---------------------------------------------------------------------------

@app.api_route(
    "/proxy/{target_host}/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
)
async def forward_request(request: Request, target_host: str, path: str):
    """Forward request to target Google API host.

    URL format: /proxy/{host}/{original_path}
    Example:    /proxy/speech.googleapis.com/v2/projects/123/locations/global/recognizers/_:recognize

    When PROXY_REGION is set and target is generativelanguage.googleapis.com,
    requests are rewritten to the Vertex AI regional endpoint format.
    The client should provide X-Project-ID header with the GCP project ID.
    """
    if not _check_auth(request):
        return PlainTextResponse("Unauthorized", status_code=401)

    if target_host not in ALLOWED_HOSTS:
        return PlainTextResponse(f"Host not allowed: {target_host}", status_code=403)

    # Forward headers, strip proxy-specific and hop-by-hop
    skip = {"host", "x-proxy-token", "x-project-id", "transfer-encoding", "connection", "content-length"}
    fwd_headers = {
        k: v for k, v in request.headers.items() if k.lower() not in skip
    }

    body = await request.body()

    # --- Vertex AI rewrite for Gemini LLM requests ---
    if target_host == "generativelanguage.googleapis.com" and PROXY_REGION:
        project_id = request.headers.get("x-project-id", "")
        vertex_url = _rewrite_gemini_to_vertex(
            path, str(request.url.query) if request.url.query else "", project_id
        )
        if vertex_url:
            logger.info(
                "Vertex AI rewrite: %s -> %s (%d bytes)",
                path[:80], vertex_url[:120], len(body),
            )

            client = _http_client
            if client is None:
                return PlainTextResponse("Proxy not ready", status_code=503)

            try:
                resp = await client.request(
                    method=request.method,
                    url=vertex_url,
                    headers=fwd_headers,
                    content=body,
                )

                if resp.status_code >= 400:
                    logger.warning(
                        "Vertex AI returned %d for %s: %s",
                        resp.status_code, vertex_url[:120], resp.text[:300],
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
                logger.error("Vertex AI connect error: %s", e)
                return PlainTextResponse(f"Upstream error: {e}", status_code=502)
            except httpx.TimeoutException:
                return PlainTextResponse("Upstream timeout", status_code=504)
        else:
            if not project_id:
                logger.warning(
                    "Gemini request without X-Project-ID header; "
                    "cannot rewrite to Vertex AI, forwarding to global endpoint"
                )

    # --- Standard forwarding (non-Gemini or no PROXY_REGION) ---

    # Rewrite to regional endpoint if proxy is in EU
    resolved_host = _resolve_host(target_host)

    # Rewrite location in STT resource paths: locations/global -> locations/eu
    resolved_path = path
    if PROXY_REGION and resolved_host != target_host:
        loc = "eu" if PROXY_REGION == "eu" else PROXY_REGION
        resolved_path = path.replace("/locations/global/", f"/locations/{loc}/")

    url = f"https://{resolved_host}/{resolved_path}"
    if request.url.query:
        url += f"?{request.url.query}"

    logger.info("%s %s (%d bytes)", request.method, url[:120], len(body))

    client = _http_client
    if client is None:
        return PlainTextResponse("Proxy not ready", status_code=503)

    try:
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


# ---------------------------------------------------------------------------
# WebSocket → gRPC streaming bridge for Speech-to-Text v2
# ---------------------------------------------------------------------------
#
# Protocol:
#   1. Client opens WS to /ws/speech-stream
#   2. Client sends JSON config:
#      {
#        "proxy_token": "...",
#        "access_token": "...",          # OAuth2 access token for Google
#        "recognizer": "projects/.../locations/.../recognizers/_",
#        "config": {                     # RecognitionConfig fields
#          "language_codes": ["ru-RU"],
#          "model": "long",
#          "encoding": "LINEAR16",       # optional, default LINEAR16
#          "sample_rate_hertz": 16000,   # optional, default 16000
#          "audio_channel_count": 1      # optional, default 1
#        }
#      }
#   3. Client sends binary audio chunks (raw PCM or WAV)
#   4. Server sends JSON results as they arrive:
#      {"results": [{"alternatives": [{"transcript": "..."}], "is_final": true}]}
#   5. Client closes WS (or sends {"command": "stop"}) to end the stream
#

@app.websocket("/ws/speech-stream")
async def ws_speech_stream(ws: WebSocket):
    """Bridge WebSocket audio stream to Google STT v2 gRPC streaming."""
    await ws.accept()

    try:
        # Step 1: Receive config message
        config_raw = await asyncio.wait_for(ws.receive_text(), timeout=10)
        config = json.loads(config_raw)

        # Auth
        if not _check_auth_value(config.get("proxy_token", "")):
            await ws.send_json({"error": "Unauthorized"})
            await ws.close(code=4001, reason="Unauthorized")
            return

        access_token = config.get("access_token", "")
        recognizer = config.get("recognizer", "")
        stt_config = config.get("config", {})

        if not recognizer:
            await ws.send_json({"error": "Missing 'recognizer' in config"})
            await ws.close(code=4002, reason="Missing recognizer")
            return

        # Rewrite recognizer location if proxy is in EU
        # Client sends: projects/{id}/locations/global/recognizers/_
        # EU endpoint requires: projects/{id}/locations/eu/recognizers/_
        if PROXY_REGION == "eu" and "/locations/global/" in recognizer:
            recognizer = recognizer.replace("/locations/global/", "/locations/eu/")
        elif PROXY_REGION and PROXY_REGION != "eu" and "/locations/global/" in recognizer:
            recognizer = recognizer.replace("/locations/global/", f"/locations/{PROXY_REGION}/")

        logger.info("WS speech-stream: recognizer=%s", recognizer[:80])

        # Step 2: Open gRPC streaming channel to Google
        try:
            from google.cloud.speech_v2 import SpeechAsyncClient
            from google.cloud.speech_v2 import types as speech_types
            from google.oauth2 import credentials as oauth_credentials
        except ImportError:
            await ws.send_json({"error": "google-cloud-speech not installed on proxy"})
            await ws.close(code=4003, reason="Missing dependency")
            return

        # Create credentials from the access token passed by the client
        creds = oauth_credentials.Credentials(token=access_token)

        # Use regional gRPC endpoint if proxy is in EU
        client_kwargs = {"credentials": creds}
        resolved_speech = _resolve_host("speech.googleapis.com")
        if resolved_speech != "speech.googleapis.com":
            from google.api_core import client_options
            client_kwargs["client_options"] = client_options.ClientOptions(
                api_endpoint=resolved_speech,
            )
            logger.info("WS speech-stream: using regional endpoint %s", resolved_speech)

        client = SpeechAsyncClient(**client_kwargs)

        # Build recognition config
        encoding_map = {
            "LINEAR16": speech_types.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
            "MULAW": speech_types.ExplicitDecodingConfig.AudioEncoding.MULAW,
            "ALAW": speech_types.ExplicitDecodingConfig.AudioEncoding.ALAW,
        }
        enc_name = stt_config.get("encoding", "LINEAR16")
        encoding = encoding_map.get(enc_name, encoding_map["LINEAR16"])

        rec_config = speech_types.RecognitionConfig(
            explicit_decoding_config=speech_types.ExplicitDecodingConfig(
                encoding=encoding,
                sample_rate_hertz=stt_config.get("sample_rate_hertz", 16000),
                audio_channel_count=stt_config.get("audio_channel_count", 1),
            ),
            language_codes=stt_config.get("language_codes", ["ru-RU"]),
            model=stt_config.get("model", "long"),
        )

        streaming_config = speech_types.StreamingRecognitionConfig(
            config=rec_config,
            streaming_features=speech_types.StreamingRecognitionFeatures(
                interim_results=stt_config.get("interim_results", True),
            ),
        )

        # Audio queue: WS receiver pushes chunks, gRPC sender pops them
        audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue()

        async def request_generator():
            """Yield gRPC streaming requests: config first, then audio chunks."""
            yield speech_types.StreamingRecognizeRequest(
                recognizer=recognizer,
                streaming_config=streaming_config,
            )
            while True:
                chunk = await audio_queue.get()
                if chunk is None:
                    break
                yield speech_types.StreamingRecognizeRequest(audio=chunk)

        # Step 3: Run bidirectional streaming

        async def receive_audio():
            """Receive audio chunks from WebSocket and push to gRPC queue."""
            try:
                while True:
                    msg = await ws.receive()
                    if msg["type"] == "websocket.disconnect":
                        break
                    if "bytes" in msg and msg["bytes"]:
                        await audio_queue.put(msg["bytes"])
                    elif "text" in msg and msg["text"]:
                        try:
                            cmd = json.loads(msg["text"])
                            if cmd.get("command") == "stop":
                                break
                        except json.JSONDecodeError:
                            pass
            except WebSocketDisconnect:
                pass
            finally:
                await audio_queue.put(None)  # signal end of audio

        async def send_results():
            """Receive gRPC streaming results and forward to WebSocket."""
            try:
                response_stream = await client.streaming_recognize(
                    requests=request_generator()
                )
                async for response in response_stream:
                    results = []
                    for result in response.results:
                        alts = []
                        for alt in result.alternatives:
                            alts.append({
                                "transcript": alt.transcript,
                                "confidence": alt.confidence,
                            })
                        results.append({
                            "alternatives": alts,
                            "is_final": result.is_final,
                            "stability": result.stability,
                        })
                    if results:
                        try:
                            await ws.send_json({"results": results})
                        except Exception:
                            break
            except Exception as e:
                logger.error("gRPC streaming error: %s", e)
                try:
                    await ws.send_json({"error": str(e)})
                except Exception:
                    pass

        recv_task = asyncio.create_task(receive_audio())
        send_task = asyncio.create_task(send_results())

        # Wait for send_results to finish naturally (after all audio processed)
        try:
            await asyncio.wait_for(send_task, timeout=60)
        except asyncio.TimeoutError:
            logger.warning("WS speech-stream: gRPC response timed out")
            send_task.cancel()

        if not recv_task.done():
            recv_task.cancel()
        for task in [recv_task, send_task]:
            if not task.done():
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

        # Cleanup gRPC client
        transport = getattr(client, "_transport", None)
        if transport and hasattr(transport, "close"):
            try:
                await transport.close()
            except Exception:
                pass

        logger.info("WS speech-stream: session ended")

    except asyncio.TimeoutError:
        logger.warning("WS speech-stream: config timeout")
        await ws.close(code=4004, reason="Config timeout")
    except WebSocketDisconnect:
        logger.info("WS speech-stream: client disconnected")
    except Exception as e:
        logger.error("WS speech-stream error: %s", e)
        try:
            await ws.send_json({"error": str(e)})
            await ws.close(code=4005, reason="Internal error")
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info")
