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
PROXY_REGION = os.environ.get("PROXY_REGION", "")

EU_ENDPOINT_REWRITES = {
    "speech.googleapis.com": "eu-speech.googleapis.com",
    "texttospeech.googleapis.com": "eu-texttospeech.googleapis.com",
}

REGIONAL_ENDPOINT_REWRITES = {
    "speech.googleapis.com": "{region}-speech.googleapis.com",
    "texttospeech.googleapis.com": "{region}-texttospeech.googleapis.com",
}

_VERTEX_EU_DEFAULT_REGION = "europe-west1"
_GEMINI_MODEL_RE = re.compile(r"v1beta/models/([^?]+)")


def _resolve_host(target_host: str) -> str:
    if not PROXY_REGION or target_host not in EU_ENDPOINT_REWRITES:
        return target_host
    if PROXY_REGION == "eu":
        return EU_ENDPOINT_REWRITES.get(target_host, target_host)
    template = REGIONAL_ENDPOINT_REWRITES.get(target_host)
    if template:
        return template.format(region=PROXY_REGION)
    return target_host


def _vertex_region() -> str:
    if PROXY_REGION == "eu":
        return _VERTEX_EU_DEFAULT_REGION
    return PROXY_REGION if PROXY_REGION else ""


def _rewrite_gemini_to_vertex(path: str, project_id: str, region: str) -> tuple[str, str] | None:
    m = _GEMINI_MODEL_RE.search(path)
    if not m:
        return None
    model_and_action = m.group(1)
    host = f"{region}-aiplatform.googleapis.com"
    new_path = f"v1/projects/{project_id}/locations/{region}/publishers/google/models/{model_and_action}"
    return host, new_path


# ---------------------------------------------------------------------------
# Persistent HTTP client (connection pool)
# ---------------------------------------------------------------------------

_http_client: httpx.AsyncClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
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
# Diagnostics
# ---------------------------------------------------------------------------

@app.get("/diagnostics")
async def diagnostics(request: Request):
    if not _check_auth(request):
        return PlainTextResponse("Unauthorized", status_code=401)

    client = _http_client or httpx.AsyncClient(timeout=5.0)
    results = {}

    probe_hosts = [
        "speech.googleapis.com",
        "eu-speech.googleapis.com",
        "texttospeech.googleapis.com",
        "eu-texttospeech.googleapis.com",
        "generativelanguage.googleapis.com",
        "oauth2.googleapis.com",
    ]
    region = _vertex_region()
    if region:
        probe_hosts.append(f"{region}-aiplatform.googleapis.com")

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
    if not _check_auth(request):
        return PlainTextResponse("Unauthorized", status_code=401)

    if target_host not in ALLOWED_HOSTS:
        return PlainTextResponse(f"Host not allowed: {target_host}", status_code=403)

    skip = {"host", "x-proxy-token", "x-project-id", "transfer-encoding", "connection", "content-length"}
    fwd_headers = {
        k: v for k, v in request.headers.items() if k.lower() not in skip
    }
    body = await request.body()

    client = _http_client
    if client is None:
        return PlainTextResponse("Proxy not ready", status_code=503)

    # --- Vertex AI rewrite for Gemini ---
    if target_host == "generativelanguage.googleapis.com" and PROXY_REGION:
        project_id = request.headers.get("x-project-id", "")
        region = _vertex_region()
        if project_id and region:
            result = _rewrite_gemini_to_vertex(path, project_id, region)
            if result:
                vertex_host, vertex_path = result
                vertex_url = f"https://{vertex_host}/{vertex_path}"
                if request.url.query:
                    vertex_url += f"?{request.url.query}"
                logger.info("Vertex AI rewrite: %s (%d bytes)", vertex_url[:120], len(body))
                try:
                    resp = await client.request(
                        method=request.method, url=vertex_url,
                        headers=fwd_headers, content=body,
                    )
                    if resp.status_code >= 400:
                        logger.warning("Vertex AI %d: %s", resp.status_code, resp.text[:300])
                    resp_headers = {
                        k: v for k, v in resp.headers.items()
                        if k.lower() not in ("transfer-encoding", "connection", "content-encoding")
                    }
                    return Response(content=resp.content, status_code=resp.status_code, headers=resp_headers)
                except httpx.ConnectError as e:
                    return PlainTextResponse(f"Upstream error: {e}", status_code=502)
                except httpx.TimeoutException:
                    return PlainTextResponse("Upstream timeout", status_code=504)

    # --- Standard forwarding ---
    resolved_host = _resolve_host(target_host)
    resolved_path = path
    if PROXY_REGION and resolved_host != target_host:
        loc = "eu" if PROXY_REGION == "eu" else PROXY_REGION
        resolved_path = path.replace("/locations/global/", f"/locations/{loc}/")

    url = f"https://{resolved_host}/{resolved_path}"
    if request.url.query:
        url += f"?{request.url.query}"

    logger.info("%s %s (%d bytes)", request.method, url[:120], len(body))

    try:
        resp = await client.request(
            method=request.method, url=url, headers=fwd_headers, content=body,
        )
        resp_headers = {
            k: v for k, v in resp.headers.items()
            if k.lower() not in ("transfer-encoding", "connection", "content-encoding")
        }
        return Response(content=resp.content, status_code=resp.status_code, headers=resp_headers)
    except httpx.ConnectError as e:
        return PlainTextResponse(f"Upstream error: {e}", status_code=502)
    except httpx.TimeoutException:
        return PlainTextResponse("Upstream timeout", status_code=504)


# ---------------------------------------------------------------------------
# WebSocket → gRPC streaming bridge for Speech-to-Text v2
# ---------------------------------------------------------------------------
#
# Session-persistent protocol (WS stays open for entire voice session):
#
#   1. Client opens WS to /ws/speech-stream
#   2. Client sends auth JSON: {"proxy_token": "...", "access_token": "...",
#      "recognizer": "...", "config": {...}}
#   3. Server responds: {"status": "ready"}
#   4. For each speech turn:
#      a. Client sends audio chunks (binary)
#      b. Server sends partial results: {"results": [...]}
#      c. Client sends {"command": "end_turn"} when speech ends
#      d. Server sends final results + {"status": "turn_complete"}
#   5. Client sends {"command": "close"} or closes WS to end session
#

@app.websocket("/ws/speech-stream")
async def ws_speech_stream(ws: WebSocket):
    """Session-persistent WS bridge to Google STT v2 gRPC streaming."""
    await ws.accept()

    try:
        # Step 1: Auth + config (once per session)
        config_raw = await asyncio.wait_for(ws.receive_text(), timeout=10)
        config = json.loads(config_raw)

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

        # Rewrite recognizer location for EU
        if PROXY_REGION == "eu" and "/locations/global/" in recognizer:
            recognizer = recognizer.replace("/locations/global/", "/locations/eu/")
        elif PROXY_REGION and PROXY_REGION != "eu" and "/locations/global/" in recognizer:
            recognizer = recognizer.replace("/locations/global/", f"/locations/{PROXY_REGION}/")

        # Import gRPC deps
        try:
            from google.cloud.speech_v2 import SpeechAsyncClient
            from google.cloud.speech_v2 import types as speech_types
            from google.oauth2 import credentials as oauth_credentials
        except ImportError:
            await ws.send_json({"error": "google-cloud-speech not installed on proxy"})
            await ws.close(code=4003, reason="Missing dependency")
            return

        # Build recognition config (reused across turns)
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

        # Create gRPC client (reused across turns)
        creds = oauth_credentials.Credentials(token=access_token)
        client_kwargs = {"credentials": creds}
        resolved_speech = _resolve_host("speech.googleapis.com")
        if resolved_speech != "speech.googleapis.com":
            from google.api_core import client_options
            client_kwargs["client_options"] = client_options.ClientOptions(
                api_endpoint=resolved_speech,
            )

        grpc_client = SpeechAsyncClient(**client_kwargs)

        await ws.send_json({"status": "ready"})
        logger.info("WS speech-stream: session started, recognizer=%s", recognizer[:80])

        # Step 2: Turn loop — each turn opens a new gRPC stream
        while True:
            # Wait for audio or commands
            audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
            turn_active = False
            turn_results: list[dict] = []

            async def request_generator():
                yield speech_types.StreamingRecognizeRequest(
                    recognizer=recognizer,
                    streaming_config=streaming_config,
                )
                while True:
                    chunk = await audio_queue.get()
                    if chunk is None:
                        break
                    yield speech_types.StreamingRecognizeRequest(audio=chunk)

            async def send_results(response_stream):
                try:
                    async for response in response_stream:
                        results = []
                        for result in response.results:
                            alts = [{"transcript": a.transcript, "confidence": a.confidence}
                                    for a in result.alternatives]
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
                    logger.error("gRPC turn error: %s", e)
                    try:
                        await ws.send_json({"error": str(e)})
                    except Exception:
                        pass

            send_task = None

            try:
                while True:
                    msg = await ws.receive()

                    if msg["type"] == "websocket.disconnect":
                        if turn_active:
                            await audio_queue.put(None)
                        raise WebSocketDisconnect()

                    if "bytes" in msg and msg["bytes"]:
                        if not turn_active:
                            # First audio chunk starts a new turn
                            turn_active = True
                            response_stream = await grpc_client.streaming_recognize(
                                requests=request_generator()
                            )
                            send_task = asyncio.create_task(send_results(response_stream))
                            logger.debug("WS speech-stream: turn started")

                        await audio_queue.put(msg["bytes"])

                    elif "text" in msg and msg["text"]:
                        try:
                            cmd = json.loads(msg["text"])
                        except json.JSONDecodeError:
                            continue

                        command = cmd.get("command", "")

                        if command in ("end_turn", "stop"):
                            if turn_active:
                                await audio_queue.put(None)  # signal gRPC stream end
                                if send_task:
                                    try:
                                        await asyncio.wait_for(send_task, timeout=10)
                                    except asyncio.TimeoutError:
                                        logger.warning("gRPC turn response timed out")
                                        send_task.cancel()
                                turn_active = False
                                send_task = None
                                await ws.send_json({"status": "turn_complete"})
                                logger.debug("WS speech-stream: turn complete")

                            if command == "stop":
                                # Legacy compat: "stop" ends the session
                                break

                        elif command == "close":
                            if turn_active:
                                await audio_queue.put(None)
                            break

            except WebSocketDisconnect:
                pass

            break  # exit the outer turn loop

        # Cleanup gRPC client
        transport = getattr(grpc_client, "_transport", None)
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
