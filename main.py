"""
Authenticated HTTP forward proxy for Google Cloud APIs.

Deployed on render.com (US/EU) so Russian-hosted servers can reach
Google APIs through a non-restricted IP.

Supports:
  1. REST forwarding — /proxy/{host}/{path}  (all Google REST APIs)
  2. WebSocket gRPC bridge — /ws/speech-stream  (STT v2 streaming recognition)

Pipeline sets env vars:
  GOOGLE_PROXY_URL=https://your-proxy.onrender.com
  GOOGLE_PROXY_TOKEN=your-secret
"""

import asyncio
import json
import logging
import os

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
    "speech.googleapis.com",
    "texttospeech.googleapis.com",
    "generativelanguage.googleapis.com",
    "oauth2.googleapis.com",
    "www.googleapis.com",
}

app = FastAPI(title="Google API Proxy", docs_url=None, redoc_url=None)

# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def _check_auth(request: Request) -> bool:
    token = request.headers.get("x-proxy-token", "")
    return token == PROXY_SECRET


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "grpc_streaming": True}


# ---------------------------------------------------------------------------
# REST forward proxy (existing)
# ---------------------------------------------------------------------------

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
        if config.get("proxy_token") != PROXY_SECRET:
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
        client = SpeechAsyncClient(credentials=creds)

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
        )

        # Audio queue: WS receiver pushes chunks, gRPC sender pops them
        audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue()

        async def request_generator():
            """Yield gRPC streaming requests: config first, then audio chunks."""
            # First message: streaming config
            yield speech_types.StreamingRecognizeRequest(
                recognizer=recognizer,
                streaming_config=streaming_config,
            )
            # Subsequent messages: audio data from the queue
            while True:
                chunk = await audio_queue.get()
                if chunk is None:  # sentinel: end of stream
                    break
                yield speech_types.StreamingRecognizeRequest(audio=chunk)

        # Step 3: Run bidirectional streaming
        # - Task A: receive audio from WS, push to queue
        # - Task B: receive results from gRPC, send to WS

        stream_ended = asyncio.Event()

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
                stream_ended.set()

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
                            break  # WS closed
            except Exception as e:
                logger.error("gRPC streaming error: %s", e)
                try:
                    await ws.send_json({"error": str(e)})
                except Exception:
                    pass

        # Run both tasks concurrently
        recv_task = asyncio.create_task(receive_audio())
        send_task = asyncio.create_task(send_results())

        # Wait for send_results to finish — it naturally ends when gRPC
        # stream closes (after request_generator yields all audio + sentinel).
        # recv_task feeds the queue; send_task drains gRPC responses.
        # If recv_task fails (WS disconnect), the sentinel is still pushed
        # via the finally block, so send_task will eventually finish.
        try:
            await asyncio.wait_for(send_task, timeout=60)
        except asyncio.TimeoutError:
            logger.warning("WS speech-stream: gRPC response timed out")
            send_task.cancel()

        # Cleanup recv_task if still running
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
