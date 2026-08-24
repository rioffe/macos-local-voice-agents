import argparse
import asyncio
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict

# Add local pipecat to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pipecat", "src"))

import uvicorn
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI
from loguru import logger

from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v2 import LocalSmartTurnAnalyzerV2
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.whisper.stt import WhisperSTTServiceMLX
from pipecat.transports.base_transport import TransportParams
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.transports.network.small_webrtc import SmallWebRTCTransport
from pipecat.transports.network.webrtc_connection import IceServer, SmallWebRTCConnection
from pipecat.processors.aggregators.llm_response import LLMUserAggregatorParams

from tts_mlx_isolated import TTSMLXIsolated
from config import Config

load_dotenv(override=True)

app = FastAPI()

pcs_map: Dict[str, SmallWebRTCConnection] = {}

# Resolved from env / .env at import; CLI flags refine it in main() below.
cfg: Config = Config.from_env()


def _ice_servers_from(cfg: Config) -> list:
    """Build IceServer objects from the configured server URLs."""
    return [IceServer(urls=url) for url in cfg.ice_servers]


async def run_bot(webrtc_connection: "SmallWebRTCConnection", cfg: Config):
    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=cfg.vad_stop_secs)),
            turn_analyzer=LocalSmartTurnAnalyzerV2(
                smart_turn_model_path=cfg.smart_turn_model,  # "" => download from HuggingFace
                params=SmartTurnParams(),
            ),
        ),
    )

    stt = WhisperSTTServiceMLX(model=cfg.stt_model)  # accepts an MLX repo id or MLXModel name

    # The Kokoro/Marvis worker is selected inside TTSMLXIsolated by model name
    # (prefix "Marvis-AI" -> marvis_worker.py, otherwise kokoro_worker.py).
    tts = TTSMLXIsolated(model=cfg.tts_model, voice=cfg.tts_voice or None, sample_rate=cfg.tts_sample_rate)

    llm = OpenAILLMService(
        api_key=cfg.llm_api_key,
        model=cfg.llm_model,
        base_url=cfg.llm_base_url,
        max_tokens=cfg.llm_max_tokens,
    )

    context = OpenAILLMContext(
        [
            {
                "role": "user",
                "content": cfg.system_prompt,
            }
        ],
    )
    context_aggregator = llm.create_context_aggregator(
        context,
        # Whisper (MLX) isn't streaming, so it delivers the full text all at
        # once, after the UserStoppedSpeaking frame. Keep aggregation_timeout
        # de minimis since we don't expect any transcript aggregation.
        user_params=LLMUserAggregatorParams(aggregation_timeout=cfg.aggregation_timeout),
    )

    #
    # RTVI events for Pipecat client UI
    #
    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            rtvi,
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=cfg.enable_metrics,
            enable_usage_metrics=cfg.enable_usage_metrics,
        ),
        observers=[RTVIObserver(rtvi)],
    )

    @rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        await rtvi.set_bot_ready()
        # Kick off the conversation
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant):
        logger.info(f"Participant joined: {participant}")
        await transport.capture_participant_transcription(participant["id"])

    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        logger.info(f"Participant left: {participant}")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)


@app.post("/api/offer")
async def offer(request: dict, background_tasks: BackgroundTasks):
    pc_id = request.get("pc_id")

    if pc_id and pc_id in pcs_map:
        pipecat_connection = pcs_map[pc_id]
        logger.info(f"Reusing existing connection for pc_id: {pc_id}")
        await pipecat_connection.renegotiate(
            sdp=request["sdp"],
            type=request["type"],
            restart_pc=request.get("restart_pc", False),
        )
    else:
        pipecat_connection = SmallWebRTCConnection(_ice_servers_from(cfg))
        await pipecat_connection.initialize(sdp=request["sdp"], type=request["type"])

        @pipecat_connection.event_handler("closed")
        async def handle_disconnected(webrtc_connection: SmallWebRTCConnection):
            logger.info(f"Discarding peer connection for pc_id: {webrtc_connection.pc_id}")
            pcs_map.pop(webrtc_connection.pc_id, None)

        # Run example function with SmallWebRTC transport arguments.
        background_tasks.add_task(run_bot, pipecat_connection, cfg)

    answer = pipecat_connection.get_answer()
    # Updating the peer connection inside the map
    pcs_map[answer["pc_id"]] = pipecat_connection

    return answer


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield  # Run app
    coros = [pc.disconnect() for pc in pcs_map.values()]
    await asyncio.gather(*coros)
    pcs_map.clear()


def _apply_cli(cfg: Config, args: argparse.Namespace) -> Config:
    """Layer command-line overrides on top of the env-resolved config.

    Only flags the user actually passed are in `args` (they default to
    argparse.SUPPRESS), so unset flags fall through to env / default.
    """
    for name in (
        "host",
        "port",
        "llm_model",
        "llm_base_url",
        "llm_api_key",
        "llm_max_tokens",
        "stt_model",
        "tts_model",
        "tts_voice",
        "tts_sample_rate",
        "vad_stop_secs",
        "smart_turn_model",
        "aggregation_timeout",
    ):
        val = getattr(args, name, None)
        if val is not None:
            setattr(cfg, name, val)

    if getattr(args, "system_prompt", None):
        cfg.system_prompt = args.system_prompt
    if getattr(args, "system_prompt_file", None):
        cfg.system_prompt = Path(args.system_prompt_file).read_text()
    if getattr(args, "ice_servers", None):
        cfg.ice_servers = [
            s.strip() for s in args.ice_servers.replace(";", ",").split(",") if s.strip()
        ]
    if getattr(args, "no_metrics", False):
        cfg.enable_metrics = False
    if getattr(args, "no_usage_metrics", False):
        cfg.enable_usage_metrics = False

    return cfg


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Pipecat Bot Runner (configurable)")
    p.add_argument("--host", help=f"HTTP host (env BOT_HOST, default {cfg.host})")
    p.add_argument("--port", type=int, help=f"HTTP port (env BOT_PORT, default {cfg.port})")
    p.add_argument("--llm-model", help=f"Ollama/LLM model tag (env LLM_MODEL, default {cfg.llm_model!r})")
    p.add_argument("--llm-base-url", help="LLM OpenAI-compatible base URL (env LLM_BASE_URL)")
    p.add_argument("--llm-api-key", help="LLM API key (env LLM_API_KEY)")
    p.add_argument("--llm-max-tokens", type=int, help="LLM max output tokens (env LLM_MAX_TOKENS)")
    p.add_argument("--stt-model", help="MLX Whisper repo id / MLXModel name (env STT_MODEL)")
    p.add_argument("--tts-model", help="TTS model (env TTS_MODEL)")
    p.add_argument("--tts-voice", help="TTS voice (env TTS_VOICE)")
    p.add_argument("--tts-sample-rate", type=int, help="TTS sample rate (env TTS_SAMPLE_RATE)")
    p.add_argument("--vad-stop-secs", type=float, help="VAD stop_secs (env VAD_STOP_SECS)")
    p.add_argument("--smart-turn-model", help="End-of-turn model path (env SMART_TURN_MODEL)")
    p.add_argument("--aggregation-timeout", type=float, help="User transcript coalescence timeout (env AGGREGATION_TIMEOUT)")
    p.add_argument("--system-prompt", help="Inline system prompt text (overrides SYSTEM_PROMPT)")
    p.add_argument("--system-prompt-file", help="Read the system prompt from this file")
    p.add_argument("--ice-servers", help="Comma-/;-separated STUN/TURN URLs (env ICE_SERVERS)")
    p.add_argument("--no-metrics", action="store_true", help="Disable pipeline metrics")
    p.add_argument("--no-usage-metrics", action="store_true", help="Disable usage metrics")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    cfg = _apply_cli(cfg, args)  # precedence: CLI > env > default

    logger.info("Resolved bot configuration:\n" + cfg.describe())
    uvicorn.run(app, host=cfg.host, port=cfg.port)
