"""
Central configuration for the local voice bot.

Resolution order for every value:  command-line args  >  env / .env  >  default.

The built-in defaults reproduce the original hard-coded behaviour, so the bot
runs with *zero* configuration. Set any variable below (or the matching `--flag`)
to swap models or behaviour without editing code. All values are read from the
environment after `python-dotenv` has loaded `.env` (see `bot.py`).

Environment variables
---------------------
    BOT_HOST, BOT_PORT                     HTTP server bind (default localhost:7860)
    LLM_MODEL, LLM_BASE_URL, LLM_API_KEY, LLM_MAX_TOKENS
                                           OpenAI-compatible LLM (default Ollama gemma3n:e4b)
    STT_MODEL                              MLX Whisper repo id or MLXModel name
    TTS_MODEL, TTS_VOICE, TTS_SAMPLE_RATE  TTS engine (default Kokoro af_heart, 24 kHz)
    VAD_STOP_SECS                          voice-activity "silence means done" in seconds
    SMART_TURN_MODEL                       end-of-turn model path ("" = download from HF)
    SYSTEM_PROMPT, SYSTEM_PROMPT_FILE      chat prompt (inline text or a file's contents)
    AGGREGATION_TIMEOUT                    user transcript coalescence timeout (seconds)
    ICE_SERVERS                            comma-separated STUN/TURN server URLs
    ENABLE_METRICS, ENABLE_USAGE_METRICS   pipeline metrics on/off
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from typing import List, Optional

# The chat prompt. Kept as the original text so a zero-config run is unchanged.
DEFAULT_SYSTEM_PROMPT = """\
"You are Pipecat, a friendly, helpful chatbot.

Your input is text transcribed in realtime from the user's voice. There may be transcription errors. Adjust your responses automatically to account for these errors.

Your output will be converted to audio so don't include special characters in your answers and do not use any markdown or special formatting.

Respond to what the user said in a creative and helpful way. Keep your responses brief unless you are explicitly asked for long or detailed responses. Normally you should use one or two sentences at most. Keep each sentence short. Prefer simple sentences. Try not to use long sentences with multiple comma clauses.

Start the conversation by saying, "Hello, I'm Pipecat!" Then stop and wait for the user.
"""

DEFAULT_ICE_SERVERS = ["stun:stun.l.google.com:19302"]


def _env_str(name: str, default: str) -> str:
    v = os.environ.get(name)
    return default if v is None or v == "" else v


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name]) if os.environ.get(name) not in (None, "") else default
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name]) if os.environ.get(name) not in (None, "") else default
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    return v.strip().lower() in {"1", "true", "yes", "on", "y"}


@dataclass
class Config:
    # --- HTTP server ---
    host: str = "localhost"
    port: int = 7860

    # --- LLM (OpenAI-compatible; Ollama by default) ---
    llm_model: str = "gemma3n:e4b"          # any tag shown by `ollama list`
    llm_base_url: str = "http://127.0.0.1:11434/v1"
    llm_api_key: str = "dummyKey"           # Ollama ignores this; required by the constructor
    llm_max_tokens: int = 4096

    # --- STT (MLX Whisper) ---
    stt_model: str = "mlx-community/whisper-large-v3-turbo-q4"   # or e.g. "mlx-community/whisper-large-v3-turbo"

    # --- TTS (isolated subprocess; Kokoro or Marvis, auto-selected by model name) ---
    tts_model: str = "mlx-community/Kokoro-82M-bf16"
    tts_voice: str = "af_heart"            # Kokoro voice id; ignored by the Marvis worker
    tts_sample_rate: int = 24000

    # --- VAD / turn detection ---
    vad_stop_secs: float = 0.2            # seconds of silence that end a turn
    smart_turn_model: str = ""            # "" => downloaded from Hugging Face

    # --- Conversation prompt ---
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    system_prompt_file: Optional[str] = None   # if set, its contents replace system_prompt

    # --- Misc ---
    aggregation_timeout: float = 0.05     # user transcript coalescence timeout (Whisper is non-streaming)
    ice_servers: List[str] = None  # type: ignore[assignment]  # populated in __post_init__
    enable_metrics: bool = True
    enable_usage_metrics: bool = True

    def __post_init__(self) -> None:
        if self.ice_servers is None:
            self.ice_servers = list(DEFAULT_ICE_SERVERS)
        # A prompt file takes precedence over inline system_prompt.
        if self.system_prompt_file:
            try:
                with open(self.system_prompt_file, encoding="utf-8") as f:
                    self.system_prompt = f.read()
            except OSError as e:
                raise SystemExit(f"could not read SYSTEM_PROMPT_FILE {self.system_prompt_file!r}: {e}") from e

    # --- Construction ---
    @classmethod
    def from_env(cls) -> "Config":
        """Build a Config from the process environment, applying defaults."""
        c = cls()
        c.host = _env_str("BOT_HOST", c.host)
        c.port = _env_int("BOT_PORT", c.port)

        c.llm_model = _env_str("LLM_MODEL", c.llm_model)
        c.llm_base_url = _env_str("LLM_BASE_URL", c.llm_base_url)
        c.llm_api_key = _env_str("LLM_API_KEY", c.llm_api_key)
        c.llm_max_tokens = _env_int("LLM_MAX_TOKENS", c.llm_max_tokens)

        c.stt_model = _env_str("STT_MODEL", c.stt_model)

        c.tts_model = _env_str("TTS_MODEL", c.tts_model)
        c.tts_voice = _env_str("TTS_VOICE", c.tts_voice)
        c.tts_sample_rate = _env_int("TTS_SAMPLE_RATE", c.tts_sample_rate)

        c.vad_stop_secs = _env_float("VAD_STOP_SECS", c.vad_stop_secs)
        c.smart_turn_model = _env_str("SMART_TURN_MODEL", c.smart_turn_model)

        raw_prompt = os.environ.get("SYSTEM_PROMPT")
        if raw_prompt is not None:
            c.system_prompt = raw_prompt
        c.system_prompt_file = os.environ.get("SYSTEM_PROMPT_FILE") or None

        c.aggregation_timeout = _env_float("AGGREGATION_TIMEOUT", c.aggregation_timeout)

        raw_ice = os.environ.get("ICE_SERVERS")
        if raw_ice:
            c.ice_servers = [s.strip() for s in raw_ice.replace(";", ",").split(",") if s.strip()]

        c.enable_metrics = _env_bool("ENABLE_METRICS", c.enable_metrics)
        c.enable_usage_metrics = _env_bool("ENABLE_USAGE_METRICS", c.enable_usage_metrics)

        return c

    def describe(self) -> str:
        """One-line-per-field summary, for logging at startup."""
        lines = [
            f"  LLM    : {self.llm_model} @ {self.llm_base_url} (max_tokens={self.llm_max_tokens})",
            f"  STT    : {self.stt_model}",
            f"  TTS    : {self.tts_model} (voice={self.tts_voice!r}, {self.tts_sample_rate} Hz)",
            f"  VAD    : Silero stop_secs={self.vad_stop_secs}",
            f"  turn   : LocalSmartTurn (model_path={self.smart_turn_model or '<download>'})",
            f"  server : http://{self.host}:{self.port}/api/offer",
            f"  ICE    : {', '.join(self.ice_servers)}",
            f"  metrics: enabled={self.enable_metrics}, usage={self.enable_usage_metrics}",
            f"  prompt : {len(self.system_prompt)} chars",
        ]
        return "\n".join(lines)
