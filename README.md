# Local voice agents on macOS with Pipecat

![screenshot](assets/debug-console-screenshot.png)

Pipecat is an open-source, vendor-neutral framework for building real-time voice (and video) AI applications.

This repository contains an example of a voice agent running with all local models on macOS. On an M-series mac, you can achieve voice-to-voice latency of <800 ms with relatively strong models.

The [server/bot.py](server/bot.py) file uses these models:

  - Silero VAD
  - smart-turn v2
  - MLX Whisper
  - Gemma3n 4B 
  - Kokoro TTS

But you can swap any of them out for other models, or completely reconfigure the pipeline. It's easy to add tool calling, MCP server integrations, use parallel pipelines to do async inference alongside the voice conversations, add custom processing steps, configure interrupt handling to work differently, etc.

The bot and web client here communicate using a low-latency, local, serverless WebRTC connection. For more information on serverless WebRTC, see the Pipecat [SmallWebRTCTransport docs](https://docs.pipecat.ai/server/services/transport/small-webrtc) and this [article](https://www.daily.co/blog/you-dont-need-a-webrtc-server-for-your-voice-agents/). You could switch over to a different Pipecat transport (for example, a WebSocket-based transport), but WebRTC is the best choice for realtime audio.

For a deep dive into voice AI, including network transport, optimizing for latency, and notes on designing tool calling and complex workflows, see the [Voice AI & Voice Agents Illustrated Guide](https://voiceaiandvoiceagents.com/).

# Architecture (deep dive)

For a full architecture overview — a high-level layer map, the detailed per-hop data flow, the Pipecat pipeline, WebRTC signaling & connection lifecycle, and the isolated-TTS protocol — see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). It ends with **Appendix A**, a glossary that defines every abbreviation, protocol, framework, and model name used in the project.

A rendered PDF of it can be built from the repo with the vendored `md2pdf.sh` wrapper via the `Makefile`:

```shell
# Crisp, *vector* diagrams at 0.5in margins -> docs/ARCHITECTURE.pdf
make

# Override the page margin:
make MARGIN=0.3in
```

This runs `sh md2pdf.sh --toc --mermaid --margin 0.5in docs/ARCHITECTURE.md`. The `--mermaid` flag draws each Mermaid diagram as a **vector PDF** (crisp at any zoom) rather than a low-res raster. The generated `docs/ARCHITECTURE.pdf` is a build artifact and is git-ignored — it's reproducible with `make`.

# Models and dependencies

Silero VAD and MLX Whisper run inside the Pipecat process. When the agent code starts, it will need to download model weights that aren't already cached, so first startup can take some time.

The LLM service in this bot uses the OpenAI-compatible chat completion HTTP API. So you will need to run a local OpenAI-compatible LLM server. 

The easiest way to run a local LLM server on macOS is [Ollama](https://ollama.com/). The bot in this repo points at Ollama by default (see `server/bot.py`), which serves an OpenAI-compatible API on `http://127.0.0.1:11434/v1`. Pull the model the bot expects and start Ollama:

```shell
ollama pull gemma3n:e4b

# Start the Ollama server if it isn't already running (default port 11434)
ollama serve

# Confirm the model is available
ollama list
```

> Other OpenAI-compatible local servers work too — for example [LM Studio](https://lmstudio.ai/). Whatever you use, make sure its `/v1` endpoint matches the `base_url` in `server/bot.py`. LM Studio's default is `http://127.0.0.1:1234/v1`; Ollama's is `http://127.0.0.1:11434/v1`.

# Run the voice agent

The core voice agent code lives in a single file: [server/bot.py](server/bot.py). There's one custom service here that's not included in Pipecat core: we implemented a local MLX-Audio frame processor on top of the excellent [mlx-audio library](https://github.com/Blaizzy/mlx-audio). It's `TTSMLXIsolated`, which runs the actual TTS synthesis in an **isolated subprocess** (`kokoro_worker.py` for Kokoro, `marvis_worker.py` for Marvis, chosen by model name; a simple JSON-over-stdio protocol) to sidestep MLX/Metal threading conflicts on Apple Silicon. See the *Isolated TTS (avoiding Metal threading conflicts)* section of the architecture doc for the full protocol.

Note that the first time you start the bot it will take some time to initialize the three models. It can be 30 seconds or more before the bot is fully ready to go. Subsequent startups will be much faster.

A few of those models are loaded inside the Pipecat process *before* the first conversation, so pre-warming them makes the first startup much faster. Two of them are the speech models, and `server/preflight.py` pre-downloads and verifies both by mimicking exactly what `bot.py` does:

   * **Silero VAD** — packaged inside the `pipecat` wheel itself (`pipecat/audio/vad/data/silero_vad.onnx`) and loaded via `importlib.resources`; `preflight.py` just confirms it's present and that ONNXRuntime can open it.
   * **MLX Whisper** (`WhisperSTTServiceMLX`, `MLXModel.LARGE_V3_TURBO_Q4`) — `preflight.py` runs `mlx_whisper.transcribe` on a 3-second silent buffer, which triggers a `snapshot_download` from `mlx-community/whisper-large-v3-turbo-q4` and warms the model into MLX memory.

```shell
cd server/

# Pre-download & verify the speech models (downloads if missing, no-op if cached)
uv run python preflight.py

# Verify only, no network
uv run python preflight.py --offline
```

For the TTS model, one more easy step: a quick generation from the command line before the first run so you're not waiting on a large HuggingFace download:

The `mlx-audio` TTS entry point is the `mlx_audio.tts.generate` script. It doesn't take an `--output` flag anymore: use `--play` to speak the result immediately (and avoid writing a file), or `--file_prefix` to save it to disk.

```shell
mlx_audio.tts.generate --model "mlx-community/Kokoro-82M-bf16" \
    --text "Once upon a midnight dreary, or something like that! It's been a long time since I left high-school" \
    --play
# or
mlx_audio.tts.generate --model "Marvis-AI/marvis-tts-250m-v0.1" \
    --text "Hello, I'm Pipecat!" \
    --play
```

If you're using uv

```
uv run bot.py
```

> Note: `uv` does not install `pip` into the project virtual environment by default. If you need `pip` inside the `uv`-managed environment (for example to run a one-off `pip install` of an extra package), install it with:
>
> ```
> uv pip install pip
> ```
>
> You can then use it inside the venv, e.g. `uv run pip install some-package`. For most of this project you won't need `pip` at all — `uv run bot.py` is preferred.

If you're using pip

```
python3.12 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

python bot.py
```

After you run the first time and have all the models cached, you can set the HF_HUB_OFFLINE environment variable to prevent the Hugging Face libraries from going to the network and checking for model updates. This makes the initial bot startup and first conversation turn a lot faster.

```
HF_HUB_OFFLINE=1 uv run bot.py
```

# Configuration

The bot is fully configurable — **no code edits** to swap models or tune knobs. Every option is a field in the `Config` dataclass in [`server/config.py`](server/config.py) with a built-in default, so a zero-config run is identical to the original hard-coded behaviour. Values resolve in order **command-line flag > env / `.env` > default**, and the bot logs the *resolved* config on startup so you can confirm which value won.

Copy `server/env.example` → `server/.env` and uncomment what you want, or flip a single option on the command line:

```shell
# swap to a different LLM (a tag from `ollama list`)
uv run bot.py --llm-model qwen3:8b

# a different voice + a bigger STT model, and turn metrics off
uv run bot.py --tts-voice am_michael --stt-model mlx-community/whisper-large-v3-turbo --no-metrics

# run a custom system prompt from a file
uv run bot.py --system-prompt-file ./prompts/assistant.md

# list every option
uv run bot.py --help
```

Key variables (the full list + descriptions live in [`server/config.py`](server/config.py); `bot.py --help` prints them too):

| Variable (env) | Flag | Default | Purpose |
| --- | --- | --- | --- |
| `LLM_MODEL` | `--llm-model` | `gemma3n:e4b` | Ollama / OpenAI-compatible model tag |
| `LLM_BASE_URL` | `--llm-base-url` | `http://127.0.0.1:11434/v1` | LLM endpoint |
| `LLM_MAX_TOKENS` | `--llm-max-tokens` | `4096` | Max LLM output tokens |
| `STT_MODEL` | `--stt-model` | `.../whisper-large-v3-turbo-q4` | MLX Whisper id / `MLXModel` name |
| `TTS_MODEL` | `--tts-model` | `.../Kokoro-82M-bf16` | TTS engine (Kokoro/Marvis, auto-picked by name) |
| `TTS_VOICE` | `--tts-voice` | `af_heart` | Kokoro voice id |
| `TTS_SAMPLE_RATE` | `--tts-sample-rate` | `24000` | TTS output sample rate |
| `VAD_STOP_SECS` | `--vad-stop-secs` | `0.2` | Silence (s) that ends a turn |
| `SMART_TURN_MODEL` | `--smart-turn-model` | `""` | End-of-turn model path (`""` = download) |
| `SYSTEM_PROMPT(_FILE)` | `--system-prompt[-file]` | built-in | Chat prompt |
| `AGGREGATION_TIMEOUT` | `--aggregation-timeout` | `0.05` | User transcript coalescence (Whisper is non-streaming) |
| `ICE_SERVERS` | `--ice-servers` | google STUN | Comma/separated STUN/TURN URLs |
| `ENABLE(_METRICS_)_[USAGE]_` | `--no-metrics` / `--no-usage-metrics` | `true` | Pipeline / usage metrics |
| `BOT_HOST` / `BOT_PORT` | `--host` / `--port` | `localhost` / `7860` | HTTP bind |

# Start the web client

The web client is a React app. You can connect to your local macOS agent using any client that can negotiate a serverless WebRTC connection. The client in this repo is based on [voice-ui-kit](https://github.com/pipecat-ai/voice-ui-kit) and just uses that library's standard debug console template.

```shell
cd client/

npm i

npm run dev

# Navigate to URL shown in terminal in your web browser
```