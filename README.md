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

The core voice agent code lives in a single file: [server/bot.py](server/bot.py). There's one custom service here that's not included in Pipecat core: we implemented a local MLX-Audio frame processor on top of the excellent [mlx-audio library](https://github.com/Blaizzy/mlx-audio).

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

# Start the web client

The web client is a React app. You can connect to your local macOS agent using any client that can negotiate a serverless WebRTC connection. The client in this repo is based on [voice-ui-kit](https://github.com/pipecat-ai/voice-ui-kit) and just uses that library's standard debug console template.

```shell
cd client/

npm i

npm run dev

# Navigate to URL shown in terminal in your web browser
```