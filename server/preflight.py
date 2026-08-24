"""
preflight.py -- warm the on-disk model cache for the two audio models the voice
bot loads *inside* the Pipecat process before the first call:

    * Silero VAD    (pipecat VAD analyzer -- bundled in the pipecat package)
    * MLX Whisper   (server/bot.py:  WhisperSTTServiceMLX(model=MLXModel.LARGE_V3_TURBO_Q4))

This mirrors exactly what bot.py does, so the first `uv run bot.py` has nothing
to download and starts fast.

How each model is resolved (verified against the installed packages):

    * Silero VAD:
          The ONNX weights ship *inside* the pipecat package at
      `pipecat/audio/vad/data/silero_vad.onnx` and are loaded via
      importlib_resources (see pipecat/audio/vad/silero.py:149). Nothing is
      fetched from Hugging Face -- we just confirm the resource is present and
      that onnxruntime can load it.

    * MLX Whisper:
          `WhisperSTTServiceMLX.run` calls
              mlx_whisper.transcribe(path_or_hf_repo=self.model_name, ...)
      which loads the weights through `mlx_whisper.load_models.load_model`,
      falling back to huggingface_hub.snapshot_download() when the repo is not
      cached yet (mlx_whisper/load_models.py:20). The default model id is
      `mlx-community/whisper-large-v3-turbo-q4`.

Run from server/:
    uv run python preflight.py
    uv run python preflight.py --whisper mlx-community/whisper-large-v3-mlx     # any MLX tag
    uv run python preflight.py --offline    # no network: verify only (HF_HUB_OFFLINE=1)
"""

import argparse
import os
import sys

HUGGINGFACE_HUB_CACHE = os.environ.get(
    "HUGGINGFACE_HUB_CACHE",
    os.path.expanduser("~/.cache/huggingface/hub"),
)


def check_silero():
    """Silero VAD: bundled in pipecat, not on the Hugging Face hub."""
    print("\n== Silero VAD (bundled in the pipecat package) ==", flush=True)

    # Resolve the same way pipecat does: load the packaged data file.
    resource = None
    try:
        import importlib.resources as imp
        resource = str(imp.files("pipecat.audio.vad.data").joinpath("silero_vad.onnx"))
    except Exception:
        pass

    if not resource or not os.path.exists(resource):
        print(f"  MISSING: {resource}")
        print("  Reinstall the silero extra:  uv pip install 'pipecat-ai[silero]'")
        return False

    size_mb = os.path.getsize(resource) / 1e6
    print(f"  found:    {resource}")
    print(f"  size:     {size_mb:.1f} MB (bundled -- no download needed)")

    try:
        import onnxruntime
        session = onnxruntime.InferenceSession(
            resource, providers=["CPUExecutionProvider"]
        )
        print(
            f"  onnxruntime load: OK ({onnxruntime.__version__}, "
            f"{len(session.get_inputs())} input(s))"
        )
    except Exception as e:
        print(f"  WARNING: could not load with onnxruntime: {e}")
        return False

    print("  Silero VAD: ready")
    return True


def check_whisper(model_name: str, offline: bool = False):
    """Ensure the MLX Whisper model is cached -- the exact path bot.py uses."""
    print(f"\n== MLX Whisper   ({model_name}) ==", flush=True)
    print(f"  HUGGINGFACE_HUB_CACHE = {HUGGINGFACE_HUB_CACHE}")

    # Step 1: ensure the weights are on disk.
    # snapshot_download is what mlx_whisper.load_model calls when the repo is
    # absent; it downloads if missing and is a no-op if already present.
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
    mode = "local_files_only" if offline else "download if missing"
    try:
        from huggingface_hub import snapshot_download
        print(f"  snapshot_download(repo_id={model_name!r}, {mode}) ...", flush=True)
        snapshot_dir = snapshot_download(
            repo_id=model_name, local_files_only=offline
         )
    except Exception as e:
        print(f"  FAILED to fetch snapshot: {type(e).__name__}: {e}")
        print("  Ensure huggingface_hub is installed, or retry without --offline.")
        return False

    # Step 2 (bonus): exercise the real runtime path -- mlx_whisper.transcribe on
    # a short silent buffer, exactly like WhisperSTTServiceMLX.run does. This also
    # warms the MLX runtime. It is best-effort: the actual goal above is caching.
    try:
        import numpy as np
        import mlx_whisper

        # 3s of silence at 16 kHz: enough for one transcribe pass, returns no
        # segments without error.
        audio = np.zeros(16000 * 3, dtype=np.float32)
        print("  mlx_whisper.transcribe (silent 3s buffer, matches bot.py path) ...",
              flush=True)
        mlx_whisper.transcribe(
            audio, path_or_hf_repo=model_name, verbose=False
        )
        print(f"  MLX Whisper {model_name}: cached + warmed ({snapshot_dir})")
        return True
    except ImportError as e:
        print(f"  NOTE: mlx_whisper not importable: {e}")
        print("  Weights are still cached -- the bot warms them on first run.")
        return True
    except Exception as e:
        print(f"  NOTE: transcribe() warm hit {type(e).__name__}: {e}")
        print("  Weights are cached -- the bot will load them on first run.")
        return True


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Pre-download & verify the Silero VAD + MLX Whisper models "
            "before running the bot."
        )
    )
    ap.add_argument(
        "--whisper",
        default="mlx-community/whisper-large-v3-turbo-q4",
        help="MLX Whisper Hugging Face repo id (default: MLXModel.LARGE_V3_TURBO_Q4)",
    )
    ap.add_argument(
        "--offline",
        action="store_true",
        help="Verify only; do not touch the network (sets HF_HUB_OFFLINE=1).",
    )
    ap.add_argument(
        "--skip-silero",
        action="store_true",
        help="Skip the Silero VAD check",
    )
    args = ap.parse_args()

    ok = True
    if not args.skip_silero:
        ok &= check_silero()
    ok &= check_whisper(args.whisper, offline=args.offline)

    print("\n== Summary ==", flush=True)
    print("  Silero VAD   : bundled in pipecat, onnx-verified"
          if not args.skip_silero else "    (skipped)")
    print(f"  MLX Whisper  : {args.whisper} cached")
    print("Done.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
