#!/usr/bin/env python3
"""
bot_gui.py -- a PyQt5 control panel for the macOS local voice agent.

The agent itself is a WebRTC server (server/bot.py) that a browser client
connects to; this GUI is purely the *control surface*: it edits the same
options that live in server/config.py, then launches bot.py as a subprocess
and tails its logs.

Design notes
------------
* It reuses the project's own Config so the form pre-fills with the
  *effective* configuration (env / .env / defaults), not the bare dataclass
  defaults.
* Only fields the user actually changes are passed as --flags to bot.py, so
  everything else falls through to env / .env unchanged. This mirrors the
  project's precedence: CLI > env > default.
* The subprocess is a child of the GUI; closing the window (or pressing Stop)
  close signals that whole process group -- the uv-run -> python -> uvicorn
  chain -- so the port is always freed. Logs stream back over a pipe by
  a QThread, so the main thread stays free.

Run it
------
It needs PyQt5 (not in the voice venv). Any Python with PyQt5 works:

    pip install PyQt5
    python server/bot_gui.py

or, if PyQt5 lives in the project venv:

    uv run --with PyQt5 python server/bot_gui.py

The launch uses `uv run` when `uv` is on PATH, otherwise
server/.venv/bin/python, so the agent boots in its real runtime either way.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import fields

from PyQt5 import QtCore, QtGui, QtWidgets

# --- Locate the server dir and reuse its Config snapshot --------------------
SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SERVER_DIR)

# Mirror bot.py: load .env (if present) before reading env-driven defaults.
try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(SERVER_DIR, ".env"))
except Exception:
    pass

try:
    from config import Config

    EFFECTIVE = Config.from_env()
except Exception:
    # Fall back to a bare Config so the GUI still opens without config deps.
    CONFIG_AVAILABLE = False

    class Config:  # type: ignore
        pass

    EFFECTIVE = Config()
    CONFIG_AVAILABLE = True


# field name -> bot.py CLI flag. Booleans use the *disable* flag so we only
# add a flag when the user turns a default-on option off.
FIELD_FLAG = {
    "host": "--host",
    "port": "--port",
    "llm_model": "--llm-model",
    "llm_base_url": "--llm-base-url",
    "llm_api_key": "--llm-api-key",
    "llm_max_tokens": "--llm-max-tokens",
    "stt_model": "--stt-model",
    "tts_model": "--tts-model",
    "tts_voice": "--tts-voice",
    "tts_sample_rate": "--tts-sample-rate",
    "vad_stop_secs": "--vad-stop-secs",
    "smart_turn_model": "--smart-turn-model",
    "aggregation_timeout": "--aggregation-timeout",
    "ice_servers": "--ice-servers",
    "system_prompt_file": "--system-prompt-file",
    "system_prompt": "--system-prompt",
    "enable_metrics": "--no-metrics",
    "enable_usage_metrics": "--no-usage-metrics",
}

# Group the form the way bot.py / config.py do, with friendly labels.
# Each entry: (field_name, label, kind)  kind in {text, int, float, bool, file}
GROUPS = [
    ("Server", [
        ("host", "Host", "text"),
        ("port", "Port", "int"),
    ]),
    ("LLM (OpenAI-compatible / Ollama)", [
        ("llm_model", "Model", "text"),
        ("llm_base_url", "Base URL", "text"),
        ("llm_api_key", "API key", "text"),
        ("llm_max_tokens", "Max tokens", "int"),
    ]),
    ("STT (MLX Whisper)", [
        ("stt_model", "Model", "text"),
    ]),
    ("TTS (Kokoro / Marvis, auto-picked by model name)", [
        ("tts_model", "Model", "text"),
        ("tts_voice", "Voice", "text"),
        ("tts_sample_rate", "Sample rate", "int"),
    ]),
    ("VAD / turn detection", [
        ("vad_stop_secs", "VAD stop secs", "float"),
        ("smart_turn_model", "Smart-turn model (blank = download)", "text"),
    ]),
    ("Conversation prompt", [
        ("system_prompt_file", "Prompt file (prefer over inline)", "file"),
        ("system_prompt", "Inline prompt (used when file is blank)", "text"),
    ]),
    ("Misc", [
        ("ice_servers", "ICE servers (comma separated)", "text"),
        ("aggregation_timeout", "Aggregation timeout", "float"),
        ("enable_metrics", "Pipeline metrics", "bool"),
        ("enable_usage_metrics", "Usage metrics", "bool"),
    ]),
]

RUNNER_AUTO, RUNNER_UV, RUNNER_VENV, RUNNER_PY = "auto", "uv", "venv", "python"


def _effective_value(name: str):
    """The current effective value of a Config field (or None if unknown)."""
    return getattr(EFFECTIVE, name, None)


class ReaderThread(QtCore.QThread):
        # Drains a Popen stdout (stderr merged in) and forwards lines to the GUI
        # over a queued signal, so Qt's main event loop is never blocked.
    line = QtCore.pyqtSignal(str, str)
    exited = QtCore.pyqtSignal(int)

    def __init__(self, proc: "subprocess.Popen") -> None:
        super().__init__()
        self.proc = proc
        self._alive = True

    def run(self) -> None:
        try:
            for raw in iter(self.proc.stdout.readline, b""):
                if not self._alive or not raw:
                    break
                self.line.emit("OUT", raw.decode("utf-8", "replace").rstrip("\n"))
        except Exception as e:
            self.line.emit("ERR", f"reader error: {e}")
        self._alive = False
        try:
            code = self.proc.wait()
        except Exception:
            code = -1
        self.exited.emit(int(code))


class BotControlPanel(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("macOS Local Voice Agent - Control Panel")
        self.resize(760, 860)

        self.widgets: dict[str, QtWidgets.QWidget] = {}
            # The bot runs as a child in its own process group (see start_bot),
            # so Stop / close signals the whole uv-run -> python -> uvicorn
            # chain at once and frees the configured port.
        self.proc: "subprocess.Popen" | None = None
        self.reader: "ReaderThread" | None = None
        self._build_ui()
        self._apply_values()
        self._update_state()

    # --- UI construction ----------------------------------------------------
    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)

        self._build_form(root)
        self._build_log(root)
        self._build_actions(root)
        self._build_statusbar()

    def _build_form(self, parent_layout: "QtWidgets.QVBoxLayout") -> None:
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        form_host = QtWidgets.QWidget()
        parent_layout.addWidget(scroll, 1)
        form = QtWidgets.QFormLayout(form_host)

        self.form = form
        scroll.setWidget(form_host)

        for title, items in GROUPS:
            box = QtWidgets.QGroupBox(title)
            box_form = QtWidgets.QFormLayout(box)
            for name, label, kind in items:
                self._add_field(box_form, name, label, kind)
            parent_layout.addWidget(box)

    def _add_field(self, form: "QtWidgets.QFormLayout", name: str,
                   label: str, kind: str) -> None:
        wid: QtWidgets.QWidget
        if kind == "int":
            wid = QtWidgets.QSpinBox()
            wid.setRange(0, 2_000_000_000)
        elif kind == "float":
            wid = QtWidgets.QDoubleSpinBox()
            wid.setRange(0.0, 1_000_000.0)
            wid.setDecimals(4)
        elif kind == "bool":
            wid = QtWidgets.QCheckBox()
        elif kind == "file":
            wid = self._build_file_field(form, name, label)
            form.addRow(wid)
            return
        else:  # text
            wid = QtWidgets.QLineEdit()
        form.addRow(label, wid)
        self.widgets[name] = wid

    def _build_file_field(self, form: "QtWidgets.QFormLayout",
                          name: str, label: str) -> QtWidgets.QWidget:
        row = QtWidgets.QWidget()
        row_l = QtWidgets.QHBoxLayout(row)
        row_l.setContentsMargins(0, 0, 0, 0)
        edit = QtWidgets.QLineEdit()
        browse = QtWidgets.QPushButton("...")
        browse.setFixedWidth(32)
        row_l.addWidget(edit, 1)
        row_l.addWidget(browse)
        browse.clicked.connect(lambda _=False, e=edit: self._pick_file(e))
        form.addRow(label, row)
        self.widgets[name] = edit
        return row

    def _build_log(self, parent_layout: "QtWidgets.QVBoxLayout") -> None:
        log_host = QtWidgets.QGroupBox("Logs")
        log_v = QtWidgets.QVBoxLayout(log_host)
        self.log_view = QtWidgets.QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setStyleSheet("QTextEdit { background: #1e1e1e; color: #d4d4d4; }")
        log_v.addWidget(self.log_view)
        parent_layout.addWidget(log_host, 1)

    def _build_actions(self, parent_layout: "QtWidgets.QVBoxLayout") -> None:
        bar = QtWidgets.QHBoxLayout()
        self.run_btn = QtWidgets.QPushButton("Start")
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.open_btn = QtWidgets.QPushButton("Open web client")
        self.save_btn = QtWidgets.QPushButton("Save .env")
        self.clear_btn = QtWidgets.QPushButton("Clear")
        self.runner_combo = QtWidgets.QComboBox()
        for r in [RUNNER_AUTO, RUNNER_UV, RUNNER_VENV, RUNNER_PY]:
            self.runner_combo.addItem(r, r)
        run_row = QtWidgets.QHBoxLayout()
        run_row.addWidget(self.run_btn)
        run_row.addWidget(self.stop_btn)
        run_row.addWidget(self.open_btn)
        run_row.addWidget(self.clear_btn)
        run_row.addStretch()
        stop_row = QtWidgets.QHBoxLayout()
        stop_row.addWidget(self.save_btn)
        stop_row.addStretch()
        stop_row.addWidget(self.runner_combo)
        parent_layout.addLayout(run_row)
        parent_layout.addLayout(stop_row)

        self.run_btn.clicked.connect(self.start_bot)
        self.stop_btn.clicked.connect(self.stop_bot)
        self.open_btn.clicked.connect(self.open_web_client)
        self.clear_btn.clicked.connect(self.log_view.clear)
        self.save_btn.clicked.connect(self.save_env)

    def _build_statusbar(self) -> None:
        self.status_label = QtWidgets.QLabel("stopped")
        self.status_label.setFrameShape(QtWidgets.QFrame.Box)
        self.url_label = QtWidgets.QLabel("")
        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(self.status_label)
        bar.addWidget(self.url_label)
        bar.addStretch()
        wrap = QtWidgets.QWidget()
        wrap.setLayout(bar)
        self.statusBar().addPermanentWidget(wrap)


    # ---- Process wiring (subprocess lifecycle) ---------------------------
    def _on_line(self, tag: str, text: str) -> None:
        self._log(tag, text)

    def _on_exited(self, code: int) -> None:
        self._log("SVC", f"process exited with code {code}")
        # Free the port even if the server outlived the group (re-parented).
        self._reap_port()
        self.proc = None
        self.reader = None
        self._update_state()

    def _signal_group(self, pid: int, sig: int) -> None:
        try:
            os.killpg(os.getpgid(pid), sig)
        except ProcessLookupError:
            return
        except OSError:
            # Group may be gone; fall back to killing the pid directly.
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass

    def _listeners_on(self, port: int) -> list:
        # PIDs still bound+listening on the given port (best effort, via lsof).
        lsof = shutil.which("lsof")
        if not lsof:
            return []
        try:
            out = subprocess.run(
                [lsof, "-ti", f"tcp:{port}", "-sTCP:LISTEN"],
                capture_output=True, text=True, timeout=3).stdout
            pids = [int(t.lstrip("-")) for t in out.split()
                    if t.lstrip("-").isdigit()]
            return sorted(set(pids))
        except Exception:
            return []

    def _running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def _reap_port(self) -> None:
        # Hard guarantee the port is freed: TERM whatever is still listening,
        # then SIGKILL stragglers after a brief grace.
        port = self._value("port") or 7860
        try:
            pids = self._listeners_on(int(port))
        except Exception:
            pids = []
        if not pids:
            return
        self._log("GUI", f"reaping listener(s) {pids} still on :{port}")
        for pid in pids:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        time.sleep(0.5)
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


    def _log(self, tag: str, text: str) -> None:
        self.log_view.append(f"[{tag}] {text}")
        # Keep the view near the end without thrashing the scrollbar.
        sb = self.log_view.verticalScrollBar()
        sb.setValue(sb.maximum())

    # --- State / helpers ----------------------------------------------------
    def _update_state(self) -> None:
        running = self._running()
        self.run_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        self.open_btn.setEnabled(not running)
        self.runner_combo.setEnabled(not running)
        self.status_label.setText("running" if running else "stopped")
        self.status_label.setStyleSheet(
            "QLabel { background: #c8e6c9; padding: 3px 10px; }"
            if running else
            "QLabel { background: #eeeeee; padding: 3px 10px; }"
        )
        host = self._value("host") or "localhost"
        port = self._value("port") or "7860"
        self.url_label.setText(f"http://{host}:{port}   (POST /api/offer)")

    def _pick_file(self, edit: QtWidgets.QLineEdit) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Choose prompt file",
            os.path.join(os.path.dirname(SERVER_DIR), "prompts"),
        )
        if path:
            edit.setText(path)

    def _value(self, name: str):
        """Read the current form value for a field as a native type."""
        wid = self.widgets.get(name)
        if wid is None:
            return None
        kind = next((k for _, items in GROUPS for n, _, k in items if n == name), "text")
        if isinstance(wid, QtWidgets.QCheckBox):
            return wid.isChecked()
        if isinstance(wid, QtWidgets.QSpinBox):
            return wid.value()
        if isinstance(wid, QtWidgets.QDoubleSpinBox):
            return wid.value()
        return (wid.text() if isinstance(wid, QtWidgets.QLineEdit) else "")

    def _set_value(self, name: str, val) -> None:
        wid = self.widgets.get(name)
        if wid is None:
            return
        if isinstance(wid, QtWidgets.QCheckBox):
            wid.setChecked(bool(val))
        elif isinstance(wid, QtWidgets.QSpinBox):
            wid.setValue(int(val or 0))
        elif isinstance(wid, QtWidgets.QDoubleSpinBox):
            wid.setValue(float(val or 0.0))
        elif isinstance(wid, QtWidgets.QLineEdit) or isinstance(
                wid, QtWidgets.QWidget):
            # File fields live inside a row container; find the QLineEdit child.
            edit = wid if isinstance(wid, QtWidgets.QLineEdit) else wid
            if isinstance(val, list):
                val = ", ".join(str(v) for v in val)
            edit.setText("" if val is None else str(val))

    def _apply_values(self) -> None:
        """Pre-fill the form from the effective Config (env/.env/defaults)."""
        for _, items in GROUPS:
            for name, _, _ in items:
                val = _effective_value(name)
                if val is not None:
                    # ice_servers is a list on Config; normalise for the text field.
                    if name == "ice_servers":
                        self._set_value(name, ", ".join(val))
                    else:
                        self._set_value(name, val)

    # --- Subprocess launch --------------------------------------------------
    def _runner_argv(self, bot_py: str, flags: list[str]) -> list[str]:
        """Choose the interpreter/runner and assemble the full argv for the bot."""
        choice = self.runner_combo.currentData() or RUNNER_AUTO
        venv_py = os.path.join(SERVER_DIR, ".venv", "bin", "python")
        uv = shutil.which("uv")
        if choice == RUNNER_UV or (choice == RUNNER_AUTO and uv):
            return [uv or "uv", "run", "python", bot_py, *flags]
        if choice == RUNNER_VENV or (choice == RUNNER_AUTO and os.path.exists(venv_py)):
            return [venv_py, bot_py, *flags]
        if choice == RUNNER_PY:
            return [sys.executable, bot_py, *flags]
        # auto fell through: best guess is the system python.
        return [sys.executable, bot_py, *flags]

    def _build_flags(self) -> list[str]:
        """Flags for changed fields only, so untouched values fall through to env."""
        out: list[str] = []
        for _, items in GROUPS:
            for name, label, kind in items:
                cur = self._value(name)
                eff = _effective_value(name)
                if kind in ("int", "float", "text", "file"):
                    cur_s = "" if cur is None else (",".join(cur) if isinstance(cur, list) else str(cur))
                    eff_s = "" if eff is None else (",".join(eff) if isinstance(eff, list) else str(eff))
                    if name == "system_prompt_file" and not cur_s:
                        continue  # blank = use whatever env/inline says
                                        # Pass the value raw: subprocess.Popen runs the program without a
                    # shell, so shell-quoting would be kept literally -- that is what
                    # shipped "qwen3.8:27b-mlx" to Ollama. Empty value => no override.
                    if cur_s != eff_s and cur_s:
                        out.append(f"{FIELD_FLAG[name]}={cur_s}")
                elif kind == "bool":
                    # Only emit the disable flag when the user turns a default-on
                    # option off.
                    if not cur and eff:
                        out.append(FIELD_FLAG[name])
        return out

    def start_bot(self) -> None:
        if self._running():
            return
        bot_py = os.path.join(SERVER_DIR, "bot.py")
        flags = self._build_flags()
        argv = self._runner_argv(bot_py, flags)
        self.log_view.clear()
        self._log("GUI", "launching: " + " ".join(argv))
        self._log("GUI", "cwd: " + SERVER_DIR + "         |   flags: " + " ".join(flags) or " (none)")
            # bot.py loads server/.env itself and our --flags win over .env by
            # precedence (CLI > env > default), so no child env injection needed.
        try:
            proc = subprocess.Popen(
                 argv,
                 cwd=SERVER_DIR,
                 stdout=subprocess.PIPE,
                 stderr=subprocess.STDOUT,
                 start_new_session=True,        # own process group; killable as a unit
                 )
        except Exception as e:         # a bad runner/flag must not abort the app
            self._log("ERR", f"failed to start process: {e}")
            self._update_state()
            return
        self.proc = proc
        self.reader = ReaderThread(proc)
        self.reader.line.connect(self._on_line)
        self.reader.exited.connect(self._on_exited)
        self.reader.start()
        self._update_state()

    def stop_bot(self) -> None:
        proc = self.proc
        if proc is None or proc.poll() is not None:
            return
        self._log("GUI", "stopping...")
            # Terminate the whole group: TERM, brief grace, then escalate to KILL.
        self._signal_group(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._log("GUI", "still up after 5s, escalating to SIGKILL")
            self._signal_group(proc.pid, signal.SIGKILL)
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                pass
        self._reap_port()
        self.proc = None
        self._update_state()

    def open_web_client(self) -> None:
        host = self._value("host") or "localhost"
        port = self._value("port") or "7860"
        url = QtCore.QUrl(f"http://{host}:{port}")
        QtGui.QDesktopServices.openUrl(url)

    def save_env(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save configuration to .env",
            os.path.join(SERVER_DIR, ".env"),
        )
        if not path:
            return
        lines = [
            "# Generated by bot_gui.py",
            f"BOT_HOST={_quoted(self._value('host') or 'localhost')}",
            f"BOT_PORT={self._value('port') or 7860}",
            f"LLM_MODEL={_quoted(self._value('llm_model') or 'gemma3n:e4b')}",
            f"LLM_BASE_URL={_quoted(self._value('llm_base_url') or 'http://127.0.0.1:11434/v1')}",
            f"LLM_API_KEY={_quoted(self._value('llm_api_key') or 'dummyKey')}",
            f"LLM_MAX_TOKENS={self._value('llm_max_tokens') or 4096}",
            f"STT_MODEL={_quoted(self._value('stt_model') or 'mlx-community/whisper-large-v3-turbo-q4')}",
            f"TTS_MODEL={_quoted(self._value('tts_model') or 'mlx-community/Kokoro-82M-bf16')}",
            f"TTS_VOICE={_quoted(self._value('tts_voice') or 'af_heart')}",
            f"TTS_SAMPLE_RATE={self._value('tts_sample_rate') or 24000}",
            f"VAD_STOP_SECS={self._value('vad_stop_secs') or 0.2}",
            f"SMART_TURN_MODEL={_quoted(self._value('smart_turn_model') or '')}",
            f"SYSTEM_PROMPT_FILE={_quoted(self._value('system_prompt_file') or '')}",
            f"AGGREGATION_TIMEOUT={self._value('aggregation_timeout') or 0.05}",
            f"ICE_SERVERS={_quoted(self._value('ice_servers') or 'stun:stun.l.google.com:19302')}",
            f"ENABLE_METRICS={'true' if self._value('enable_metrics') else 'false'}",
            f"ENABLE_USAGE_METRICS={'true' if self._value('enable_usage_metrics') else 'false'}",
        ]
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            self._log("GUI", f"wrote {path}")
        except OSError as e:
            self._log("ERR", f"could not write {path}: {e}")

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:    # noqa: N802
        if self._running():
            self.stop_bot()
        super().closeEvent(event)


# --- Module helpers ---------------------------------------------------------

def _quoted(s: str) -> str:
    return f'"{s}"' if any(c in s for c in " \t:=\"'<>|&") else s


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("MacOS Local Voice Agent")
    win = BotControlPanel()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
