"""One serialized persistent TTS worker shared by all voice channels."""

import atexit
import base64
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import time
import request_control

import config


class QwenBackend:
    def __init__(self):
        self._lock = threading.Lock()
        self._process = None
        self._responses = None

    @staticmethod
    def _read_output(stream, responses):
        try:
            for line in stream:
                try:
                    responses.put(json.loads(line))
                except json.JSONDecodeError:
                    responses.put({"error": "Invalid response from Qwen worker"})
        finally:
            responses.put({"error": "Qwen worker exited; see qwen_tts.log"})
            stream.close()

    def _receive(self):
        deadline = time.monotonic() + config.QWEN_TIMEOUT_SEC
        while True:
            try:
                request_control.check()
            except request_control.RequestCancelled:
                self.close()
                raise
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.close()
                raise TimeoutError("Qwen TTS timed out; worker terminated")
            try:
                response = self._responses.get(timeout=min(.1, remaining))
                request_control.check()
                break
            except queue.Empty:
                continue
        if "error" in response:
            raise RuntimeError(response["error"])
        return response

    def _start(self):
        if self._process is not None and self._process.poll() is None:
            return
        self.close()
        if not Path(config.QWEN_PYTHON).is_file() or not Path(config.QWEN_MODEL_DIR).is_dir():
            raise RuntimeError("Qwen Python/model not found; check QWEN_PYTHON and QWEN_MODEL_DIR")
        with open(Path(config.BASE_DIR) / "qwen_tts.log", "a", encoding="utf-8") as log:
            self._process = subprocess.Popen([
                config.QWEN_PYTHON, "-u", str(Path(__file__).with_name("qwen_worker.py")),
                "--model", config.QWEN_MODEL_DIR, "--cache", config.QWEN_CACHE_DIR,
                "--parent-pid", str(os.getpid()), "--max-tokens", str(config.QWEN_MAX_TOKENS),
                "--speaker", config.QWEN_SPEAKER,
                "--runtime", config.QWEN_RUNTIME,
            ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
                text=True, encoding="utf-8", cwd=config.BASE_DIR,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        self._responses = queue.Queue()
        threading.Thread(target=self._read_output,
            args=(self._process.stdout, self._responses), daemon=True).start()
        try:
            if not self._receive().get("ready"):
                raise RuntimeError("Qwen worker did not initialize")
        except Exception:
            self.close()
            raise

    def warmup(self):
        with self._lock:
            self._start()

    def synthesize(self, text, style="normal"):
        with self._lock:
            request_control.check()
            self._start()
            try:
                self._process.stdin.write(json.dumps({"text": text, "style": style,
                    "speaker": config.QWEN_SPEAKER}, ensure_ascii=False) + "\n")
                self._process.stdin.flush()
                return base64.b64decode(self._receive()["pcm"], validate=True)
            except Exception:
                self.close()
                raise

    def close(self):
        process, self._process = self._process, None
        if process is None:
            return
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
        if process.stdin:
            process.stdin.close()


backend = QwenBackend()
atexit.register(backend.close)
