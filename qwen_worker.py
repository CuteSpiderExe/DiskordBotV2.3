"""Persistent isolated CustomVoice process. stdout is a JSON-lines protocol."""

import argparse
import base64
import contextlib
import json
import os
from pathlib import Path
import sys
import threading
import time


def warmup_model(model, speaker, instructions, convert):
    """Exercise generation and audio decoding; never publish warmup audio."""
    wavs, rate = model.generate_custom_voice(
        text="Привет!", language="Russian", speaker=speaker,
        instruct=instructions, temperature=0.8, top_p=0.9, max_new_tokens=64)
    convert(wavs[0], rate)


def watch_parent(pid):
    # A watchdog may forcibly kill bot.py; do not leave an orphan model on GPU.
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x00100000, False, pid)
        if not handle:
            os._exit(0)
        kernel.WaitForSingleObject(handle, 0xFFFFFFFF)
        kernel.CloseHandle(handle)
        os._exit(0)
    else:
        import time
        while os.getppid() == pid:
            time.sleep(2)
        os._exit(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--speaker", default="Ono_Anna")
    parser.add_argument("--runtime", choices=("standard", "faster"), default="standard")
    args = parser.parse_args()
    threading.Thread(target=watch_parent, args=(args.parent_pid,), daemon=True).start()
    cache = Path(args.cache)
    for key, suffix in [("HF_HOME", "huggingface"), ("TORCH_HOME", "torch"), ("NUMBA_CACHE_DIR", "numba")]:
        os.environ[key] = str(cache / suffix)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    sys.stdin.reconfigure(encoding="utf-8")
    transport = sys.stdout

    def send(data):
        transport.write(json.dumps(data, ensure_ascii=False) + "\n")
        transport.flush()

    with contextlib.redirect_stdout(sys.stderr):
        import torch
        from qwen_tts import Qwen3TTSModel
        from audio_pcm import stereo48k
        from voice_style import STYLE_INSTRUCTIONS, delivery_instruction
        torch.set_num_threads(4)
        if args.runtime == "faster":
            from faster_qwen3_tts import FasterQwen3TTS
            model = FasterQwen3TTS.from_pretrained(args.model, device="cuda:0",
                dtype=torch.bfloat16, attn_implementation="sdpa",
                max_seq_len=max(2048, args.max_tokens + 512))
        else:
            model = Qwen3TTSModel.from_pretrained(args.model, device_map="cuda:0",
                dtype=torch.bfloat16, attn_implementation="sdpa")
        started = time.monotonic()
        with torch.inference_mode():
            warmup_model(model, args.speaker, STYLE_INSTRUCTIONS["normal"], stereo48k)
        torch.cuda.synchronize()
        print(f"[warmup] Пробный синтез завершён за {time.monotonic() - started:.1f}s; "
              f"{args.runtime}, cuda, bfloat16, sdpa; аудио не воспроизводится.", flush=True)
        send({"ready": True})
        for line in sys.stdin:
            try:
                request = json.loads(line)
                style = request.get("style", "normal")
                language = request.get("language", "Russian")
                if language not in ("Russian", "Japanese"):
                    raise ValueError("Unsupported speech language")
                with torch.inference_mode():
                    wavs, rate = model.generate_custom_voice(text=request["text"], language=language,
                        speaker=request["speaker"], instruct=delivery_instruction(style, language),
                        temperature=0.8, top_p=0.9, max_new_tokens=args.max_tokens)
                pcm = stereo48k(wavs[0], rate)
                send({"pcm": base64.b64encode(pcm).decode("ascii")})
            except Exception as exc:
                send({"error": str(exc)})


if __name__ == "__main__":
    main()
