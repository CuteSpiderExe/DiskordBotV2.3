"""Все настройки бота. Значения берутся из .env, здесь — дефолты."""

import os
import re
import sys
import tempfile

from dotenv import load_dotenv

# В обычном запуске (python bot.py) — папка со скриптом. В собранном
# PyInstaller-приложении (--onedir) sys.executable указывает на сам
# BotV2.exe — значит, это и есть папка, где лежат .env/character.txt/
# memory.json, независимо от того, откуда её запустили (ярлык, другой CWD).
if getattr(sys, "frozen", False):
    BASE_DIR = os.path.dirname(sys.executable)
    # В собранном PyInstaller-приложении print() в stdout почему-то никуда
    # не долетает (даже с PYTHONUNBUFFERED=1), а stderr работает нормально —
    # заворачиваем print()'ы туда же. Кодировку stderr тоже приходится
    # фиксировать явно: PYTHONIOENCODING/PYTHONUTF8 здесь не долетают до
    # уже созданного потока, поэтому кириллица без reconfigure() превращается
    # в "???".
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    sys.stdout = sys.stderr
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Silero сам ищет models.yml относительно текущей рабочей директории —
# фиксируем CWD на папку приложения, чтобы это работало независимо от того,
# как и откуда запущен BotV2.exe (ярлык, другой рабочий каталог и т.п.).
os.chdir(BASE_DIR)


def _path(name: str) -> str:
    return name if os.path.isabs(name) else os.path.join(BASE_DIR, name)


load_dotenv(_path(".env"))

# --- Discord ---
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")

# --- LLM (Ollama) ---
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "huihui_ai/qwen3-abliterated:8b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_NUM_THREAD = int(os.getenv("OLLAMA_NUM_THREAD", "12"))
# Модель по умолчанию тянет свой полный обучающий контекст (у qwen3 — 40960),
# если не задать num_ctx явно. KV-кэш на такой контекст не помещается в VRAM
# вместе с весами, и Ollama выгружает большую часть слоёв на CPU — это и есть
# основная причина зависаний/долгих ответов. Голосовому боту с короткой
# историей (SHORT_HISTORY_LEN) хватает несколько тысяч токенов с запасом.
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "4096"))
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "30m")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "256"))
LLM_THINK = os.getenv("LLM_THINK", "true").lower() in {"1", "true", "yes", "on"}
# Если Ollama не отвечает дольше этого — считаем запрос зависшим (сеть/GPU-память
# и т.п.) и обрываем его, а не ждём вечно.
OLLAMA_TIMEOUT_SEC = float(os.getenv("OLLAMA_TIMEOUT_SEC", "45"))

# --- STT (faster-whisper) ---
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "small")
WHISPER_CPU_THREADS = int(os.getenv("WHISPER_CPU_THREADS", "8"))

# --- TTS ---
TTS_ENGINE = os.getenv("TTS_ENGINE", "qwen").lower()

# Qwen runs in the separate environment already used for the voice trials.
QWEN_RUNTIME = os.getenv("QWEN_RUNTIME", "faster").lower()
QWEN_PYTHON = _path(os.getenv("QWEN_PYTHON", "qwen_tts_trial/faster_env/Scripts/python.exe"))
QWEN_MODEL_DIR = _path(os.getenv("QWEN_MODEL_DIR", "qwen_tts_trial/weights/custom"))
QWEN_CACHE_DIR = _path(os.getenv("QWEN_CACHE_DIR", "qwen_tts_trial/cache"))
QWEN_SPEAKER = os.getenv("QWEN_SPEAKER", "Ono_Anna")
QWEN_MAX_TOKENS = int(os.getenv("QWEN_MAX_TOKENS", "768"))
QWEN_TIMEOUT_SEC = float(os.getenv("QWEN_TIMEOUT_SEC", "180"))

# --- TTS (Silero) ---
SILERO_SPEAKER = os.getenv("SILERO_SPEAKER", "xenia")  # xenia/baya/kseniya жен., aidar/eugene муж.
SILERO_SAMPLE_RATE = 48000
SILERO_CPU_THREADS = int(os.getenv("SILERO_CPU_THREADS", "4"))

# --- Персонаж и память ---
CHARACTER_FILE = _path(os.getenv("CHARACTER_FILE", "character.txt"))
MEMORY_FILE = _path(os.getenv("MEMORY_FILE", "memory.json"))
SHORT_HISTORY_LEN = int(os.getenv("SHORT_HISTORY_LEN", "16"))
SUMMARIZE_AFTER = int(os.getenv("SUMMARIZE_AFTER", "24"))

# --- Аудио-константы ---
DISCORD_SAMPLE_RATE = 48000   # PCM, который отдаёт/принимает Discord (stereo 16-bit)
VAD_SAMPLE_RATE = 16000       # webrtcvad поддерживает 8/16/32/48 кГц; Whisper любит 16k
FRAME_MS = 20
SILENCE_LIMIT_SEC = float(os.getenv("SILENCE_LIMIT_SEC", "0.8"))
REQUIRE_WAKE_WORD = os.getenv("REQUIRE_WAKE_WORD", "false").lower() in {"1", "true", "yes", "on"}
# Пока бот говорит, чужая речь распознаётся как обычная фраза (без мгновенного
# барж-ина по VAD) — перебивает её только если в распознанном тексте есть
# одно из этих слов (обращение по имени). Через запятую, регистр не важен.
WAKE_WORDS = [w.strip().lower() for w in os.getenv("WAKE_WORDS", "ари,ори,ария,эри").split(",") if w.strip()]
# Сравнение по границам слова, а не по подстроке — иначе "ори" ложно
# совпадает с середине слов вроде "истОРИю", "приОРИтет".
WAKE_WORD_RE = (
    re.compile(r"\b(?:" + "|".join(re.escape(w) for w in WAKE_WORDS) + r")\b", re.IGNORECASE)
    if WAKE_WORDS
    else None
)
# Минимальная длительность фразы, чтобы вообще отдавать её в Whisper (сек)
MIN_UTTERANCE_SEC = float(os.getenv("MIN_UTTERANCE_SEC", "0.4"))
# Через сколько секунд тишины в канале бот сам заговорит на случайную тему
IDLE_CHATTER_SEC = float(os.getenv("IDLE_CHATTER_SEC", "120"))
IDLE_SONG_SEC = float(os.getenv("IDLE_SONG_SEC", "300"))

# --- Watchdog (watchdog.py) ---
# Некоторые вызовы discord-voice-клиента (play/stop/start_recording) в этой
# экспериментальной ветке py-cord иногда зависают насмерть на нативном
# уровне и держат GIL — тогда не спасает даже asyncio.wait_for, и весь
# процесс замирает целиком. bot.py периодически обновляет этот файл;
# watchdog.py снаружи процесса следит за этим и перезапускает bot.py
# принудительно, если файл давно не обновлялся.
HEARTBEAT_FILE = _path(os.getenv("HEARTBEAT_FILE", "heartbeat.txt"))
HEARTBEAT_INTERVAL_SEC = float(os.getenv("HEARTBEAT_INTERVAL_SEC", "3"))
HEARTBEAT_TIMEOUT_SEC = float(os.getenv("HEARTBEAT_TIMEOUT_SEC", "20"))

TEMP_DIR = tempfile.gettempdir()
