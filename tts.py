"""Local Qwen CustomVoice or Silero -> Discord 48 kHz stereo PCM.

Qwen runs persistently in its own environment to isolate its dependencies.
"""

import re
from pathlib import Path
from xml.sax.saxutils import escape as _xml_escape

import numpy as np
from num2words import num2words

import config
from audio_pcm import stereo48k
from voice_style import delivery, strip_tool_markup, strip_html_markup

# Эмодзи и markdown-мусор, который иногда проскакивает из LLM: TTS их не
# озвучивает осмысленно (тишина/спотыкание), поэтому чистим перед синтезом.
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F1E6-\U0001F1FF"
    "\U00002190-\U000021FF"
    "\U00002B00-\U00002BFF"
    "\U0000FE0F"
    "]+",
    flags=re.UNICODE,
)
_MARKDOWN_RE = re.compile(r"[*_~`#]+")
# Многоточия и повторы знаков препинания ("Ой-ой-ой...", "Что?!") — обычная
# разговорная манера персонажа, но движки трактуют каждый знак буквально и
# на таких местах спотыкаются/сбивают паузы. Схлопываем в одиночные знаки —
# смысл и интонация не теряются, а произношение становится ровнее.
_ELLIPSIS_RE = re.compile(r"\.{2,}|…+")
_REPEAT_PUNCT_RE = re.compile(r"([!?])\1+")
# Нарезка на предложения в llm.py иногда выделяет кусок из одних знаков
# препинания (например "..." после закрывающей кавычки предыдущего
# предложения) — синтез текста без единой буквы/цифры падает или молчит.
# Отсекаем такое здесь же, как и пустой текст.
_HAS_LETTER_RE = re.compile(r"\w", re.UNICODE)

# Нормализуем числа для предсказуемого чтения; род единиц угадываем
# по следующему слову, как и в прежнем синтезе.
_NUMBER_RE = re.compile(r"\d+")
_FEMININE_UNIT_RE = re.compile(r"^(минут|секунд|недел|штук)", re.IGNORECASE)


def _numbers_to_words(text: str) -> str:
    def _repl(m: re.Match) -> str:
        tail = text[m.end() : m.end() + 15].strip()
        is_feminine = bool(_FEMININE_UNIT_RE.match(tail))
        try:
            # num2words падает с KeyError на gender=None — параметр либо
            # опущен вовсе (маскулинная форма по умолчанию), либо строка.
            if is_feminine:
                return num2words(int(m.group()), lang="ru", gender="feminine")
            return num2words(int(m.group()), lang="ru")
        except Exception:
            return m.group()

    return _NUMBER_RE.sub(_repl, text)


def _clean_for_tts(text: str) -> str:
    text = _EMOJI_RE.sub("", text)
    text = _MARKDOWN_RE.sub("", text)
    if config.TTS_ENGINE != "qwen":
        text = _ELLIPSIS_RE.sub(",", text)
    text = _REPEAT_PUNCT_RE.sub(r"\1", text)
    text = _numbers_to_words(text)
    return re.sub(r"\s+", " ", text).strip()


def _to_stereo48k_int16(mono_float: np.ndarray, sample_rate: int) -> bytes:
    return stereo48k(mono_float, sample_rate)


if config.TTS_ENGINE == "qwen":
    from qwen_backend import backend

    def _synthesize(text: str, style: str = "normal") -> bytes:
        return backend.synthesize(text, style)

elif config.TTS_ENGINE == "silero":
    import torch
    torch.set_num_threads(config.SILERO_CPU_THREADS)
    from silero import silero_tts

    _silero_model, _ = silero_tts(language="ru", speaker="v5_ru")
    _silero_model.to(torch.device("cpu"))

    # Без явного <break> Silero почти не делает пауз на знаках препинания —
    # читает ровным потоком, "пустовато". SSML проверен эмпирически: реально
    # удлиняет паузу (не просто косметика в тексте, которую движок игнорирует).
    _PAUSE_RE = re.compile(r"([,;:])|([.!?]+)")

    def _to_ssml(text: str) -> str:
        def _repl(m: re.Match) -> str:
            if m.group(1):
                return m.group(1) + '<break time="180ms"/>'
            return m.group(2) + '<break time="350ms"/>'

        body = _PAUSE_RE.sub(_repl, _xml_escape(text))
        return f"<speak><p>{body}</p></speak>"

    def _synthesize(text: str, style: str = "normal") -> bytes:
        audio = _silero_model.apply_tts(
            ssml_text=_to_ssml(text),
            speaker=config.SILERO_SPEAKER,
            sample_rate=config.SILERO_SAMPLE_RATE,
        )  # float32 tensor, mono, [-1..1]
        return _to_stereo48k_int16(audio.numpy(), config.SILERO_SAMPLE_RATE)


else:
    raise ValueError(f"Unknown TTS_ENGINE: {config.TTS_ENGINE}")


def warmup():
    if config.TTS_ENGINE == "qwen":
        backend.warmup()


def synthesize_pcm(text: str) -> bytes:
    """Текст -> сырой PCM 48kHz stereo 16-bit (готовый для discord.PCMAudio)."""
    text, style = delivery(strip_tool_markup(strip_html_markup(text)))
    if isinstance(style, dict):
        style = {"segments": [{**part, "text": _clean_for_tts(part["text"])}
                              for part in style["segments"]]}
    text = _clean_for_tts(text)
    if not text or not _HAS_LETTER_RE.search(text):
        return b""
    return _synthesize(text, style)
