"""TTS-слой: два движка на выбор через config.TTS_ENGINE.

f5     — F5-TTS с клонированным голосом (voice_ref.wav/.txt), GPU, живее.
silero — старый движок (полностью локально, CPU), оставлен как быстрый
         откат через .env, если f5 не устроит или GPU будет занят/недоступен.

Единый интерфейс наружу — synthesize_pcm(text), всегда отдаёт сырой PCM
48kHz stereo 16-bit (готовый для discord.PCMAudio), независимо от движка.
"""

import re
from pathlib import Path
from xml.sax.saxutils import escape as _xml_escape

import numpy as np
import torch
import torchaudio

import config

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


def _clean_for_tts(text: str) -> str:
    text = _EMOJI_RE.sub("", text)
    text = _MARKDOWN_RE.sub("", text)
    text = _ELLIPSIS_RE.sub(",", text)
    text = _REPEAT_PUNCT_RE.sub(r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def _to_stereo48k_int16(mono_float: np.ndarray, sample_rate: int) -> bytes:
    """Моно float [-1..1] на произвольной частоте -> формат, который ждёт
    Discord: 48kHz stereo 16-bit PCM."""
    if sample_rate != config.DISCORD_SAMPLE_RATE:
        t = torch.from_numpy(mono_float).unsqueeze(0)
        t = torchaudio.functional.resample(t, sample_rate, config.DISCORD_SAMPLE_RATE)
        mono_float = t.squeeze(0).numpy()
    mono_int16 = (mono_float * 32767.0).clip(-32768, 32767).astype(np.int16)
    stereo = np.repeat(mono_int16[:, None], 2, axis=1)  # дублируем канал: mono -> stereo
    return stereo.tobytes()


if config.TTS_ENGINE == "f5":
    import f5_tts.infer.utils_infer as _f5_utils_infer
    from f5_tts.api import F5TTS

    # Библиотека печатает ref_text/gen_text через голый print() внутри своего
    # модуля напрямую (не через параметр show_info=, который перекрывает
    # только остальные её сообщения) — те две строки лога только загромождали
    # бы bot.log на каждую фразу. Глушим точечно: подменяем print только в
    # неймспейсе этого модуля, на остальной процесс не влияет.
    _f5_utils_infer.print = lambda *a, **k: None

    _f5_model = F5TTS(
        model="F5TTS_v1_Base",
        ckpt_file=config.F5_CKPT_FILE,
        vocab_file=config.F5_VOCAB_FILE,
        device="cuda",
    )
    _f5_ref_text = Path(config.F5_REF_TEXT_FILE).read_text(encoding="utf-8").strip()

    def _synthesize(text: str) -> bytes:
        wav, sr, _ = _f5_model.infer(
            ref_file=config.F5_REF_AUDIO,
            ref_text=_f5_ref_text,
            gen_text=text,
            speed=config.F5_SPEED,
            show_info=lambda *a, **k: None,  # иначе печатает ref_text/gen_text в лог на каждую фразу
            progress=None,  # иначе на каждую фразу лишний прогресс-бар в лог
        )
        return _to_stereo48k_int16(wav, sr)

else:
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

    def _synthesize(text: str) -> bytes:
        audio = _silero_model.apply_tts(
            ssml_text=_to_ssml(text),
            speaker=config.SILERO_SPEAKER,
            sample_rate=config.SILERO_SAMPLE_RATE,
        )  # float32 tensor, mono, [-1..1]
        return _to_stereo48k_int16(audio.numpy(), config.SILERO_SAMPLE_RATE)


def synthesize_pcm(text: str) -> bytes:
    """Текст -> сырой PCM 48kHz stereo 16-bit (готовый для discord.PCMAudio)."""
    text = _clean_for_tts(text)
    if not text or not _HAS_LETTER_RE.search(text):
        return b""
    return _synthesize(text)
