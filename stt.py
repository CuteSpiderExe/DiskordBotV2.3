"""STT-слой: faster-whisper, принимает аудио напрямую из памяти (numpy)."""

import re

import numpy as np
from faster_whisper import WhisperModel

import config

_model = WhisperModel(
    config.WHISPER_MODEL_SIZE,
    device="cpu",
    compute_type="int8",
    cpu_threads=config.WHISPER_CPU_THREADS,
)

# "Заученные" галлюцинации: Whisper видел эти фразы (титры фильмов/роликов)
# тысячи раз в обучении и на тишине/шуме выдаёт их уверенно — no_speech_prob/
# avg_logprob такое не ловят (модель не "сомневается", она реально помнит
# именно этот текст). Список пополняется по мере того, что реально всплывает.
_HALLUCINATION_RE = re.compile(
    r"редактор субтитров|корректор [а-я]\.|субтитры создавал|"
    r"продолжение следует|подписывайтесь на канал",
    re.IGNORECASE,
)
# Одно и то же слово подряд 3+ раза — тоже типичный узор галлюцинации на
# шуме, а не то, что говорят вслух; compression_ratio не всегда успевает
# сработать на короткой фразе.
_REPEATED_WORD_RE = re.compile(r"\b(\w+)\b(?:\s+\1\b){2,}", re.IGNORECASE)


def transcribe_pcm16(pcm_bytes: bytes) -> str:
    """
    Принимает сырой PCM: mono, 16 kHz, 16-bit signed — ровно то,
    что накапливает VoiceSink. Никаких временных файлов.
    """
    if not pcm_bytes:
        return ""
    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    # Буфер может содержать паузу внутри (две фразы, разделённые молчанием
    # короче SILENCE_LIMIT_SEC, склеиваются в одну — см. voice_sink.py). Без
    # vad_filter Whisper пытается озвучить и эту паузу, что даёт мусор на стыке.
    # condition_on_previous_text=False — иначе одна галлюцинация на тишине
    # затравливает следующий сегмент внутри того же вызова, и модель начинает
    # уверенно "договаривать" её же текст дальше.
    # word_timestamps + hallucination_silence_threshold — встроенный в саму
    # faster-whisper механизм именно под этот класс галлюцинаций: если
    # внутри куска есть подозрительно долгая тишина без слов, а модель
    # всё равно что-то "услышала" по бокам от неё — считает это признаком
    # галлюцинации и обрезает, а не просто доверяет тексту.
    segments, _ = _model.transcribe(
        audio,
        language="ru",
        vad_filter=True,
        condition_on_previous_text=False,
        word_timestamps=True,
        hallucination_silence_threshold=2.0,
    )
    parts = []
    for seg in segments:
        # webrtcvad на входе иногда пропускает фоновый шум/музыку как "речь"
        # (VAD различает только наличие звука, а не то, что это слова) — на
        # таком куске Whisper не столько ошибается, сколько галлюцинирует:
        # уверенно договаривает связный, но никем не произнесённый текст.
        # Признак именно этого случая, а не плохой дикции: модель сама
        # считает, что это скорее не речь (no_speech_prob), и одновременно
        # не уверена в буквах итогового текста (avg_logprob); либо текст
        # зациклился на повторе (compression_ratio) — оба паттерна типичны
        # для галлюцинаций Whisper на тишине/шуме, а не для реальной фразы.
        if seg.no_speech_prob > 0.6 and seg.avg_logprob < -1.0:
            continue
        if seg.compression_ratio > 2.4:
            continue
        if _HALLUCINATION_RE.search(seg.text) or _REPEATED_WORD_RE.search(seg.text):
            continue
        parts.append(seg.text)
    return " ".join(parts).strip()
