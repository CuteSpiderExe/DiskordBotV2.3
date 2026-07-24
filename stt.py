"""STT-слой: faster-whisper, принимает аудио напрямую из памяти (numpy)."""

import numpy as np
from faster_whisper import WhisperModel

import config

_model = WhisperModel(
    config.WHISPER_MODEL_SIZE,
    device="cpu",
    compute_type="int8",
    cpu_threads=config.WHISPER_CPU_THREADS,
)


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
    segments, _ = _model.transcribe(
        audio, language="ru", vad_filter=True, condition_on_previous_text=False
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
        parts.append(seg.text)
    return " ".join(parts).strip()
