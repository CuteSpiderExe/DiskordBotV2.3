"""Приём голоса из Discord: VAD, накопление фраз, запуск обработки.

Написано под ветку py-cord fix/voice-rec-2:
- метод write() обязан быть помечен @Filters.container
  (иначе AttributeError: '__sink_listeners__');
- user приходит как discord.Object (у него есть .id);
- data — объект VoiceData, сырой PCM лежит в data.pcm (fallback: data.data).
"""

import asyncio
import audioop
import time
from collections import defaultdict

import discord
import webrtcvad
from discord.sinks import Filters

import config


class VoiceSink(discord.sinks.Sink):
    def __init__(self, loop: asyncio.AbstractEventLoop, on_utterance, is_bot_speaking):
        """
        on_utterance(uid: int, pcm16k_mono: bytes) — корутина, вызывается когда
            пользователь закончил фразу;
        is_bot_speaking() -> bool — пока бот говорит, входящие пакеты вообще
            не обрабатываются (не только не прерывают её — не буферизуются
            и не распознаются впоследствии). Приём и воспроизведение
            одновременно (полный дуплекс) в этой ветке py-cord регулярно
            ломает расшифровку входящего звука при DAVE — та же проблема
            воспроизводится и в других библиотеках (discord.js и т.п.), это
            не баг нашего кода, а незрелая поддержка приёма при E2EE во всей
            экосистеме. Единственная рабочая защита — не слушать, пока сама
            говорит. Плата: настоящий барж-ин (прервать её на полуслове)
            больше не работает, а фразы, сказанные во время её ответа,
            теряются, а не откладываются на потом.
        """
        super().__init__()
        self.loop = loop
        self.on_utterance = on_utterance
        self.is_bot_speaking = is_bot_speaking

        self.vad = webrtcvad.Vad(3)
        self.buffers: dict[int, list[bytes]] = defaultdict(list)
        self.last_voice_ts: dict[int, float] = {}
        self.speaking: dict[int, bool] = defaultdict(bool)
        self._watchdog_task = loop.create_task(self._silence_watchdog())

    # --- Приём пакетов ---

    @Filters.container
    def write(self, data, user):
        try:
            self._handle(data, user)
        except Exception as e:
            print(f"[sink] Ошибка в write(): {e!r}")

    def _handle(self, data, user):
        if user is None:
            return
        uid = int(user.id) if hasattr(user, "id") else int(user)

        now = time.monotonic()
        if self.is_bot_speaking():
            return

        raw = getattr(data, "pcm", None) or getattr(data, "data", None)
        if raw is None:
            return

        # 48kHz stereo -> 16kHz mono (для VAD и Whisper)
        mono = audioop.tomono(raw, 2, 0.5, 0.5)
        mono16k, _ = audioop.ratecv(
            mono, 2, 1, config.DISCORD_SAMPLE_RATE, config.VAD_SAMPLE_RATE, None
        )

        is_speech = self._is_speech(mono16k)

        if is_speech:
            self.buffers[uid].append(mono16k)
            self.last_voice_ts[uid] = now
            self.speaking[uid] = True
        else:
            if self.speaking[uid] and now - self.last_voice_ts.get(uid, now) > config.SILENCE_LIMIT_SEC:
                self._finalize(uid)

    def _finalize(self, uid: int):
        """Закрыть фразу и отдать накопленный буфер на распознавание."""
        self.speaking[uid] = False
        pcm = b"".join(self.buffers.pop(uid, []))
        min_bytes = int(config.VAD_SAMPLE_RATE * config.MIN_UTTERANCE_SEC) * 2
        if len(pcm) >= min_bytes:
            asyncio.run_coroutine_threadsafe(self.on_utterance(uid, pcm), self.loop)

    async def _silence_watchdog(self):
        """Discord иногда просто перестаёт слать пакеты, как только пользователь
        замолчал (без завершающих "тихих" фреймов) — тогда write() больше не
        вызывается вообще, и проверка паузы внутри _handle() никогда не
        срабатывает, буфер зависает до следующей фразы. Этот таймер закрывает
        фразу по паузе независимо от того, приходят новые пакеты или нет."""
        while True:
            await asyncio.sleep(0.1)
            now = time.monotonic()
            for uid in list(self.speaking.keys()):
                if self.speaking[uid] and now - self.last_voice_ts.get(uid, now) > config.SILENCE_LIMIT_SEC:
                    self._finalize(uid)

    def _is_speech(self, mono16k: bytes) -> bool:
        frame_bytes = int(config.VAD_SAMPLE_RATE * config.FRAME_MS / 1000) * 2
        if len(mono16k) < frame_bytes:
            return False
        try:
            return self.vad.is_speech(mono16k[-frame_bytes:], config.VAD_SAMPLE_RATE)
        except Exception:
            return False

    def cleanup(self):
        self._watchdog_task.cancel()
        self.buffers.clear()
