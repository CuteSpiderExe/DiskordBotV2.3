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
import threading
from collections import defaultdict, deque

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
            и не распознаются впоследствии). Ограничение одновременного
            воспроизведения и приёма оставлено для текущей ветки py-cord.
            Во время синтеза, пока воспроизведение не началось, приём работает.
        """
        super().__init__()
        self.loop = loop
        self.on_utterance = on_utterance
        self.is_bot_speaking = is_bot_speaking

        # write вызывается потоком Discord, watchdog — из asyncio.
        self._lock = threading.RLock()
        self._closed = False
        self.vads = {}
        self._resample_states = {}
        self._remainders = defaultdict(bytes)
        self._preroll = defaultdict(lambda: deque(maxlen=max(1, 200 // config.FRAME_MS)))
        self._speech_bytes = defaultdict(int)
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
        with self._lock:
            if not self._closed:
                self._handle_locked(data, user)

    def _handle_locked(self, data, user):
        if user is None:
            return
        uid = int(user.id) if hasattr(user, "id") else int(user)

        now = time.monotonic()
        if self.is_bot_speaking():
            self._pause_capture()
            return

        raw = getattr(data, "pcm", None) or getattr(data, "data", None)
        if raw is None:
            return

        # 48kHz stereo -> 16kHz mono (для VAD и Whisper)
        mono = audioop.tomono(raw, 2, 0.5, 0.5)
        mono16k, state = audioop.ratecv(
            mono, 2, 1, config.DISCORD_SAMPLE_RATE, config.VAD_SAMPLE_RATE,
            self._resample_states.get(uid),
        )
        self._resample_states[uid] = state
        self._expire(now)
        frames = self._remainders[uid] + mono16k
        frame_bytes = config.VAD_SAMPLE_RATE * config.FRAME_MS // 1000 * 2
        end = len(frames) // frame_bytes * frame_bytes
        self._remainders[uid] = frames[end:]
        for offset in range(0, end, frame_bytes):
            self._accept_frame(uid, frames[offset:offset + frame_bytes], now)

    def _accept_frame(self, uid, frame, now):
        vad = self.vads.get(uid)
        if vad is None:
            vad = self.vads[uid] = webrtcvad.Vad(3)
        is_speech = vad.is_speech(frame, config.VAD_SAMPLE_RATE)
        if is_speech:
            if not self.speaking[uid]:
                self.buffers[uid].extend(self._preroll[uid])
                self._preroll[uid].clear()
            self.last_voice_ts[uid] = now
            self.speaking[uid] = True
            self._speech_bytes[uid] += len(frame)
        if self.speaking[uid]:
            # Сохраняем паузы и тихие окончания внутри всей фразы.
            self.buffers[uid].append(frame)
        else:
            self._preroll[uid].append(frame)

    def _finalize(self, uid: int):
        """Закрыть фразу и отдать накопленный буфер на распознавание."""
        self.speaking[uid] = False
        pcm = b"".join(self.buffers.pop(uid, []))
        min_bytes = int(config.VAD_SAMPLE_RATE * config.MIN_UTTERANCE_SEC) * 2
        voiced_bytes = self._speech_bytes.pop(uid, 0)
        self.last_voice_ts.pop(uid, None)
        if voiced_bytes >= min_bytes:
            asyncio.run_coroutine_threadsafe(self.on_utterance(uid, pcm), self.loop)

    def _expire(self, now):
        for uid in list(self.last_voice_ts):
            if now - self.last_voice_ts[uid] > config.SILENCE_LIMIT_SEC:
                self._finalize(uid)

    def _pause_capture(self):
        # Не склеиваем речь до и после ответа; уже принятую фразу отдаём STT.
        for uid in list(self.last_voice_ts):
            self._finalize(uid)
        self._preroll.clear()
        self._remainders.clear()
        self._resample_states.clear()
        self.vads.clear()

    async def _silence_watchdog(self):
        """Discord иногда просто перестаёт слать пакеты, как только пользователь
        замолчал (без завершающих "тихих" фреймов) — тогда write() больше не
        вызывается вообще, и проверка паузы внутри _handle() никогда не
        срабатывает, буфер зависает до следующей фразы. Этот таймер закрывает
        фразу по паузе независимо от того, приходят новые пакеты или нет."""
        while True:
            await asyncio.sleep(0.1)
            with self._lock:
                if self._closed:
                    return
                if self.is_bot_speaking():
                    self._pause_capture()
                else:
                    self._expire(time.monotonic())

    def cleanup(self):
        with self._lock:
            self._closed = True
            self.buffers.clear()
            self.last_voice_ts.clear()
            self.speaking.clear()
            self._speech_bytes.clear()
            self._preroll.clear()
            self._remainders.clear()
            self._resample_states.clear()
            self.vads.clear()
        self.loop.call_soon_threadsafe(self._watchdog_task.cancel)
