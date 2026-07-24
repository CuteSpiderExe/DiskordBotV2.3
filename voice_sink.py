"""Приём голоса из Discord: VAD, накопление фраз, запуск обработки.

Написано под ветку py-cord fix/voice-rec-2:
- метод write() обязан быть помечен @Filters.container
  (иначе AttributeError: '__sink_listeners__');
- user приходит как discord.Object (у него есть .id);
- data — объект VoiceData, сырой PCM лежит в data.pcm (fallback: data.data).
"""

import asyncio
import audioop
import os
import time
from collections import defaultdict

import discord
import webrtcvad
from discord.sinks import Filters

import config


class VoiceSink(discord.sinks.Sink):
    # Если пакетов не было вообще ни от кого дольше этого — считаем, что приём
    # голоса молча умер (известный баг DAVE-rekey в этой ветке py-cord: без
    # исключения, без колбэка остановки записи) и просим пересоздать запись.
    # Раньше было 10с — оказалось слишком агрессивно: обычная пауза в
    # разговоре (никто не говорит) — это не то же самое, что "никто вообще не
    # прислал ни пакета", но на практике оба случая выглядят одинаково для
    # этой проверки. Сейчас on_dead() перезапускает весь процесс (см. bot.py) —
    # это заметно дороже, чем было раньше, поэтому ложные срабатывания на
    # простую тишину стали куда чувствительнее.
    RECORDING_DEAD_TIMEOUT_SEC = 25.0
    # Поток конкретного участника иногда умирает отдельно, пока остальные всё
    # ещё слышны — тогда общий таймер выше не срабатывает вообще. Если от
    # уже замеченного участника не было ни одного пакета так долго — тоже
    # просим пересоздать запись (не так агрессивно, как общий таймер: короткая
    # естественная пауза в разговоре — это нормально, а вот столько — уже нет).
    PER_USER_DEAD_TIMEOUT_SEC = 45.0

    def __init__(self, loop: asyncio.AbstractEventLoop, on_utterance, is_bot_speaking, on_dead=None):
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
        on_dead() -> bool — корутина, вызывается, если приём голоса молча умер
            (см. RECORDING_DEAD_TIMEOUT_SEC). Возвращает True, если удалось
            восстановиться на месте, не пересоздавая sink (например,
            переинициализировали DAVE-сессию без выхода из канала) — тогда
            этот же sink продолжает работать и таймер сбрасывается. Любое
            другое значение — sink считается замененным (например, после
            полного переподключения), дальнейшее срабатывание не нужно.
        """
        super().__init__()
        self.loop = loop
        self.on_utterance = on_utterance
        self.is_bot_speaking = is_bot_speaking
        self.on_dead = on_dead

        self.vad = webrtcvad.Vad(3)
        self.buffers: dict[int, list[bytes]] = defaultdict(list)
        self.last_voice_ts: dict[int, float] = {}
        self.speaking: dict[int, bool] = defaultdict(bool)
        self._watchdog_task = loop.create_task(self._silence_watchdog())
        self._last_any_pkt_ts = time.monotonic()
        self._last_pkt_ts: dict[int, float] = {}
        self._dead_fired = False
        self._dead_watchdog_task = loop.create_task(self._dead_watchdog())

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

        # Пока бот говорит — вообще не трогаем входящий звук (см. докстринг
        # __init__). last_any/last_pkt всё равно обновляем, иначе сторож
        # зависшей записи ложно сработает на время каждого её ответа.
        now = time.monotonic()
        self._last_any_pkt_ts = now
        self._last_pkt_ts[uid] = now
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

    async def _dead_watchdog(self):
        """Приём голоса в этой ветке py-cord иногда молча умирает, без единого
        события (ни исключения, ни колбэка остановки записи, см.
        make_after_recording в bot.py) — либо целиком (RECORDING_DEAD_TIMEOUT_SEC
        без единого пакета ни от кого), либо только поток одного конкретного
        участника, пока остальные всё ещё слышны (PER_USER_DEAD_TIMEOUT_SEC).
        В обоих случаях просим пересоздать запись через on_dead(), а не ждём
        молча до ручного перезахода. Срабатывает один раз за жизнь sink —
        после пересоздания трекинг начинается заново на новом экземпляре.

        Пока бот говорит, никто и не обязан ничего слать (все и так молчат
        и слушают её) — это НЕ признак сломанного приёма. Проверку паузы
        пропускаем на время её речи целиком, а сразу после того как она
        замолчала — сбрасываем таймер, вместо того чтобы засчитывать всё
        время её монолога как "тишину в приёме" и тут же дёргать
        переподключение прямо посреди следующей фразы."""
        was_speaking = False
        while True:
            await asyncio.sleep(2)
            now = time.monotonic()
            if self.is_bot_speaking():
                was_speaking = True
                continue
            if was_speaking:
                was_speaking = False
                self._last_any_pkt_ts = now
                for uid in self._last_pkt_ts:
                    self._last_pkt_ts[uid] = now
                continue
            if self._dead_fired or self.on_dead is None:
                continue
            dead = now - self._last_any_pkt_ts > self.RECORDING_DEAD_TIMEOUT_SEC or any(
                now - last_ts > self.PER_USER_DEAD_TIMEOUT_SEC for last_ts in self._last_pkt_ts.values()
            )
            if dead:
                self._dead_fired = True
                # _dead_watchdog сама крутится на self.loop (создана через
                # loop.create_task), поэтому можно ждать on_dead() напрямую,
                # без run_coroutine_threadsafe (тот нужен только из _handle,
                # который выполняется в чужом потоке приёма пакетов).
                try:
                    recovered_in_place = await asyncio.wait_for(self.on_dead(), timeout=45)
                except asyncio.TimeoutError:
                    # on_dead() сам защищён таймаутами на каждый рискованный
                    # вызов, но если корутина внутри застряла в блокирующем
                    # нативном вызове без yield — отмена на уровне asyncio её
                    # не прервёт (наблюдалось: vc.disconnect() завис так, что
                    # даже собственный wait_for внутри него не сработал, хотя
                    # остальной bot.loop продолжал жить). Внешний watchdog.py
                    # по heartbeat это тоже не поймает — другие задачи как ни
                    # в чём не бывало продолжают его обновлять. Единственный
                    # надёжный выход — убить процесс целиком; watchdog.py
                    # увидит, что процесс завершился, и поднимет заново.
                    print(
                        "[sink] on_dead() завис (>45с) — реальный процесс висит, "
                        "остальной bot.loop жив. Убиваю процесс, watchdog.py поднимет заново."
                    )
                    os._exit(1)
                if recovered_in_place:
                    # Восстановились без пересоздания sink (например,
                    # переинициализировали DAVE-сессию на месте) — разрешаем
                    # сторожу сработать снова при следующей проблеме, а не
                    # только один раз за всю жизнь этого экземпляра.
                    self._dead_fired = False
                    self._last_any_pkt_ts = time.monotonic()
                    self._last_pkt_ts.clear()

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
        self._dead_watchdog_task.cancel()
        self.buffers.clear()
