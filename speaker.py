"""Очередь воспроизведения: предложения синтезируются и проигрываются по
порядку; поддерживается барж-ин (пользователь перебил — всё останавливается).
"""

import asyncio
import io

import discord

import tts


class _RawPCM(discord.AudioSource):
    """AudioSource поверх готового PCM 48kHz stereo 16-bit из памяти."""

    FRAME_BYTES = 3840  # 20мс * 48000Гц * 2 канала * 2 байта
    # Обычно длина синтеза не кратна ровно 20мс — раньше последний неполный
    # кусок (до 20мс реальной речи) просто отбрасывался целиком, обрезая
    # конец фразы. Дополняем тишиной до полного фрейма вместо отбрасывания.
    #
    # В начале — известная особенность голосовых клиентов Discord: первые
    # ~100-200мс звука после тишины в канале слушатели иногда не слышат,
    # decoder/jitter-buffer ещё не успевает "проснуться" (это же периодически
    # заметно и когда говорят реальные люди после паузы). Отдаём немного
    # тишины перед настоящей речью, чтобы под удар попадала не она.
    LEAD_SILENCE_MS = 120
    # Симметричная причина в конце: джиттер-буфер на приёме обычно
    # придерживает последние несколько пакетов про запас (сглаживает сеть) —
    # если поток обрывается резко, эти удержанные пакеты рискуют вообще не
    # воспроизвестись. Хвост тишины даёт буферу время их отпустить.
    TRAIL_SILENCE_MS = 100

    def __init__(self, pcm: bytes):
        lead_silence = b"\x00" * int(48000 * 2 * 2 * self.LEAD_SILENCE_MS / 1000)
        trail_silence = b"\x00" * int(48000 * 2 * 2 * self.TRAIL_SILENCE_MS / 1000)
        self._buf = io.BytesIO(lead_silence + pcm + trail_silence)

    def read(self) -> bytes:
        chunk = self._buf.read(self.FRAME_BYTES)
        if not chunk:
            return b""  # реальный конец потока
        if len(chunk) < self.FRAME_BYTES:
            chunk += b"\x00" * (self.FRAME_BYTES - len(chunk))  # добиваем тишиной, не теряя хвост
        return chunk

    def is_opus(self) -> bool:
        return False


class Speaker:
    """Принимает предложения текстом, говорит их по очереди.

    interrupt() — барж-ин: мгновенно замолкает и чистит очередь.
    """

    def __init__(self, vc, loop: asyncio.AbstractEventLoop):
        self.vc = vc
        self.loop = loop
        self.queue: asyncio.Queue = asyncio.Queue()
        self.speaking = False
        # Пока играет текущая фраза, здесь уже синтезируется следующая (если
        # её текст уже готов) — без этого между фразами возникала пауза
        # ~1-2с (время синтеза), даже когда LLM обгоняла TTS и текст был
        # готов заранее. См. _worker().
        self._prefetch_task: asyncio.Task | None = None
        self._worker_task = loop.create_task(self._worker())

    @property
    def busy(self) -> bool:
        return self.speaking or not self.queue.empty()

    async def say(self, sentence: str):
        await self.queue.put(sentence)

    def say_threadsafe(self, sentence: str):
        asyncio.run_coroutine_threadsafe(self.queue.put(sentence), self.loop)

    async def interrupt(self):
        """Барж-ин: очистить очередь и оборвать текущее воспроизведение.

        vc.stop() в этой экспериментальной ветке py-cord иногда зависает
        насмерть (похоже на тот же баг DAVE-rekey) — вызванный синхронно
        из корутины, он тогда блокирует ВЕСЬ event loop бота целиком, а не
        только голос. Уносим его в отдельный поток и ограничиваем таймаутом,
        чтобы зависший stop() не парализовал бота полностью."""
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
                self.queue.task_done()
            except asyncio.QueueEmpty:
                break
        if self._prefetch_task is not None:
            self._prefetch_task.cancel()
            self._prefetch_task = None
        if self.vc.is_playing():
            try:
                await asyncio.wait_for(asyncio.to_thread(self.vc.stop), timeout=5.0)
            except asyncio.TimeoutError:
                print("[speaker] vc.stop() завис (>5с) — пропускаю, event loop не блокирован.")

    async def close(self):
        await self.interrupt()
        self._worker_task.cancel()

    async def _synthesize(self, sentence: str) -> bytes:
        try:
            return await asyncio.to_thread(tts.synthesize_pcm, sentence)
        finally:
            self.queue.task_done()

    async def _worker(self):
        while True:
            try:
                if self._prefetch_task is not None:
                    # Уже начали синтезировать эту фразу заранее, пока играла
                    # предыдущая (см. конец цикла) — просто дожидаемся
                    # результата, без новой паузы на синтез между фразами.
                    task = self._prefetch_task
                    self._prefetch_task = None
                    try:
                        pcm = await task
                    except asyncio.CancelledError:
                        continue  # оборвано барж-ином (interrupt()) — просто пропускаем
                else:
                    sentence = await self.queue.get()
                    pcm = await self._synthesize(sentence)
            except Exception as e:
                # Раньше эта часть была вне try/except: любая ошибка синтеза
                # (например, из RUAccent или F5-TTS) тихо убивала весь
                # _worker() насмерть — бот продолжал отвечать текстом,
                # а голос замолкал навсегда до перезапуска процесса, без
                # единой строки в логе. Теперь пропускаем только эту фразу.
                print(f"[speaker] Ошибка синтеза: {e!r}")
                self.speaking = False
                continue

            try:
                if not self.vc.is_connected():
                    continue
                done = asyncio.Event()

                def _after(err, _done=done):
                    if err:
                        print(f"[speaker] Ошибка воспроизведения: {err}")
                    self.loop.call_soon_threadsafe(_done.set)

                self.speaking = True
                # vc.play(), как и vc.stop() (см. interrupt()), в этой ветке
                # py-cord иногда зависает насмерть — уносим в поток с
                # таймаутом, чтобы не заморозить весь event loop бота.
                try:
                    await asyncio.wait_for(
                        asyncio.to_thread(self.vc.play, _RawPCM(pcm), after=_after), timeout=5.0
                    )
                except asyncio.TimeoutError:
                    print("[speaker] vc.play() завис (>5с) — пропускаю фразу.")
                    continue

                # Текущая фраза уже играет — если следующая уже в очереди
                # (LLM успела сгенерировать её заранее), начинаем синтезировать
                # её прямо сейчас, параллельно с воспроизведением, а не после.
                try:
                    next_sentence = self.queue.get_nowait()
                except asyncio.QueueEmpty:
                    next_sentence = None
                if next_sentence is not None:
                    self._prefetch_task = asyncio.create_task(self._synthesize(next_sentence))

                await done.wait()
            except Exception as e:
                print(f"[speaker] Ошибка синтеза/воспроизведения: {e!r}")
            finally:
                self.speaking = False
