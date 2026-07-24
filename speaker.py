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

    def __init__(self, pcm: bytes):
        self._buf = io.BytesIO(pcm)

    def read(self) -> bytes:
        chunk = self._buf.read(self.FRAME_BYTES)
        if len(chunk) < self.FRAME_BYTES:
            return b""  # конец
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
        if self.vc.is_playing():
            try:
                await asyncio.wait_for(asyncio.to_thread(self.vc.stop), timeout=5.0)
            except asyncio.TimeoutError:
                print("[speaker] vc.stop() завис (>5с) — пропускаю, event loop не блокирован.")

    async def close(self):
        await self.interrupt()
        self._worker_task.cancel()

    async def _worker(self):
        while True:
            sentence = await self.queue.get()
            try:
                pcm = await asyncio.to_thread(tts.synthesize_pcm, sentence)
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
                await done.wait()
            except Exception as e:
                print(f"[speaker] Ошибка синтеза/воспроизведения: {e!r}")
            finally:
                self.speaking = False
                self.queue.task_done()
