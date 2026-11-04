"""Clevar — Discord AI-персонаж (текст + голос).

Архитектура:
  config.py     — настройки
  memory.py     — персистентная память о собеседниках
  llm.py        — Ollama (стриминг, персонаж, сжатие истории)
  stt.py        — faster-whisper (распознавание из памяти)
  tts.py        — Qwen CustomVoice / Silero (PCM без ffmpeg)
  speaker.py    — очередь воспроизведения
  voice_sink.py — приём голоса, VAD, детекция фраз

Поток голосового ответа:
  голос -> VAD -> фраза -> Whisper -> полный ответ Ollama ->
  единый синтез TTS -> Discord. Пока бот говорит, входящий звук не обрабатывается (полный
  дуплекс ломает расшифровку DAVE в этой ветке py-cord) — барж-ина нет.
"""

import asyncio
import logging
import os
import re
import time

import discord
from discord.ext import commands

import config
import discord_tools
import llm
import request_control
import stt
import tts
from speaker import Speaker
from voice_sink import VoiceSink

if os.environ.get("BOTV2_DEBUG_LOG"):
    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

intents = discord.Intents.default()
intents.message_content = True
intents.voice_states = True
bot = commands.Bot(command_prefix="ari!", intents=intents)

# guild_id -> Speaker
speakers: dict[int, Speaker] = {}
# guild_id -> должна ли запись продолжаться (для авто-восстановления после сбоя)
voice_sessions: dict[int, bool] = {}
# guild_id -> лок, чтобы две фразы подряд не обрабатывались параллельно (гонка в memory/LLM-стриме)
utterance_locks: dict[int, asyncio.Lock] = {}
# guild_id -> когда последний раз кто-то (пользователь или бот) говорил
last_activity: dict[int, float] = {}
last_human_activity: dict[int, float] = {}
last_idle_song: dict[int, float] = {}
# guild_id -> фоновая задача, которая болтает, если в канале долго тишина
idle_tasks: dict[int, asyncio.Task] = {}
# guild_id-и, для которых прямо сейчас идёт подключение (защита от гонки
# при одновременном ari!join и авто-входе/переподключении после кика)
_joining_guilds: set[int] = set()


def _touch_activity(guild_id: int):
    last_activity[guild_id] = time.monotonic()


async def _vc_call(func, *args, timeout: float = 5.0, label: str = "") -> bool:
    """Вызывает синхронный метод discord.VoiceClient (start_recording,
    stop_recording и т.п.) в отдельном потоке с таймаутом. В этой
    экспериментальной ветке py-cord такие вызовы иногда зависают насмерть
    (похоже на баг DAVE-rekey) — вызванные напрямую из корутины, они тогда
    блокируют весь event loop бота целиком, а не только голос."""
    try:
        await asyncio.wait_for(asyncio.to_thread(func, *args), timeout=timeout)
        return True
    except asyncio.TimeoutError:
        print(f"[voice] {label or getattr(func, '__name__', func)}() завис (>{timeout:.0f}с) — пропускаю.")
        return False


def _voice_status_text(member: discord.Member | None) -> str:
    """Статус мика/звука собеседника — только для фона в контексте LLM,
    сама она про это первой не заговаривает (см. _system_prompt)."""
    if member is None or member.voice is None:
        return ""
    vs = member.voice
    parts = []
    if vs.self_mute:
        parts.append("сам выключил микрофон (self-mute)")
    if vs.mute:
        parts.append("заглушен сервером (server mute)")
    if vs.self_deaf:
        parts.append("сам выключил звук (self-deaf)")
    if vs.deaf:
        parts.append("оглушён сервером (server deaf)")
    if vs.self_video:
        parts.append("включена камера")
    if vs.self_stream:
        parts.append("стримит экран")
    return "; ".join(parts) if parts else "микрофон и звук включены, камеры/стрима нет"


# ---------- Текстовый чат ----------


_warmed_up = False


async def _heartbeat_writer():
    """Пишет текущее время в файл каждые HEARTBEAT_INTERVAL_SEC — watchdog.py
    снаружи процесса следит за этим файлом и перезапускает бота, если запись
    остановилась (значит event loop замер целиком, см. config.HEARTBEAT_FILE)."""
    while True:
        try:
            with open(config.HEARTBEAT_FILE, "w", encoding="utf-8") as f:
                f.write(str(time.time()))
        except Exception:
            pass
        await asyncio.sleep(config.HEARTBEAT_INTERVAL_SEC)


@bot.event
async def on_ready():
    global _warmed_up
    print(f"Бот запущен как {bot.user}")
    if not _warmed_up:
        _warmed_up = True
        asyncio.create_task(_heartbeat_writer())

        async def _warmup():
            for label, warmup in (("tts", tts.warmup), ("llm", llm.warmup)):
                try:
                    await asyncio.to_thread(warmup)
                    print(f"[{label}] Модель прогрета, готова к запросам.")
                except Exception as e:
                    print(f"[{label}] Не удалось прогреть модель: {e}")

        asyncio.create_task(_warmup())

        # on_voice_state_update реагирует только на будущие события захода в
        # канал — если люди уже сидели в голосовом до (пере)запуска бота, без
        # этого он не подключится сам, пока кто-то не выйдет и не зайдёт заново.
        for guild in bot.guilds:
            if guild.voice_client is not None:
                continue
            # channel.members требует кэш участников сервера (Members Intent),
            # которого у нас нет и не должно быть для такого бота — из-за
            # этого сразу после (пере)запуска кэш почти пуст, и
            # channel.members ложно выглядит пустым, даже если люди реально
            # сидят в канале. channel.voice_states — низкий уровень специально
            # для этого случая: строится из voice-состояний (voice_states
            # intent, он есть), не из кэша участников.
            occupied = [
                channel
                for channel in guild.voice_channels
                if channel != guild.afk_channel
                and any(uid != bot.user.id for uid in channel.voice_states)
            ]
            ghost_channel = next(
                (c for c in guild.voice_channels if bot.user.id in c.voice_states), None
            )
            if ghost_channel is not None:
                print(f"[voice] Висит призраком в {ghost_channel.name} после сбоя — переподключаюсь туда же.")
                try:
                    await _join_channel(guild, ghost_channel, announce=False)
                except Exception as e:
                    print(f"[voice] Не удалось переподключиться в {ghost_channel.name}: {e}")
            elif len(occupied) == 1:
                channel = occupied[0]
                print(f"[voice] В {channel.name} уже кто-то есть — захожу сама.")
                try:
                    # Это (пере)запуск процесса, а не событие в разговоре —
                    # никто её не звал, поэтому здесь всегда тихий вход.
                    await _join_channel(guild, channel, announce=False)
                except Exception as e:
                    print(f"[voice] Не удалось самостоятельно зайти в канал: {e}")
            elif len(occupied) > 1:
                # Люди сидят сразу в нескольких каналах, и её самой нигде нет
                # (иначе сработала бы ветка выше) — непонятно, к кому идти, а
                # угадывать (брать просто первый по списку) значит с той же
                # вероятностью прийти не туда. Ждём явного ari!join.
                names = ", ".join(c.name for c in occupied)
                print(f"[voice] Люди сразу в нескольких каналах ({names}) — не угадываю, жду ari!join.")


@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return
    if bot.user in message.mentions:
        content = message.content.replace(f"<@{bot.user.id}>", "").strip()
        if content:
            status = _voice_status_text(message.author if isinstance(message.author, discord.Member) else None)
            discord_tools.bind(
                discord_tools.ToolContext(
                    loop=bot.loop,
                    guild=message.guild,
                    message=message,
                    channel=message.channel,
                    speaker_id=message.author.id,
                )
            )
            async with message.channel.typing():
                answer = await asyncio.to_thread(
                    llm.ask, str(message.author.id), content, False, status, message.author.display_name
                )
            await message.channel.send(answer)
    await bot.process_commands(message)


# ---------- Голосовой пайплайн ----------


# guild_id -> {uid: pcm} — по одной отложенной (самой свежей) фразе на
# каждого, кто говорил, пока бот был занят предыдущей. Так при разговоре
# нескольких людей никто не теряется совсем, но и очередь не растёт
# бесконечно — на человека хранится не более одной, самой свежей фразы.
pending_utterances: dict[int, dict[int, bytes]] = {}


def _is_singing_request(text: str) -> bool:
    # Explicit requests only; don't treat mentions of singing as a command.
    if re.search(r"\b(?:не|нельзя|перестань|хватит)\s+(?:\w+\s+){0,2}(?:пой|спой|напой|петь|спеть|напеть)\b", text, re.I):
        return False
    return bool(re.search(
        r"\b(?:спой|напой)\b|\b(?:можешь|можете|давай|прошу|хочу)\b"
        r"[^.!?]{0,60}\b(?:спеть|напеть|петь|песенку|песню)\b", text, re.I))


async def _process_utterance(guild_id: int, uid: int, pcm16k: bytes):
    speaker = speakers.get(guild_id)
    if speaker is None:
        return
    t0 = time.monotonic()
    text = await asyncio.to_thread(stt.transcribe_pcm16, pcm16k)
    t1 = time.monotonic()
    if not text or len(text) < 2:
        return
    print(f"[{time.strftime('%H:%M:%S')}] [user {uid}] {text}  (STT: {t1 - t0:.1f}с)")

    if config.REQUIRE_WAKE_WORD and (
        config.WAKE_WORD_RE is None or not config.WAKE_WORD_RE.search(text)
    ):
        print(f"[voice] Игнорирую (без обращения по имени): {text}")
        return

    guild = bot.get_guild(guild_id)
    # Принятый новый вопрос заменяет устаревшую озвучку, включая синтез.
    await speaker.interrupt()
    if _is_singing_request(text):
        if config.TTS_ENGINE != "qwen":
            await speaker.say("Напевная подача доступна с голосом Qwen.")
            return
        activity = last_human_activity.get(guild_id)
        line = await asyncio.to_thread(llm.generate_idle_chatter, singing=True)
        request_control.check()
        if (speakers.get(guild_id) is speaker and speaker.vc.is_connected()
                and last_human_activity.get(guild_id) == activity
                and not pending_utterances.get(guild_id)):
            print(f"[song voice] {line}")
            await speaker.say(line)
            last_idle_song[guild_id] = time.monotonic()
        else:
            print("[song discarded] Во время подготовки появилась новая активность.")
        return
    member = guild.get_member(uid) if guild else None
    if member is None and guild is not None:
        # Без Members Intent (сознательно не запрошен) кэш участников почти
        # никогда не знает того, кто только что заговорил в войсе —
        # fetch_member — прямой REST-запрос, от intents не зависит.
        try:
            member = await guild.fetch_member(uid)
        except discord.NotFound:
            member = None
    status = _voice_status_text(member)
    name = member.display_name if member else ""

    discord_tools.bind(
        discord_tools.ToolContext(
            loop=bot.loop, guild=guild, voice_channel=speaker.vc.channel, speaker_id=uid
        )
    )

    def _stream_worker():
        """Собирает ответ целиком; TTS получает один текст после конца стрима."""
        full = []
        for sentence in llm.stream_sentences(str(uid), text, status, name):
            request_control.check()
            full.append(sentence)
        request_control.check()
        answer = ' '.join(full).strip()
        if answer:
            print(f"[timing] Полный ответ LLM: {time.monotonic() - t1:.1f}с")
            print(f"[{time.strftime('%H:%M:%S')}] [bot] {answer}")
            speaker.say_threadsafe(answer)

    await asyncio.to_thread(_stream_worker)


async def handle_utterance(guild_id: int, uid: int, pcm16k: bytes):
    speaker = speakers.get(guild_id)
    if speaker is None:
        return
    _touch_activity(guild_id)
    last_human_activity[guild_id] = time.monotonic()

    # Сериализация: пока не обработана предыдущая фраза (STT+LLM-стрим),
    # следующую не начинаем — иначе конкурентные stream_sentences() гонятся
    # за memory.append() и перемешивают историю (user/user, assistant/assistant).
    # Если в это время говорили другие (в том числе несколько человек сразу) —
    # никто не теряется, каждый получит ответ на свою последнюю фразу, как
    # только освободится очередь (см. pending_utterances ниже).
    lock = utterance_locks.setdefault(guild_id, asyncio.Lock())
    if lock.locked():
        pending_utterances.setdefault(guild_id, {})[uid] = pcm16k
        return

    async with lock:
        await _process_utterance_safe(guild_id, uid, pcm16k)
        _touch_activity(guild_id)
        while True:
            pending = pending_utterances.pop(guild_id, None)
            if not pending:
                break
            for nxt_uid, nxt_pcm in pending.items():
                await _process_utterance_safe(guild_id, nxt_uid, nxt_pcm)
                _touch_activity(guild_id)


async def _process_utterance_safe(guild_id: int, uid: int, pcm16k: bytes):
    """Обёртка с таймаутом: если STT или Ollama зависнут (сеть, нехватка
    VRAM и т.п.) без исключения, лок на гильдию иначе не освободится никогда
    и бот молча перестанет отвечать до перезапуска."""
    request = request_control.Request()
    token = request_control.current.set(request)
    try:
        await asyncio.wait_for(
            _process_utterance(guild_id, uid, pcm16k), timeout=config.OLLAMA_TIMEOUT_SEC + 15
        )
    except asyncio.TimeoutError:
        request.cancel()
        speaker = speakers.get(guild_id)
        if speaker:
            await speaker.interrupt()
        print(f"[voice] Обработка фразы зависла (>{config.OLLAMA_TIMEOUT_SEC + 15:.0f}с), пропускаю.")
    except Exception as e:
        request.cancel()
        print(f"[voice] Ошибка обработки фразы: {e!r}")
    except asyncio.CancelledError:
        request.cancel()
        raise
    finally:
        request_control.current.reset(token)


async def idle_watcher(guild: discord.Guild):
    """Пока в канале тишина дольше IDLE_CHATTER_SEC — бот сам заговаривает
    на случайную тему, как будто ей скучно. Останавливается через ari!leave."""
    guild_id = guild.id
    while voice_sessions.get(guild_id):
        await asyncio.sleep(5)
        vc = guild.voice_client
        speaker = speakers.get(guild_id)
        if not vc or not vc.is_connected() or speaker is None:
            continue
        if speaker.busy:
            continue
        listeners = [uid for uid in vc.channel.voice_states if uid != bot.user.id]
        if not listeners:
            continue
        now = time.monotonic()
        human_at = last_human_activity.get(guild_id, now)
        singing = now - max(human_at, last_idle_song.get(guild_id, human_at)) >= config.IDLE_SONG_SEC
        idle_for = now - last_activity.get(guild_id, now)
        if not singing and idle_for < config.IDLE_CHATTER_SEC:
            continue

        # Та же серилизация, что и у обычных фраз: нельзя гонять два запроса
        # к Ollama параллельно — на CPU это не ускоряет, а топит оба в разы.
        lock = utterance_locks.setdefault(guild_id, asyncio.Lock())
        if lock.locked():
            continue
        try:
            activity = last_activity.get(guild_id)
            async with lock:
                line = await asyncio.to_thread(llm.generate_idle_chatter, singing=singing)
                if (last_activity.get(guild_id) != activity or speaker.busy
                        or pending_utterances.get(guild_id)
                        or not voice_sessions.get(guild_id)):
                    print("[idle discarded] Во время генерации появилась активность.")
                else:
                    print(f"[idle chatter] {line}")
                    await speaker.say(line)
                    if singing:
                        last_idle_song[guild_id] = time.monotonic()
        except Exception as e:
            print(f"[voice] Не удалось сгенерировать спонтанную фразу: {e}")
        # Фраза, пришедшая под локом генерации idle, сама уже вернулась
        # из handle_utterance. После освобождения лока обязательно запускаем её.
        pending = pending_utterances.pop(guild_id, {})
        for uid, pcm in pending.items():
            await handle_utterance(guild_id, uid, pcm)
        _touch_activity(guild_id)


def make_after_recording(guild: discord.Guild):
    """Колбэк остановки записи: перезапускает её после аварийного сбоя
    (известный баг DAVE-rekey в этой ветке py-cord), но не после ari!leave."""

    async def _after(*args):
        exc = next((a for a in args if isinstance(a, BaseException)), None)
        if exc:
            print(f"[voice] Запись прервалась: {exc!r}")
        if not voice_sessions.get(guild.id):
            return
        vc = guild.voice_client
        if vc and vc.is_connected() and not vc.is_recording():
            try:
                if await _vc_call(
                    vc.start_recording, _make_sink(guild), make_after_recording(guild),
                    label="start_recording",
                ):
                    print("[voice] Запись автоматически перезапущена.")
            except Exception as e:
                print(f"[voice] Не удалось перезапустить запись: {e}")

    return _after


def _make_sink(guild: discord.Guild) -> VoiceSink:
    guild_id = guild.id

    async def on_utt(uid: int, pcm: bytes):
        await handle_utterance(guild_id, uid, pcm)

    def bot_speaking() -> bool:
        sp = speakers.get(guild_id)
        return sp.speaking if sp else False

    sink = VoiceSink(bot.loop, on_utt, bot_speaking)
    return sink


async def _join_channel(
    guild: discord.Guild, channel: discord.VoiceChannel, kicked: bool = False, announce: bool = True
):
    """Общий вход в канал: используется и командой ari!join, и авто-входом,
    когда кто-то заходит в канал, и переподключением после кика.

    announce=False — тихий вход без реплики: используется, когда мы сами
    переподключаемся после технического сбоя процесса —
    это не событие в разговоре, никто её не звал и не выгонял, озвучивать
    нечего."""
    if guild.id in _joining_guilds:
        return
    _joining_guilds.add(guild.id)
    try:
        old_speaker = speakers.pop(guild.id, None)
        if old_speaker:
            try:
                await old_speaker.close()
            except Exception:
                pass

        # После реального дисконнекта (кик, разрыв сети) py-cord иногда не
        # успевает сам обнулить guild.voice_client — тогда channel.connect()
        # тут же падает с "Already connected to a voice channel", хотя мы на
        # самом деле уже не подключены. Подчищаем зависший объект перед
        # входом, а не полагаемся на то, что библиотека сама разберётся.
        if guild.voice_client is not None:
            try:
                await asyncio.wait_for(guild.voice_client.disconnect(force=True), timeout=10)
            except Exception as e:
                print(f"[voice] Не удалось закрыть зависшее подключение перед входом: {e!r}")

        # channel.connect() иногда зависает без единого исключения (тот же
        # класс багов, что и play/stop/disconnect в этой ветке py-cord) —
        # наблюдалось: минуты без единого события, heartbeat при этом жив.
        # Ограничиваем таймаутом, чтобы хотя бы получить ошибку в лог и
        # освободить _joining_guilds, вместо тихого зависания навсегда.
        vc = await asyncio.wait_for(channel.connect(), timeout=20)
        speakers[guild.id] = Speaker(vc, bot.loop)

        if announce:
            # Та же серилизация, что и у обычных фраз: greeting генерируется под
            # локом, чтобы не столкнуться лбами с первым же вопросом пользователя
            # (два одновременных запроса к CPU-модели топят друг друга по времени).
            lock = utterance_locks.setdefault(guild.id, asyncio.Lock())
            try:
                async with lock:
                    if kicked:
                        line = await asyncio.to_thread(llm.generate_kick_complaint)
                    else:
                        line = await asyncio.to_thread(llm.generate_greeting)
                print(f"[bot {'kicked-rejoin' if kicked else 'greeting'}] {line}")
                await speakers[guild.id].say(line)
            except Exception as e:
                print(f"[voice] Не удалось сгенерировать реплику при входе: {e}")

        voice_sessions[guild.id] = True
        await _vc_call(
            vc.start_recording, _make_sink(guild), make_after_recording(guild), label="start_recording"
        )
        _touch_activity(guild.id)
        last_human_activity[guild.id] = time.monotonic()
        last_idle_song.pop(guild.id, None)

        old_idle = idle_tasks.pop(guild.id, None)
        if old_idle:
            old_idle.cancel()
        idle_tasks[guild.id] = bot.loop.create_task(idle_watcher(guild))
    finally:
        _joining_guilds.discard(guild.id)


@bot.event
async def on_voice_state_update(
    member: discord.Member, before: discord.VoiceState, after: discord.VoiceState
):
    guild = member.guild

    if member.id == bot.user.id:
        # Бота выкинули из канала не через ari!leave — похоже на кик посреди
        # разговора. Сама переподключается и возмущается.
        if before.channel is not None and after.channel is None and voice_sessions.get(guild.id):
            print(f"[voice] Отключили из {before.channel.name} без ari!leave — переподключаюсь.")
            voice_sessions[guild.id] = False
            idle_task = idle_tasks.pop(guild.id, None)
            if idle_task:
                idle_task.cancel()
            speaker = speakers.pop(guild.id, None)
            if speaker:
                await speaker.close()
            await asyncio.sleep(1.5)
            try:
                await _join_channel(guild, before.channel, kicked=True)
            except Exception as e:
                print(f"[voice] Не удалось переподключиться после кика: {e}")
        return

    if member.bot:
        return

    # Живой человек зашёл в новый голосовой канал — заходим сами, если нас
    # ещё нигде нет в этой гильдии.
    if after.channel is not None and after.channel != before.channel:
        if after.channel == guild.afk_channel:
            return
        if guild.voice_client is not None or guild.id in _joining_guilds:
            return
        print(f"[voice] {member.display_name} зашёл в {after.channel.name} — захожу сама.")
        try:
            await _join_channel(guild, after.channel)
        except Exception as e:
            print(f"[voice] Не удалось самостоятельно зайти в канал: {e}")


# ---------- Команды ----------


@bot.command(name="join")
async def join_cmd(ctx: commands.Context):
    if not ctx.author.voice:
        await ctx.send("Зайди сначала в голосовой канал.")
        return
    if ctx.voice_client:
        await ctx.send("Я уже в голосовом канале.")
        return
    await _join_channel(ctx.guild, ctx.author.voice.channel)


@bot.command(name="leave")
async def leave_cmd(ctx: commands.Context):
    vc = ctx.voice_client
    if not vc:
        await ctx.send("Я не в голосовом канале.")
        return
    voice_sessions[ctx.guild.id] = False
    pending_utterances.pop(ctx.guild.id, None)
    idle_task = idle_tasks.pop(ctx.guild.id, None)
    if idle_task:
        idle_task.cancel()
    speaker = speakers.pop(ctx.guild.id, None)
    if speaker:
        await speaker.close()
    if vc.is_recording():
        await _vc_call(vc.stop_recording, label="stop_recording")
    await vc.disconnect()
    await ctx.send("Отключился.")


@bot.command(name="sing", aliases=["пой"])
async def sing_cmd(ctx: commands.Context):
    """Сочинить и напеть короткую песенку в текущем голосовом канале."""
    if ctx.guild is None:
        await ctx.send("Команда работает в чате сервера.")
        return
    guild_id = ctx.guild.id
    speaker = speakers.get(guild_id)
    if speaker is None or not speaker.vc.is_connected():
        await ctx.send("Сначала позови меня в голосовой канал: ari!join.")
        return
    if config.TTS_ENGINE != "qwen":
        await ctx.send("Напевная подача доступна с голосом Qwen.")
        return
    lock = utterance_locks.setdefault(guild_id, asyncio.Lock())
    if lock.locked() or speaker.busy:
        await ctx.send("Сейчас отвечаю. Попроси спеть, когда закончу.")
        return
    _touch_activity(guild_id)
    last_human_activity[guild_id] = time.monotonic()
    activity = last_activity[guild_id]
    try:
        async with lock:
            await ctx.send("Сочиняю песенку…")
            line = await asyncio.to_thread(llm.generate_idle_chatter, singing=True)
            if (speakers.get(guild_id) is not speaker or not speaker.vc.is_connected()
                    or last_activity.get(guild_id) != activity
                    or pending_utterances.get(guild_id)):
                await ctx.send("Песенку отложила: начался разговор или изменилось подключение.")
                return
            print(f"[song command] {line}")
            await speaker.say(line)
            last_idle_song[guild_id] = time.monotonic()
            _touch_activity(guild_id)
    except Exception as exc:
        print(f"[song command] Ошибка: {exc!r}")
        await ctx.send("Не удалось подготовить песенку. Подробности в логе.")
    finally:
        pending = pending_utterances.pop(guild_id, {})
        for uid, pcm in pending.items():
            await handle_utterance(guild_id, uid, pcm)


@bot.command(name="voice")
async def voice_cmd(ctx: commands.Context, speaker_name: str = None):
    """Показать или сменить голос активного движка."""
    if config.TTS_ENGINE == "qwen":
        valid = {v.lower(): v for v in ("Ono_Anna", "Vivian", "Serena", "Sohee")}
        if speaker_name is None:
            await ctx.send(f"Текущий голос: **{config.QWEN_SPEAKER}** (Qwen). Доступны: {', '.join(valid.values())}")
        elif speaker_name.lower() in valid:
            config.QWEN_SPEAKER = valid[speaker_name.lower()]
            await ctx.send(f"Голос сменён на **{config.QWEN_SPEAKER}**.")
        else:
            await ctx.send(f"Доступны: {', '.join(valid.values())}")
        return
    valid = ("xenia", "baya", "kseniya")
    if speaker_name is None:
        await ctx.send(f"Текущий голос: **{config.SILERO_SPEAKER}**. Доступны: {', '.join(valid)}")
    elif speaker_name in valid:
        config.SILERO_SPEAKER = speaker_name
        await ctx.send(f"Голос сменён на **{speaker_name}**.")
    else:
        await ctx.send(f"Доступны: {', '.join(valid)}")


@bot.command(name="model")
async def model_cmd(ctx: commands.Context, model_name: str = None):
    """Показать или сменить LLM-модель: ari!model gemma4:e2b"""
    if model_name is None:
        await ctx.send(f"Текущая модель: **{config.OLLAMA_MODEL}**")
        return
    config.OLLAMA_MODEL = model_name
    await ctx.send(f"Модель сменена на **{model_name}** (должна быть скачана в Ollama).")


if __name__ == "__main__":
    if not config.DISCORD_TOKEN:
        raise SystemExit("Не задан DISCORD_TOKEN в .env")
    bot.run(config.DISCORD_TOKEN)
