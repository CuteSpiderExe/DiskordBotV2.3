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
conversation_languages: dict[int, str] = {}
# guild_id -> фоновая задача, которая болтает, если в канале долго тишина
idle_tasks: dict[int, asyncio.Task] = {}
# guild_id-и, для которых прямо сейчас идёт подключение (защита от гонки
# при одновременных командах ari!join)
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

        print("[voice] Вход в голосовой канал только по команде ari!join.")


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
    llm.update_reply_language(str(uid), text)
    language = llm.get_reply_language(str(uid))
    conversation_languages[guild_id] = language
    # Принятый новый вопрос заменяет устаревшую озвучку, включая синтез.
    await speaker.interrupt()
    if _is_singing_request(text):
        if config.TTS_ENGINE != "qwen":
            await speaker.say("Напевная подача доступна с голосом Qwen.")
            return
        activity = last_human_activity.get(guild_id)
        line = await asyncio.to_thread(llm.generate_idle_chatter, singing=True, language=language)
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
        answer = llm.prepare_voice_answer(str(uid), ' '.join(full).strip())
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
                line = await asyncio.to_thread(llm.generate_idle_chatter, singing=singing,
                    language=conversation_languages.get(guild_id, "Russian"))
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
    """Вход в голосовой канал по явной команде ari!join."""
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
    if member.id != bot.user.id or before.channel is None or after.channel is not None:
        return
    guild_id = member.guild.id
    voice_sessions[guild_id] = False
    pending_utterances.pop(guild_id, None)
    idle_task = idle_tasks.pop(guild_id, None)
    if idle_task:
        idle_task.cancel()
    speaker = speakers.pop(guild_id, None)
    if speaker:
        await speaker.close()
    print("[voice] Отключена от канала. Для входа используй ari!join.")


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
    language = llm.get_reply_language(str(ctx.author.id))
    conversation_languages[guild_id] = language
    try:
        async with lock:
            await ctx.send("Сочиняю песенку…")
            line = await asyncio.to_thread(llm.generate_idle_chatter, singing=True, language=language)
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


@bot.command(name="language", aliases=["lang", "язык"])
async def language_cmd(ctx: commands.Context, language: str = None):
    """Язык голосовых ответов для тебя: ari!language ru или ja."""
    key = str(ctx.author.id)
    names = {"Russian": "русский", "Japanese": "японский"}
    valid = {"ru": "Russian", "русский": "Russian", "russian": "Russian",
             "ja": "Japanese", "jp": "Japanese", "японский": "Japanese", "japanese": "Japanese"}
    if language is None:
        await ctx.send(f"Язык моих голосовых ответов тебе: **{names[llm.get_reply_language(key)]}**. "
                       "Переключение: `ari!language ru` или `ari!language ja`. "
                       "Входящую речь распознаю по-русски.")
        return
    selected = valid.get(language.casefold())
    if selected is None:
        await ctx.send("Доступны: `ari!language ru` (русский), `ari!language ja` (японский).")
        return
    llm.set_reply_language(key, selected)
    if ctx.guild is not None:
        conversation_languages[ctx.guild.id] = selected
    await ctx.send(f"Буду отвечать тебе голосом на языке: **{names[selected]}**. "
                   "Настройка действует до перезапуска; входящая речь — русская.")


@bot.command(name="voice", aliases=["голос"])
async def voice_cmd(ctx: commands.Context, speaker_name: str = None):
    """Показать или сменить голос активного движка."""
    if config.TTS_ENGINE == "qwen":
        valid = {v.lower(): v for v in ("Ono_Anna", "Vivian", "Serena", "Sohee")}
        if speaker_name is None:
            await ctx.send(f"Текущий голос: **{config.QWEN_SPEAKER}** (Qwen). Доступны: {', '.join(valid.values())}")
        elif speaker_name.lower() in valid:
            config.QWEN_SPEAKER = valid[speaker_name.lower()]
            await ctx.send(f"Голос сменён на **{config.QWEN_SPEAKER}** для всех собеседников до перезапуска.")
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
