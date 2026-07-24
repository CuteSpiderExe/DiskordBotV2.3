"""Инструменты (function calling) для LLM: реальные действия в Discord,
которые модель может вызвать сама во время ответа (закрепить сообщение,
замьютить участника голосового канала и т.п.) — по правам, выданным боту
при приглашении на сервер.

Функции здесь — обычные Python-функции с Google-style докстрингами: ollama
сама строит из них JSON-схему инструмента (см. ollama._utils.convert_function_to_tool),
поэтому сигнатура и докстринг каждой функции ниже — это ровно то, что видит
модель при выборе, что и с какими аргументами вызвать.

Контекст конкретного вызова (гильдия, канал, сообщение) передаётся не
аргументом функции (модель не должна сама подставлять id канала), а через
contextvar — bind() выставляется в bot.py перед вызовом llm.ask()/
stream_sentences(), а asyncio.to_thread() переносит текущий contextvars.Context
в рабочий поток, где эти функции реально исполняются.
"""

import asyncio
import contextvars
import difflib
from dataclasses import dataclass

import discord


@dataclass
class ToolContext:
    loop: asyncio.AbstractEventLoop
    guild: discord.Guild
    message: discord.Message | None = None
    channel: discord.abc.Messageable | None = None
    voice_channel: discord.VoiceChannel | None = None
    # Discord id того, кто сейчас обращается к боту (см. _resolve_member) —
    # передаётся напрямую, а не через кэш участников, которого у бота нет
    # (см. комментарий в bot.py on_ready про Members Intent).
    speaker_id: int | None = None


_ctx_var: contextvars.ContextVar = contextvars.ContextVar("tool_ctx", default=None)


def bind(ctx: ToolContext) -> None:
    _ctx_var.set(ctx)


def _current() -> ToolContext | None:
    return _ctx_var.get()


def _run(coro):
    """Инструменты вызываются из рабочего потока (llm.py крутится через
    asyncio.to_thread), а discord.py требует своего event loop — выполняем
    корутину в нём и синхронно ждём результат."""
    ctx = _current()
    fut = asyncio.run_coroutine_threadsafe(coro, ctx.loop)
    return fut.result(timeout=15)


_SELF_WORDS = {"", "себя", "меня", "я", "мне", "мной", "сам", "сама", "self", "me", "myself"}


async def _resolve_member(ctx: "ToolContext", query: str) -> discord.Member | None:
    """Ищет участника голосового канала бота по имени.

    Без Members Intent (сознательно не запрошен, см. bot.py on_ready)
    guild.get_member()/channel.members работают только для уже закэшированных
    людей — обращающийся к боту в войсе почти никогда в их числе. Поэтому:
    - самоссылку («замьюти меня») резолвим по speaker_id напрямую — это
      настоящий id из голосового пакета, кэш тут вообще не нужен;
    - поиск по имени идёт по channel.voice_states (id из голосовых событий,
      не из кэша участников), а сам Member при необходимости дотягивается
      через fetch_member() — прямой REST-запрos, который не зависит от intents.
    """
    query_norm = (query or "").strip().lower()
    if query_norm in _SELF_WORDS:
        if ctx.speaker_id is None:
            return None
        try:
            return await ctx.guild.fetch_member(ctx.speaker_id)
        except discord.NotFound:
            return None

    members = []
    for uid in ctx.voice_channel.voice_states:
        member = ctx.guild.get_member(uid)
        if member is None:
            try:
                member = await ctx.guild.fetch_member(uid)
            except discord.NotFound:
                continue
        if not member.bot:
            members.append(member)

    for m in members:
        if query_norm == m.display_name.lower() or query_norm == m.name.lower():
            return m
    for m in members:
        if query_norm in m.display_name.lower() or query_norm in m.name.lower():
            return m
    return None


def _find_channel(guild: discord.Guild, query: str) -> discord.VoiceChannel | None:
    """Ищет голосовой канал по названию: точное/частичное совпадение, а если
    ничего не нашлось — самый похожий по написанию (STT редко распознаёт
    название канала дословно, да и склонение/множественное число в русском
    языке само по себе не совпадёт с точным названием подстрокой)."""
    query_norm = query.strip().lower()
    channels = guild.voice_channels
    for c in channels:
        cl = c.name.lower()
        if query_norm == cl or query_norm in cl or cl in query_norm:
            return c
    names = [c.name for c in channels]
    close = difflib.get_close_matches(query_norm, [n.lower() for n in names], n=1, cutoff=0.4)
    if close:
        return discord.utils.find(lambda c: c.name.lower() == close[0], channels)
    return None


def list_voice_channels() -> list[str]:
    """Названия голосовых каналов сервера — используется в системном
    промпте (llm.py), чтобы модель подставляла в move_member название,
    реально существующее на сервере, а не придуманное по слуху."""
    ctx = _current()
    if ctx is None or ctx.guild is None:
        return []
    return [c.name for c in ctx.guild.voice_channels]


def _find_sound(guild: discord.Guild, query: str):
    """Ищет звук со звуковой панели по названию: точное/частичное совпадение,
    а если ничего не нашлось — самый похожий по написанию (см. _find_channel —
    та же причина: модель не знает точных названий, если их нет в промпте,
    и может слегка ошибиться в формулировке)."""
    query_norm = query.strip().lower()
    sounds = guild.sounds
    for s in sounds:
        sl = s.name.lower()
        if query_norm == sl or query_norm in sl or sl in query_norm:
            return s
    names = [s.name for s in sounds]
    close = difflib.get_close_matches(query_norm, [n.lower() for n in names], n=1, cutoff=0.4)
    if close:
        return discord.utils.find(lambda s: s.name.lower() == close[0], sounds)
    return None


def list_soundboard_sounds() -> list[str]:
    """Названия звуков на звуковой панели сервера — используется в системном
    промпте (llm.py), чтобы модель знала, что реально есть на панели, а не
    просто помнила, что инструмент play_soundboard_sound существует."""
    ctx = _current()
    if ctx is None or ctx.guild is None:
        return []
    return [s.name for s in ctx.guild.sounds]


# ---------- Инструменты голосового канала ----------
# (Отключать участникам микрофон, Перемещать участников, Устанавливать статус
# голосового канала, Использовать звуковую панель) — действуют только на
# канал и участников, где сейчас реально сидит сама бот, не на весь сервер.


def set_voice_channel_status(status: str) -> str:
    """Установить статус (короткую подпись) голосового канала, в котором ты
    сейчас находишься.

    Args:
      status: Текст статуса. Пустая строка очищает статус.

    Returns:
      str: результат действия.
    """
    ctx = _current()
    if ctx is None or ctx.voice_channel is None:
        return "Недоступно: ты сейчас не в голосовом канале."
    _run(ctx.voice_channel.set_status(status or None))
    return f"Статус канала: «{status}»." if status else "Статус канала очищен."


def mute_member(member_name: str, mute: bool = True) -> str:
    """Замьютить или снять мьют с участника голосового канала, в котором ты
    сейчас находишься. Работает только для тех, кто прямо сейчас сидит в
    этом канале вместе с тобой. Если просят замьютить «меня»/«себя» —
    передай ровно это слово в member_name, отдельно резолвить имя не нужно.

    Args:
      member_name: Имя или ник участника (можно неполное совпадение), либо
        «себя»/«меня», если просьба про самого обращающегося.
      mute: True — замьютить, False — снять мьют.

    Returns:
      str: результат действия.
    """
    ctx = _current()
    if ctx is None or ctx.voice_channel is None:
        return "Недоступно: ты сейчас не в голосовом канале."

    async def _do():
        member = await _resolve_member(ctx, member_name)
        if member is not None:
            await member.edit(mute=mute)
        return member

    member = _run(_do())
    if member is None:
        return f"Не нашла в канале участника «{member_name}»."
    return f"{'Замьютила' if mute else 'Сняла мьют с'} {member.display_name}."


def move_member(member_name: str, target_channel_name: str = "") -> str:
    """Переместить участника голосового канала (в котором ты сейчас
    находишься) в другой голосовой канал, либо отключить его от голоса
    вовсе. Действует только на тех, кто сейчас в одном канале с тобой. Если
    просят переместить «меня»/«себя» — передай ровно это слово в
    member_name, отдельно резолвить имя не нужно.

    Args:
      member_name: Имя или ник участника (можно неполное совпадение), либо
        «себя»/«меня», если просьба про самого обращающегося.
      target_channel_name: Название канала, куда переместить (можно неточно —
        подставится ближайший по названию существующий канал). Пустая
        строка — отключить участника от голоса вовсе.

    Returns:
      str: результат действия.
    """
    ctx = _current()
    if ctx is None or ctx.voice_channel is None:
        return "Недоступно: ты сейчас не в голосовом канале."
    target = None
    if target_channel_name.strip():
        target = _find_channel(ctx.guild, target_channel_name)
        if target is None:
            return f"Не нашла голосовой канал «{target_channel_name}»."

    async def _do():
        member = await _resolve_member(ctx, member_name)
        if member is not None:
            await member.move_to(target)
        return member

    member = _run(_do())
    if member is None:
        return f"Не нашла в канале участника «{member_name}»."
    if target:
        return f"Переместила {member.display_name} в {target.name}."
    return f"Отключила {member.display_name} от голоса."


def play_soundboard_sound(sound_name: str) -> str:
    """Проиграть звук со звуковой панели (soundboard) сервера в голосовом
    канале, где ты сейчас находишься.

    Args:
      sound_name: Название звука на звуковой панели (можно неполное
        совпадение).

    Returns:
      str: результат действия.
    """
    ctx = _current()
    if ctx is None or ctx.voice_channel is None:
        return "Недоступно: ты сейчас не в голосовом канале."
    sound = _find_sound(ctx.guild, sound_name)
    if sound is None:
        return f"Не нашла звук «{sound_name}» на звуковой панели."
    _run(ctx.voice_channel.send_soundboard_sound(sound))
    return f"Проиграла «{sound.name}»."


# ---------- Инструменты текстового канала ----------
# (Закреплять сообщения, Отправлять TTS-сообщения, Создавать публичные
# ветки, Создавать реакции, Управлять реакциями, Управлять ветками) —
# действуют на сообщение/канал, в котором к тебе обратились текстом.


def pin_message() -> str:
    """Закрепить в текстовом канале то сообщение, на которое ты сейчас
    отвечаешь.

    Returns:
      str: результат действия.
    """
    ctx = _current()
    if ctx is None or ctx.message is None:
        return "Недоступно: сейчас нет сообщения, которое можно закрепить."
    _run(ctx.message.pin())
    return "Закрепила сообщение."


def send_tts_message(text: str) -> str:
    """Отправить в текущий текстовый канал сообщение, которое Discord
    озвучит клиентам через встроенный TTS (это отдельно от твоего
    собственного голоса в войсе).

    Args:
      text: Текст сообщения.

    Returns:
      str: результат действия.
    """
    ctx = _current()
    if ctx is None or ctx.channel is None:
        return "Недоступно: сейчас нет текстового канала."
    _run(ctx.channel.send(text, tts=True))
    return "Отправила TTS-сообщение."


def create_thread(name: str) -> str:
    """Создать публичную ветку (тред) от сообщения, на которое ты сейчас
    отвечаешь.

    Args:
      name: Название ветки.

    Returns:
      str: результат действия.
    """
    ctx = _current()
    if ctx is None or ctx.message is None:
        return "Недоступно: сейчас нет сообщения, от которого можно создать ветку."
    _run(ctx.message.create_thread(name=name))
    return f"Создала ветку «{name}»."


def add_reaction(emoji: str) -> str:
    """Поставить реакцию-эмодзи на сообщение, на которое ты сейчас
    отвечаешь.

    Args:
      emoji: Юникод-смайлик или название эмодзи сервера.

    Returns:
      str: результат действия.
    """
    ctx = _current()
    if ctx is None or ctx.message is None:
        return "Недоступно: сейчас нет сообщения, на которое можно поставить реакцию."
    query = emoji.strip(":").lower()
    custom = discord.utils.find(lambda e: e.name.lower() == query, ctx.guild.emojis)
    _run(ctx.message.add_reaction(custom or emoji))
    return f"Поставила реакцию {emoji}."


def remove_reaction(emoji: str, member_name: str = "") -> str:
    """Убрать чью-то реакцию с сообщения, на которое ты сейчас отвечаешь
    (свою собственную, если имя не указано).

    Args:
      emoji: Эмодзи, которое нужно убрать.
      member_name: Чью реакцию убрать; пусто — свою собственную.

    Returns:
      str: результат действия.
    """
    ctx = _current()
    if ctx is None or ctx.message is None:
        return "Недоступно: сейчас нет сообщения."
    member = ctx.guild.me
    if member_name.strip():
        found = _find_member(ctx.guild.members, member_name)
        if found is None:
            return f"Не нашла участника «{member_name}»."
        member = found
    _run(ctx.message.remove_reaction(emoji, member))
    return "Убрала реакцию."


def manage_thread(archive: bool, locked: bool = False) -> str:
    """Заархивировать/разархивировать и закрыть/открыть текущую ветку
    (тред) — работает, только если разговор сейчас происходит внутри неё.

    Args:
      archive: True — заархивировать ветку, False — разархивировать.
      locked: True — закрыть ветку (писать смогут только модераторы).

    Returns:
      str: результат действия.
    """
    ctx = _current()
    if ctx is None or not isinstance(ctx.channel, discord.Thread):
        return "Недоступно: сейчас разговор идёт не в ветке."
    _run(ctx.channel.edit(archived=archive, locked=locked))
    return f"{'Заархивировала' if archive else 'Разархивировала'} ветку."


TEXT_TOOLS = [pin_message, send_tts_message, create_thread, add_reaction, remove_reaction, manage_thread]
VOICE_TOOLS = [set_voice_channel_status, mute_member, move_member, play_soundboard_sound]
_REGISTRY = {f.__name__: f for f in TEXT_TOOLS + VOICE_TOOLS}
MAX_TOOL_ROUNDS = 3


def dispatch(name: str, arguments: dict) -> str:
    fn = _REGISTRY.get(name)
    if fn is None:
        return f"Ошибка: неизвестный инструмент «{name}»."
    try:
        return fn(**arguments)
    except Exception as e:
        return f"Ошибка при выполнении «{name}»: {e!r}"
