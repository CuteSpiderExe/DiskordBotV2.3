"""LLM-слой: Ollama со стримингом, персонаж, нарезка потока на предложения."""

import re
from functools import wraps
import request_control
from pathlib import Path
from typing import Generator

import ollama

import config
import discord_tools
import memory
from voice_style import strip_markers, strip_tool_markup, strip_html_markup, TOOL_MARKUP_RE

_client = ollama.Client(host=config.OLLAMA_HOST, timeout=config.OLLAMA_TIMEOUT_SEC)

# Промпт (character.txt) просит не рассказывать о себе в третьем лице
# ("Сказала: ..."), но LLM это соблюдает не всегда (та же вероятностная
# природа, что и у надёжности tool calling) — подчищаем детерминированно,
# а не полагаемся только на послушание модели.
_NARRATION_RE = re.compile(
    r"^(?:Сказала|Ответила|Произнесла|Прошептала|Воскликнула|Спросила|Пробормотала|Добавила)\s*:\s*",
    re.IGNORECASE,
)
_QUOTE_CHARS_RE = re.compile(r'["\'«»“”]')
_IDLE_TOOL_MARKER_RE = re.compile(
    r"\(?\s*я\s+вызываю\s+(?:инструмент|функцию)\s*\)?|(?:\$?\{)?function_call\s*\(",
    re.IGNORECASE,
)


def _sanitize(text: str) -> str:
    text = strip_html_markup(text)
    text = strip_tool_markup(text)
    text = _NARRATION_RE.sub("", text)
    text = _QUOTE_CHARS_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def _sanitize_idle_chatter(text: str) -> str:
    text = _sanitize(text)
    marker = _IDLE_TOOL_MARKER_RE.search(text)
    if marker:
        text = text[: marker.start()].rstrip(" ,;:-")
    return text or "М-да, тишина тут звенит громче музыки."

# --- Персонаж ---


def _load_persona() -> str:
    path = Path(config.CHARACTER_FILE)
    if not path.exists():
        return (
            "Ты дружелюбный голосовой ассистент. Отвечай кратко, разговорным "
            "языком, без markdown, твой текст озвучивается вслух."
        )
    return path.read_text(encoding="utf-8").strip()


PERSONA = _load_persona()
VOICE_SUFFIX = (
    "\n\nВАЖНО: ответ будет озвучен голосом — пиши как для устной речи, без "
    "markdown, эмодзи и списков. На простой вопрос или короткую реплику отвечай "
    "одним-двумя короткими предложениями, всего до 25 слов. "
    "Предложения делай короткими — обычно 8-12 слов. "
    "Сразу отвечай по сути, без вступлений, повторов и лишних вопросов. "
    "Не добавляй шутку или встречный вопрос, если они не нужны. "
    "Не используй при обращении к собеседнику парные варианты вроде «сам/сама»; "
    "переформулируй вопрос нейтрально. "
    "Не дописывай к законченному ответу необязательные приложения, пояснения "
    "или новый вопрос. "
    "Если собеседник явно просит подробности, ответь максимум четырьмя "
    "короткими предложениями. Всегда заканчивай мысль полностью: не обрывай "
    "ответ ради краткости и не начинай новую мысль в конце."
)
if getattr(config, "TTS_ENGINE", "") == "qwen":
    VOICE_SUFFIX += (
        "\nПодача голоса: обычно пиши просто текст. При явной просьбе шептать "
        "или если подача уместна по контексту, поставь перед нужным предложением "
        "одну скрытую пометку: [whisper] — шёпот, [amused] — игриво со смешком, "
        "[sigh] — вздох, [excited] — радостно, [sad] — грустно. "
        "Не объясняй эти пометки и не добавляй их без повода. "
        "Не используй другие сценические ремарки."
    )

# Без этого модель периодически просто ПИШЕТ, что выполнила действие (закрепила
# сообщение, поставила статус), не вызывая инструмент по-настоящему — на
# тестах это давало ~17% промахов, с этой инструкцией промахов не было.
TOOLS_REMINDER = (
    "\n\nУ тебя есть инструменты (функции) для реальных действий в Discord. "
    "Если просьба совпадает с одним из них — ты ОБЯЗАНА вызвать этот "
    "инструмент по-настоящему (tool call), а не просто написать в ответе, что "
    "сделала это. Без настоящего вызова действие не произойдёт на самом деле. "
    "Никогда не печатай function_call, JSON вызова или аргументы в обычном тексте. "
    "Для звуковой панели используй play_soundboard_sound с названием доступного звука; "
    "инструмента play_sound нет."
)
FOCUS_REMINDER = (
    "\n\nДЕРЖИ ТЕМУ: сначала ответь прямо на последнее сообщение собеседника и "
    "не уходи от его вопроса. Не начинай новую тему и не добавляй длинные "
    "случайные отступления. Шутка или подкол допустимы только после ответа "
    "и только если они не мешают смыслу. Если контекста не хватает, задай "
    "один короткий уточняющий вопрос по той же теме."
)
FACTS_REMINDER = (
    "\n\nТОЧНОСТЬ: не выдумывай факты, события, цитаты, источники, возможности "
    "или результаты действий. Если не знаешь или не уверена — прямо скажи об "
    "этом, не заполняй пробелы догадками. Чётко отделяй известное из контекста "
    "от предположений; о текущих событиях не утверждай без доступной проверки. "
    "Результат действия считай подтверждённым только после ответа инструмента."
)


def _opts(**extra) -> dict:
    o = {
        "num_thread": config.OLLAMA_NUM_THREAD,
        "num_ctx": config.OLLAMA_NUM_CTX,
        "num_predict": config.LLM_MAX_TOKENS,
        "temperature": 0.35,
        "top_p": 0.85,
        "top_k": 40,
        # Абилитированная модель на коротком голосовом контексте склонна залипать
        # на одной и той же фразе из истории — сбиваем это репитейшн- и
        # presence-штрафами, а не только дефолтным repeat_penalty.
        "repeat_penalty": 1.3,
        "repeat_last_n": 128,
        "presence_penalty": 0.5,
        "frequency_penalty": 0.3,
    }
    o.update(extra)
    return o


def _system_prompt(key: str, voice: bool, status: str = "", name: str = "") -> str:
    if name:
        memory.set_name(key, name)
    system = PERSONA + (VOICE_SUFFIX if voice else "") + TOOLS_REMINDER + FOCUS_REMINDER + FACTS_REMINDER
    summary = memory.entry(key)["summary"]
    if summary:
        system += f"\n\nЧто ты помнишь об этом собеседнике: {summary}"
    if status:
        # Это просто фон для ориентировки, не повод для комментариев самой
        # по себе — упоминай только если к слову пришлось или спросили.
        system += f"\n\nТекущий статус собеседника в голосовом канале: {status}."
    if voice and name:
        # Инструменты вроде mute_member/move_member принимают имя участника
        # аргументом — без этой строки модели не от чего оттолкнуться,
        # чтобы понять, кого имеют в виду под "меня"/"себя".
        system += f"\n\nС тобой сейчас говорит: {name}. Если просят про «меня»/«себя» — это она/он."
    if voice:
        channels = discord_tools.list_voice_channels()
        if channels:
            # move_member ищет канал по названию — если модель подставит
            # реально существующее имя, а не придуманное по слуху, найдётся
            # с первой попытки, без нечёткого поиска-подстраховки.
            system += f"\n\nГолосовые каналы на сервере: {', '.join(channels)}."
        sounds = discord_tools.list_soundboard_sounds()
        if sounds:
            # Без этого модель знает только, что play_soundboard_sound
            # существует, но не какие звуки реально есть на панели —
            # угадывала бы название вслепую.
            system += f"\n\nЗвуки на звуковой панели сервера: {', '.join(sounds)}."
    return system


# --- Сжатие старой истории (вызывается в фоне) ---


def compress_history(key: str):
    if not memory.needs_compression(key):
        return
    old = memory.old_messages(key)
    if not old:
        return
    prompt = (
        "Сожми диалог ниже в краткую заметку (3-5 предложений): кто собеседник, "
        "что обсуждали, что важно помнить. Третье лицо, только факты.\n\n"
        + "\n".join(f"{m['role']}: {m['content']}" for m in old)
    )
    try:
        resp = _client.chat(
            model=config.OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options=_opts(num_predict=200),
            keep_alive=config.OLLAMA_KEEP_ALIVE,
            think=config.LLM_THINK,
        )
        memory.commit_compression(key, old, resp["message"]["content"].strip())
    except request_control.RequestCancelled:
        raise
    except Exception as e:
        print(f"[llm] Не удалось сжать память: {e}")
    memory.save()


# --- Обычный (нестриминговый) ответ — для текстового чата ---


def _conversation(fn):
    @wraps(fn)
    def wrapped(key, *args, **kwargs):
        lock = memory.conversation_lock(key)
        while not lock.acquire(timeout=0.1):
            request_control.check()
        try:
            request_control.check()
            return fn(key, *args, **kwargs)
        finally:
            lock.release()
    return wrapped


@_conversation
def ask(key: str, user_text: str, voice: bool = False, status: str = "", name: str = "") -> str:
    memory.append(key, "user", user_text)
    messages = [
        {"role": "system", "content": _system_prompt(key, voice, status, name)}
    ] + memory.entry(key)["recent"]

    answer = ""
    for _ in range(discord_tools.MAX_TOOL_ROUNDS):
        response = _client.chat(
            model=config.OLLAMA_MODEL,
            messages=messages,
            tools=discord_tools.TEXT_TOOLS,
            options=_opts(),
            keep_alive=config.OLLAMA_KEEP_ALIVE,
            think=config.LLM_THINK,
        )
        msg = response["message"]
        answer = (msg.get("content") or "").strip()
        calls = msg.get("tool_calls")
        if not calls:
            break
        # Модель попросила вызвать инструмент (закрепить сообщение, поставить
        # реакцию и т.п.) — выполняем и отдаём результат обратно, чтобы она
        # сформулировала финальный текстовый ответ уже с учётом сделанного.
        messages.append(msg)
        for call in calls:
            fn_name = call["function"]["name"]
            args = call["function"]["arguments"] or {}
            request_control.check()
            result = discord_tools.dispatch(fn_name, args)
            print(f"[tools] {fn_name}({args}) -> {result}")
            messages.append({"role": "tool", "tool_name": fn_name, "content": str(result)})

    answer = _sanitize(answer)
    memory.append(key, "assistant", answer)
    compress_history(key)
    memory.save()
    return answer


# --- Стриминговый ответ, нарезанный на предложения — для голоса ---

_SENTENCE_END = re.compile(r"([.!?…]+[\s\"»)]*)")
# Реплики короче этого клеятся со следующим предложением перед синтезом —
# см. комментарий в stream_sentences().
_MIN_TTS_CHARS = 15


def stream_sentences(key, user_text, status="", name=""):
    lock = memory.conversation_lock(key)
    while not lock.acquire(timeout=0.1):
        request_control.check()
    try:
        request_control.check()
        yield from _stream_sentences(key, user_text, status, name)
    finally:
        lock.release()


def _active_chunks(stream):
    try:
        for chunk in stream:
            request_control.check()
            yield chunk
        request_control.check()
    finally:
        close = getattr(stream, "close", None)
        if close:
            close()


def _stream_sentences(
    key: str, user_text: str, status: str = "", name: str = ""
) -> Generator[str, None, None]:
    """
    Стримит ответ LLM и выдаёт его по законченным предложениям,
    чтобы TTS мог начинать говорить, не дожидаясь всего ответа.
    В конце сохраняет полный ответ в память.
    """
    memory.append(key, "user", user_text)
    messages = [
        {"role": "system", "content": _system_prompt(key, True, status, name)}
    ] + memory.entry(key)["recent"]

    final_answer = ""
    try:
        for _ in range(discord_tools.MAX_TOOL_ROUNDS):
            stream = _client.chat(
                model=config.OLLAMA_MODEL,
                messages=messages,
                tools=discord_tools.VOICE_TOOLS,
                options=_opts(),
                keep_alive=config.OLLAMA_KEEP_ALIVE,
                # В текущей версии Ollama streaming с think=True иногда
                # отдаёт только скрытое рассуждение без финального текста;
                # для голосового ответа нужен надёжный поток content.
                think=False,
                stream=True,
            )

            buffer = ""
            round_answer = ""
            tool_calls = []
            # Короткие фрагменты объединяем со следующим предложением,
            # чтобы синтез получал достаточно контекста для естественной речи.
            pending = ""
            technical_tail = False
            for chunk in _active_chunks(stream):
                msg = chunk.get("message", {})
                calls = msg.get("tool_calls")
                if calls:
                    tool_calls.extend(calls)
                piece = msg.get("content", "")
                if not piece:
                    continue
                round_answer += piece
                if technical_tail:
                    continue
                buffer += piece
                if TOOL_MARKUP_RE.search(buffer):
                    buffer = strip_tool_markup(buffer)
                    technical_tail = True

                # Вырезаем из буфера все законченные предложения
                while True:
                    m = _SENTENCE_END.search(buffer)
                    if not m:
                        break
                    sentence = _sanitize(buffer[: m.end()])
                    buffer = buffer[m.end() :]
                    if not sentence:
                        continue
                    pending = f"{pending} {sentence}".strip() if pending else sentence
                    if len(pending) >= _MIN_TTS_CHARS:
                        yield pending
                        pending = ""
            tail = _sanitize(buffer)
            if pending:
                tail = f"{pending} {tail}".strip() if tail else pending
            if tail:
                yield tail

            final_answer = _sanitize(round_answer)
            if not tool_calls:
                break
            # Модель попросила вызвать инструмент (замьютить участника,
            # поставить статус канала и т.п.) — выполняем и уходим на ещё
            # один раунд, чтобы она озвучила результат уже своими словами.
            messages.append({"role": "assistant", "content": round_answer, "tool_calls": tool_calls})
            for call in tool_calls:
                fn_name = call["function"]["name"]
                args = call["function"]["arguments"] or {}
                request_control.check()
                result = discord_tools.dispatch(fn_name, args)
                print(f"[tools] {fn_name}({args}) -> {result}")
                messages.append({"role": "tool", "tool_name": fn_name, "content": str(result)})
    finally:
        request = request_control.current.get()
        if final_answer and not (request and request.cancelled.is_set()):
            memory.append(key, "assistant", strip_markers(final_answer))
            compress_history(key)
            memory.save()


def generate_greeting() -> str:
    """Короткое приветствие при заходе в голосовой канал (без памяти)."""
    messages = [
        {"role": "system", "content": PERSONA + VOICE_SUFFIX},
        {
            "role": "user",
            "content": "Ты только что зашла в голосовой канал Discord. "
            "Поздоровайся одним коротким предложением, в своём стиле.",
        },
    ]
    resp = _client.chat(
        model=config.OLLAMA_MODEL,
        messages=messages,
        options=_opts(temperature=0.7, num_predict=60),
        keep_alive=config.OLLAMA_KEEP_ALIVE,
        think=False,
    )
    return resp["message"]["content"].strip()


def generate_kick_complaint() -> str:
    """Реплика при переподключении после того, как бота выкинули из канала
    (не через ari!leave) — сама зашла обратно и возмущается."""
    messages = [
        {"role": "system", "content": PERSONA + VOICE_SUFFIX},
        {
            "role": "user",
            "content": "Тебя только что взяли и выкинули (кикнули) из голосового "
            "канала прямо посреди разговора, а ты сама зашла обратно. "
            "Возмутись, наезжай на того, кто это сделал — одно-два коротких "
            "предложения в своём стиле.",
        },
    ]
    resp = _client.chat(
        model=config.OLLAMA_MODEL,
        messages=messages,
        options=_opts(temperature=0.8, num_predict=80),
        keep_alive=config.OLLAMA_KEEP_ALIVE,
        think=False,
    )
    return resp["message"]["content"].strip()


def generate_idle_chatter(singing: bool = False) -> str:
    """Спонтанная реплика, когда в канале давно никто не говорил (без памяти,
    как и приветствие — это не ответ конкретному человеку)."""
    messages = [
        {
            "role": "system",
            "content": PERSONA
            + "\n\nДля этой спонтанной реплики инструменты недоступны. Не вызывай и не "
            "упоминай инструменты, не печатай function_call, JSON или код. "
            "Сейчас ты тихонько напеваешь песенку от скуки. Верни только текст "
            "собственной короткой песенки: четыре короткие рифмованные строки, "
            "всего не больше 30 слов. Без вступления, названия, сценических "
            "ремарок и вопросов. Не цитируй существующие песни.",
        },
        {
            "role": "user",
            "content": "Все молчат уже пять минут. Придумай и напой короткую "
            "песенку про тишину в канале в своём стиле.",
        },
    ]
    if not singing:
        messages = [
            {"role": "system", "content": PERSONA + VOICE_SUFFIX +
             "\nИнструменты недоступны. Не пиши вызовы функций, JSON и ремарки. "
             "Верни одну короткую разговорную реплику, без песни."},
            {"role": "user", "content": "В канале две минуты тишины. "
             "Скажи что-нибудь от себя в своём стиле, одно короткое предложение."},
        ]
    resp = _client.chat(
        model=config.OLLAMA_MODEL,
        messages=messages,
        options=_opts(temperature=0.7, num_predict=160 if singing else 60),
        keep_alive=config.OLLAMA_KEEP_ALIVE,
        think=False,
    )
    text = _sanitize_idle_chatter(resp["message"]["content"])
    return "[singing] " + strip_markers(text) if singing else text


def warmup():
    """Прогревает модель в Ollama (грузит веса в память), чтобы первый
    реальный ответ (приветствие при ari!join или первый вопрос) не платил
    за холодный старт. num_predict=1 — грузим модель, не тратясь на генерацию."""
    _client.chat(
        model=config.OLLAMA_MODEL,
        messages=[{"role": "user", "content": "привет"}],
        options=_opts(num_predict=1),
        keep_alive=config.OLLAMA_KEEP_ALIVE,
        think=False,
    )
