"""LLM-слой: Ollama со стримингом, персонаж, нарезка потока на предложения."""

import re
from pathlib import Path
from typing import Generator

import ollama

import config
import discord_tools
import memory

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


def _sanitize(text: str) -> str:
    text = _NARRATION_RE.sub("", text)
    text = _QUOTE_CHARS_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()

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
    "markdown, эмодзи и списков. 1-3 коротких предложения."
)
# Без этого модель периодически просто ПИШЕТ, что выполнила действие (закрепила
# сообщение, поставила статус), не вызывая инструмент по-настоящему — на
# тестах это давало ~17% промахов, с этой инструкцией промахов не было.
TOOLS_REMINDER = (
    "\n\nУ тебя есть инструменты (функции) для реальных действий в Discord. "
    "Если просьба совпадает с одним из них — ты ОБЯЗАНА вызвать этот "
    "инструмент по-настоящему (tool call), а не просто написать в ответе, что "
    "сделала это. Без настоящего вызова действие не произойдёт на самом деле."
)


def _opts(**extra) -> dict:
    o = {
        "num_thread": config.OLLAMA_NUM_THREAD,
        "num_ctx": config.OLLAMA_NUM_CTX,
        "num_predict": config.LLM_MAX_TOKENS,
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
    system = PERSONA + (VOICE_SUFFIX if voice else "") + TOOLS_REMINDER
    summary = memory.entry(key)["summary"]
    if summary:
        system += f"\n\nЧто ты помнишь об этом собеседнике: {summary}"
    others = memory.other_summaries(key)
    if others:
        # Личная история остаётся приватной (recent) — сюда попадают только
        # уже сжатые краткие сводки других собеседников, как фон. Не повод
        # вываливать это самой по себе, упоминай только к месту.
        lines = "\n".join(f"- {n}: {s}" for n, s in others)
        system += (
            "\n\nЧто ты помнишь о других людях в этом канале (используй, только "
            "если к слову пришлось, не пересказывай просто так):\n" + lines
        )
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
    old = memory.pop_old_messages(key)
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
            think=False,
        )
        memory.add_summary(key, resp["message"]["content"].strip())
    except Exception as e:
        print(f"[llm] Не удалось сжать память: {e}")
    memory.save()


# --- Обычный (нестриминговый) ответ — для текстового чата ---


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
            options=_opts(temperature=0.9),
            keep_alive=config.OLLAMA_KEEP_ALIVE,
            think=False,
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


def stream_sentences(
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
                options=_opts(temperature=0.9),
                keep_alive=config.OLLAMA_KEEP_ALIVE,
                think=False,
                stream=True,
            )

            buffer = ""
            round_answer = ""
            tool_calls = []
            for chunk in stream:
                msg = chunk.get("message", {})
                calls = msg.get("tool_calls")
                if calls:
                    tool_calls.extend(calls)
                piece = msg.get("content", "")
                if not piece:
                    continue
                buffer += piece
                round_answer += piece

                # Вырезаем из буфера все законченные предложения
                while True:
                    m = _SENTENCE_END.search(buffer)
                    if not m:
                        break
                    sentence = _sanitize(buffer[: m.end()])
                    buffer = buffer[m.end() :]
                    if sentence:
                        yield sentence
            tail = _sanitize(buffer)
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
                result = discord_tools.dispatch(fn_name, args)
                print(f"[tools] {fn_name}({args}) -> {result}")
                messages.append({"role": "tool", "tool_name": fn_name, "content": str(result)})
    finally:
        if final_answer:
            memory.append(key, "assistant", final_answer)
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
        options=_opts(temperature=1.0, num_predict=60),
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
        options=_opts(temperature=1.1, num_predict=80),
        keep_alive=config.OLLAMA_KEEP_ALIVE,
        think=False,
    )
    return resp["message"]["content"].strip()


def generate_idle_chatter() -> str:
    """Спонтанная реплика, когда в канале давно никто не говорил (без памяти,
    как и приветствие — это не ответ конкретному человеку)."""
    messages = [
        {"role": "system", "content": PERSONA + VOICE_SUFFIX},
        {
            "role": "user",
            "content": "В голосовом канале уже какое-то время тишина — никто "
            "не сказал ни слова. Скажи что-нибудь от себя в своём стиле: "
            "подколи тишину, предложи тему, побурчи что скучно — что угодно, "
            "лишь бы не молчать. Одно короткое предложение.",
        },
    ]
    resp = _client.chat(
        model=config.OLLAMA_MODEL,
        messages=messages,
        options=_opts(temperature=1.1, num_predict=60),
        keep_alive=config.OLLAMA_KEEP_ALIVE,
        think=False,
    )
    return resp["message"]["content"].strip()


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
