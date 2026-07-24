"""Персистентная память о собеседниках: последние сообщения + сжатая сводка."""

import json
from pathlib import Path

import config

_store: dict = {}


def load():
    global _store
    path = Path(config.MEMORY_FILE)
    if path.exists():
        try:
            _store = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            _store = {}


def save():
    Path(config.MEMORY_FILE).write_text(
        json.dumps(_store, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def entry(key: str) -> dict:
    return _store.setdefault(str(key), {"summary": "", "recent": [], "name": ""})


def append(key: str, role: str, content: str):
    entry(key)["recent"].append({"role": role, "content": content})


def set_name(key: str, name: str):
    if name:
        entry(key)["name"] = name


def other_summaries(exclude_key: str) -> list[tuple[str, str]]:
    """[(имя, сводка), ...] по всем остальным собеседникам с непустой сводкой —
    фон для "общей" памяти персонажа между разными людьми в чате/канале."""
    result = []
    for key, e in _store.items():
        if key == str(exclude_key):
            continue
        summary = e.get("summary", "")
        if summary:
            result.append((e.get("name") or f"пользователь {key}", summary))
    return result


def needs_compression(key: str) -> bool:
    return len(entry(key)["recent"]) >= config.SUMMARIZE_AFTER


def pop_old_messages(key: str) -> list:
    """Забирает старую часть истории для сжатия, оставляя хвост в recent."""
    e = entry(key)
    old = e["recent"][: -config.SHORT_HISTORY_LEN]
    e["recent"] = e["recent"][-config.SHORT_HISTORY_LEN :]
    return old


def add_summary(key: str, new_summary: str):
    e = entry(key)
    e["summary"] = (e["summary"] + " " + new_summary).strip() if e["summary"] else new_summary


load()
