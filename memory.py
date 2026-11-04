"""Персистентная память о собеседниках: последние сообщения + сжатая сводка."""

import copy
import os
import tempfile
import threading
import json
from pathlib import Path

import config
import request_control

_store: dict = {}
_lock = threading.RLock()
_conversation_locks = {}

def conversation_lock(key):
    with _lock:
        return _conversation_locks.setdefault(str(key), threading.RLock())

def _synchronized(fn):
    def wrapped(*args, **kwargs):
        with request_control.guard(), _lock:
            return fn(*args, **kwargs)
    return wrapped



@_synchronized
def load():
    global _store
    path = Path(config.MEMORY_FILE)
    if path.exists():
        try:
            _store = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            _store = {}


@_synchronized
def save():
    path = Path(config.MEMORY_FILE)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False) as f:
            temporary = f.name
            json.dump(_store, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


@_synchronized
def _entry(key: str) -> dict:
    return _store.setdefault(str(key), {"summary": "", "recent": [], "name": ""})


@_synchronized
def append(key: str, role: str, content: str):
    _entry(key)["recent"].append({"role": role, "content": content})


@_synchronized
def set_name(key: str, name: str):
    if name:
        _entry(key)["name"] = name


@_synchronized
def entry(key: str) -> dict:
    return copy.deepcopy(_entry(key))


@_synchronized
def needs_compression(key: str) -> bool:
    return len(_entry(key)["recent"]) >= config.SUMMARIZE_AFTER


@_synchronized
def old_messages(key: str) -> list:
    recent = entry(key)["recent"]
    keep = max(0, config.SHORT_HISTORY_LEN)
    return recent[:-keep] if keep else recent


@_synchronized
def commit_compression(key: str, old: list, summary: str) -> bool:
    e = _entry(key)
    if not old or not summary or e["recent"][:len(old)] != old:
        return False
    e["summary"] = (e["summary"] + " " + summary).strip()
    del e["recent"][:len(old)]
    return True


load()
