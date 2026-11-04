"""Cooperative cancellation for blocking voice workers."""

import contextvars
import threading
from contextlib import contextmanager


class RequestCancelled(Exception):
    pass


class Request:
    def __init__(self):
        self.cancelled = threading.Event()
        self.lock = threading.RLock()

    def cancel(self):
        with self.lock:
            self.cancelled.set()


current = contextvars.ContextVar("voice_request", default=None)


def check():
    request = current.get()
    if request is not None and request.cancelled.is_set():
        raise RequestCancelled()


@contextmanager
def guard():
    request = current.get()
    if request is None:
        yield
    else:
        with request.lock:
            check()
            yield
