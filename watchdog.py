"""Внешний вотчдог: перезапускает bot.py, если он перестал отвечать (см.
HEARTBEAT_* в config.py) или упал сам.

Нужен из-за того, что некоторые вызовы discord-voice-клиента (play/stop/
start_recording) в экспериментальной ветке py-cord иногда зависают насмерть
на нативном уровне и держат GIL — тогда даже asyncio.wait_for изнутри не
спасает, весь процесс замирает целиком. Единственный надёжный способ
восстановиться — снаружи, отдельным процессом.

Запускать вместо bot.py:  python watchdog.py
"""

import os
import subprocess
import sys
import time

import config

STARTUP_GRACE_SEC = 90  # время на первую загрузку моделей, пока heartbeat ещё не пишется
CHECK_INTERVAL_SEC = 3
BOT_LOG_FILE = os.path.join(config.BASE_DIR, "bot.log")


def _heartbeat_age() -> float | None:
    try:
        mtime = os.path.getmtime(config.HEARTBEAT_FILE)
        return time.time() - mtime
    except OSError:
        return None


def _kill(proc: subprocess.Popen):
    print("[watchdog] Убиваю зависший/упавший процесс бота.")
    try:
        proc.kill()
        proc.wait(timeout=10)
    except Exception as e:
        print(f"[watchdog] Не удалось корректно завершить процесс: {e!r}")


def main():
    python = sys.executable
    bot_script = os.path.join(config.BASE_DIR, "bot.py")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    while True:
        # Heartbeat от прошлого запуска не должен обмануть проверку сразу после старта
        try:
            os.remove(config.HEARTBEAT_FILE)
        except OSError:
            pass

        print(f"[watchdog] Запускаю bot.py... (лог: {BOT_LOG_FILE})")
        # У bot.py свой отдельный лог-файл, а не общий stdout с watchdog.py —
        # два процесса, пишущих в один унаследованный хендл на Windows, иногда
        # рвут записи друг друга (наблюдалось: строка обрывалась на середине
        # слова и больше не продолжалась, хотя оба процесса были живы).
        with open(BOT_LOG_FILE, "a", encoding="utf-8") as bot_log:
            bot_log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} — новый запуск =====\n")
            bot_log.flush()
            proc = subprocess.Popen(
                [python, bot_script], stdout=bot_log, stderr=subprocess.STDOUT, env=env
            )
            start = time.time()
            _watch(proc, start)


def _watch(proc: subprocess.Popen, start: float):
    """Следит за процессом, пока не решит, что пора перезапускать."""
    while True:
        time.sleep(CHECK_INTERVAL_SEC)
        if proc.poll() is not None:
            print(f"[watchdog] bot.py завершился сам (код {proc.returncode}) — перезапускаю.")
            return
        age = _heartbeat_age()
        elapsed = time.time() - start
        if age is None:
            if elapsed > STARTUP_GRACE_SEC:
                print("[watchdog] Не дождался первого heartbeat — перезапускаю.")
                _kill(proc)
                return
            continue
        if age > config.HEARTBEAT_TIMEOUT_SEC:
            print(f"[watchdog] Heartbeat не обновлялся {age:.0f}с — бот завис, перезапускаю.")
            _kill(proc)
            return


if __name__ == "__main__":
    main()
