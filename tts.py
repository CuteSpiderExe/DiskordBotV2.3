"""TTS-слой: два движка на выбор через config.TTS_ENGINE.

f5     — F5-TTS с клонированным голосом (voice_ref.wav/.txt), GPU, живее.
silero — старый движок (полностью локально, CPU), оставлен как быстрый
         откат через .env, если f5 не устроит или GPU будет занят/недоступен.

Единый интерфейс наружу — synthesize_pcm(text), всегда отдаёт сырой PCM
48kHz stereo 16-bit (готовый для discord.PCMAudio), независимо от движка.
"""

import re
from pathlib import Path
from xml.sax.saxutils import escape as _xml_escape

import numpy as np
import torch
import torchaudio
from num2words import num2words

import config

# Эмодзи и markdown-мусор, который иногда проскакивает из LLM: TTS их не
# озвучивает осмысленно (тишина/спотыкание), поэтому чистим перед синтезом.
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F1E6-\U0001F1FF"
    "\U00002190-\U000021FF"
    "\U00002B00-\U00002BFF"
    "\U0000FE0F"
    "]+",
    flags=re.UNICODE,
)
_MARKDOWN_RE = re.compile(r"[*_~`#]+")
# Многоточия и повторы знаков препинания ("Ой-ой-ой...", "Что?!") — обычная
# разговорная манера персонажа, но движки трактуют каждый знак буквально и
# на таких местах спотыкаются/сбивают паузы. Схлопываем в одиночные знаки —
# смысл и интонация не теряются, а произношение становится ровнее.
_ELLIPSIS_RE = re.compile(r"\.{2,}|…+")
_REPEAT_PUNCT_RE = re.compile(r"([!?])\1+")
# Нарезка на предложения в llm.py иногда выделяет кусок из одних знаков
# препинания (например "..." после закрывающей кавычки предыдущего
# предложения) — синтез текста без единой буквы/цифры падает или молчит.
# Отсекаем такое здесь же, как и пустой текст.
_HAS_LETTER_RE = re.compile(r"\w", re.UNICODE)

# Ни Silero, ни F5 не превращают цифры в слова сами — голая "1" в тексте
# читается моделью как попало (наблюдали: "1 минута" озвучилось как "Э
# минута"). Склонение по падежам не учитываем (это уже сложнее и не стоит
# того здесь) — род по следующему слову угадываем эвристикой, слово всё
# равно понятнее голой цифры, даже если падеж не идеальный.
_NUMBER_RE = re.compile(r"\d+")
_FEMININE_UNIT_RE = re.compile(r"^(минут|секунд|недел|штук)", re.IGNORECASE)


def _numbers_to_words(text: str) -> str:
    def _repl(m: re.Match) -> str:
        tail = text[m.end() : m.end() + 15].strip()
        is_feminine = bool(_FEMININE_UNIT_RE.match(tail))
        try:
            # num2words падает с KeyError на gender=None — параметр либо
            # опущен вовсе (маскулинная форма по умолчанию), либо строка.
            if is_feminine:
                return num2words(int(m.group()), lang="ru", gender="feminine")
            return num2words(int(m.group()), lang="ru")
        except Exception:
            return m.group()

    return _NUMBER_RE.sub(_repl, text)


def _clean_for_tts(text: str) -> str:
    text = _EMOJI_RE.sub("", text)
    text = _MARKDOWN_RE.sub("", text)
    text = _ELLIPSIS_RE.sub(",", text)
    text = _REPEAT_PUNCT_RE.sub(r"\1", text)
    text = _numbers_to_words(text)
    return re.sub(r"\s+", " ", text).strip()


# Модель обрывает волну на конце ровно там, где кончился текст — не на
# нуле амплитуды, а посреди звука. Без затухания это слышно как резкий
# щелчок/обрыв, будто фразу оборвали, а не как естественное окончание речи.
_FADE_OUT_MS = 30


def _to_stereo48k_int16(mono_float: np.ndarray, sample_rate: int) -> bytes:
    """Моно float [-1..1] на произвольной частоте -> формат, который ждёт
    Discord: 48kHz stereo 16-bit PCM."""
    if sample_rate != config.DISCORD_SAMPLE_RATE:
        t = torch.from_numpy(mono_float).unsqueeze(0)
        t = torchaudio.functional.resample(t, sample_rate, config.DISCORD_SAMPLE_RATE)
        mono_float = t.squeeze(0).numpy()
    fade_len = min(len(mono_float), int(config.DISCORD_SAMPLE_RATE * _FADE_OUT_MS / 1000))
    if fade_len > 0:
        mono_float = mono_float.copy()
        mono_float[-fade_len:] *= np.linspace(1.0, 0.0, fade_len, dtype=mono_float.dtype)
    mono_int16 = (mono_float * 32767.0).clip(-32768, 32767).astype(np.int16)
    stereo = np.repeat(mono_int16[:, None], 2, axis=1)  # дублируем канал: mono -> stereo
    return stereo.tobytes()


if config.TTS_ENGINE == "f5":
    import f5_tts.infer.utils_infer as _f5_utils_infer
    from f5_tts.api import F5TTS
    from ruaccent import RUAccent

    # Библиотека печатает ref_text/gen_text через голый print() внутри своего
    # модуля напрямую (не через параметр show_info=, который перекрывает
    # только остальные её сообщения) — те две строки лога только загромождали
    # бы bot.log на каждую фразу. Глушим точечно: подменяем print только в
    # неймспейсе этого модуля, на остальной процесс не влияет.
    _f5_utils_infer.print = lambda *a, **k: None

    _f5_model = F5TTS(
        model="F5TTS_v1_Base",
        ckpt_file=config.F5_CKPT_FILE,
        vocab_file=config.F5_VOCAB_FILE,
        device="cuda",
    )
    _f5_ref_text = Path(config.F5_REF_TEXT_FILE).read_text(encoding="utf-8").strip()

    # У F5-TTS (в отличие от Silero) нет своей автоматической расстановки
    # ударений — чекпоинт Misha24-10/F5-TTS_RUSSIAN понимает только явный "+"
    # перед ударной гласной (см. его README) и без этого почти всегда ставит
    # ударение наугад. RUAccent — ровно то, что для этого рекомендуют:
    # ~5-9мс на фразу на прогретой модели, не влияет на задержку до первого
    # слова на фоне самого синтеза (0.8-2с).
    _accentizer = RUAccent()
    _accentizer.load(omograph_model_size="turbo3.1", use_dictionary=True, tiny_mode=False)

    def _accented(text: str) -> str:
        # На неологизмах/сленге (которых у этого персонажа хватает — вроде
        # "размьючиваться") внутренняя ONNX-модель RUAccent иногда падает
        # с ValueError про token_type_ids (баг самой библиотеки на
        # незнакомых словах). Ударения — это улучшение, а не обязательное
        # условие речи: при сбое просто озвучиваем без расстановки, вместо
        # того чтобы ронять всю фразу целиком.
        try:
            return _accentizer.process_all(text)
        except Exception as e:
            print(f"[tts] RUAccent не справился ({e!r}) — озвучиваю без ударений.")
            return text

    def _synthesize(text: str) -> bytes:
        wav, sr, _ = _f5_model.infer(
            ref_file=config.F5_REF_AUDIO,
            ref_text=_f5_ref_text,
            gen_text=_accented(text),
            speed=config.F5_SPEED,
            show_info=lambda *a, **k: None,  # иначе печатает ref_text/gen_text в лог на каждую фразу
            progress=None,  # иначе на каждую фразу лишний прогресс-бар в лог
        )
        return _to_stereo48k_int16(wav, sr)

else:
    torch.set_num_threads(config.SILERO_CPU_THREADS)
    from silero import silero_tts

    _silero_model, _ = silero_tts(language="ru", speaker="v5_ru")
    _silero_model.to(torch.device("cpu"))

    # Без явного <break> Silero почти не делает пауз на знаках препинания —
    # читает ровным потоком, "пустовато". SSML проверен эмпирически: реально
    # удлиняет паузу (не просто косметика в тексте, которую движок игнорирует).
    _PAUSE_RE = re.compile(r"([,;:])|([.!?]+)")

    def _to_ssml(text: str) -> str:
        def _repl(m: re.Match) -> str:
            if m.group(1):
                return m.group(1) + '<break time="180ms"/>'
            return m.group(2) + '<break time="350ms"/>'

        body = _PAUSE_RE.sub(_repl, _xml_escape(text))
        return f"<speak><p>{body}</p></speak>"

    def _synthesize(text: str) -> bytes:
        audio = _silero_model.apply_tts(
            ssml_text=_to_ssml(text),
            speaker=config.SILERO_SPEAKER,
            sample_rate=config.SILERO_SAMPLE_RATE,
        )  # float32 tensor, mono, [-1..1]
        return _to_stereo48k_int16(audio.numpy(), config.SILERO_SAMPLE_RATE)


def synthesize_pcm(text: str) -> bytes:
    """Текст -> сырой PCM 48kHz stereo 16-bit (готовый для discord.PCMAudio)."""
    text = _clean_for_tts(text)
    if not text or not _HAS_LETTER_RE.search(text):
        return b""
    return _synthesize(text)
