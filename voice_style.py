"""Hidden sentence-level delivery cues for CustomVoice."""

import re
import json
import html

_HTML_TAG = re.compile(r"<!--.*?(?:-->|$)|</?[A-Za-z][^>]*(?:>|$)", re.DOTALL)


def strip_html_markup(text: str) -> str:
    # Decode escaped tags too (e.g. &lt;/blockquote&gt;).
    return _HTML_TAG.sub(" ", html.unescape(text)).strip()

STYLE_INSTRUCTIONS = {
    "normal": "Speak naturally and conversationally in Russian, with clear pronunciation and natural pauses.",
    "whisper": "Whisper in Russian throughout, very softly, breathily and conspiratorially.",
    "amused": "Speak in Russian playfully while smiling, with a brief soft natural chuckle.",
    "sigh": "Give a soft audible sigh, then speak in Russian with affectionate resignation.",
    "excited": "Speak in Russian with lively, happy excitement and natural intonation.",
    "sad": "Speak softly in Russian with a subdued, sad tone and natural pauses.",
    "singing": "Sing these Russian lyrics softly and melodically, like casually humming a little song to yourself. Use a simple gentle tune, sustained vowels and a clear musical rhythm, rather than ordinary spoken narration.",
}
_MARKER = re.compile(r"\[(normal|whisper|amused|sigh|excited|sad|singing)\]", re.IGNORECASE)

# Explicit delivery requests, not mentions of a mood in ordinary conversation.
_REACTION_COMMANDS = {
    "amused": r"(?:посмейся|рассмейся|засмейся|хихикни|хохотни|смеяться|посмеяться|рассмеяться)",
    "sigh": r"(?:вздохни|повздыхай|вздохнуть)",
    "whisper": r"(?:пошепчи|пошептать|(?:говори|поговори|говорить|поговорить)\s+ш[её]потом)",
    "excited": r"(?:порадуйся|обрадуйся|говори\s+радостно|поговори\s+радостно)",
    "sad": r"(?:погрусти|говори\s+грустно|поговори\s+грустно)",
    "normal": r"(?:говори\s+(?:обычно|нормально)|обычный\s+голос)",
}
_REACTION_TEXT = {
    "Russian": {"amused": "Ха-ха-ха!", "sigh": "Эх…", "whisper": "Только между нами.",
                "excited": "Ура! Как здорово!", "sad": "Мне немного грустно.", "normal": "Хорошо, говорю обычно."},
    "Japanese": {"amused": "あははは！", "sigh": "はぁ…", "whisper": "ここだけの秘密だよ。",
                 "excited": "やった！うれしい！", "sad": "少し寂しいね。", "normal": "普通に話すね。"},
}


def emotion_request(text: str):
    """Return (delivery, optional text to say) only for explicit requests."""
    text = re.sub(r"^\s*(?:ари|ария|арии)\b[\s,!.:-]*", "", text, flags=re.I)
    text = re.sub(r"\bпожалуйста\b[,]?", "", text, flags=re.I).strip(" ,.!?")
    text = re.sub(r"^(?:можешь|давай|прошу)\s+", "", text, flags=re.I).strip()
    for style, command in _REACTION_COMMANDS.items():
        if re.fullmatch(command, text, re.I):
            return style, None
    match = re.fullmatch(
        r"(?:скажи|произнеси)\s+(ш[её]потом|со\s+смехом|со\s+вздохом|радостно|грустно|обычно)"
        r"[\s,:—-]+(.+)", text, re.I)
    if match:
        styles = {"шепотом": "whisper", "шёпотом": "whisper", "со смехом": "amused",
                  "со вздохом": "sigh", "радостно": "excited", "грустно": "sad", "обычно": "normal"}
        return styles[re.sub(r"\s+", " ", match.group(1).lower())], match.group(2)
    return None


def emotion_reaction(style: str, language: str) -> str:
    return f"[{style}] " + _REACTION_TEXT[language][style]

TOOL_MARKUP_RE = re.compile(
    r"\b(?:function_call|tool_calls?)\b|<\|?(?:tool_call|function_call)\b|"
    r"\(?\s*я\s+(?:решил[аи]?\s+)?(?:вызываю|использую|использовать)\s+инструмент\b",
    re.IGNORECASE,
)


def strip_tool_markup(text: str) -> str:
    """Discard technical calls, including incomplete JSON, from spoken text."""
    match = TOOL_MARKUP_RE.search(text)
    return text[:match.start()].rstrip(" ,;:-({\n") if match else text


def delivery(text: str) -> tuple[str, str | dict]:
    """A cue applies from its position until the next cue, including normal."""
    segments = []
    style, start = "normal", 0
    for match in _MARKER.finditer(text):
        phrase = text[start:match.start()].strip()
        if phrase:
            segments.append({"text": phrase, "style": style})
        style, start = match.group(1).lower(), match.end()
    phrase = text[start:].strip()
    if phrase:
        segments.append({"text": phrase, "style": style})
    styles = {part["style"] for part in segments}
    instruction = ({"segments": segments} if len(styles) > 1
                   else next(iter(styles), "normal"))
    return strip_markers(text), instruction


def speech_language(text: str) -> str:
    return "Japanese" if re.search(r"[\u3040-\u30ff\u3400-\u9fff]", text) else "Russian"


def japanese_only(text: str) -> str:
    """Keep Japanese sentences; reject Russian translations/commentary."""
    # Parenthetical translations can themselves contain sentence punctuation.
    text = re.sub(r"\([^()]*[А-Яа-яЁёA-Za-z][^()]*\)", "", text)
    sentences = re.split(r"(?<=[。！？.!?？])\s*|\n+", text)
    kept = []
    for part in sentences:
        if speech_language(part) != "Japanese":
            continue
        if re.search(r"[А-Яа-яЁё]", part):
            # A translation may be embedded in a Russian sentence. Keep its
            # Japanese spans instead of dropping the whole sentence and audio.
            spans = re.findall(r"[\u3040-\u30ff\u3400-\u9fff]"
                               r"[\u3040-\u30ff\u3400-\u9fff\d\s、。，！？「」『』ー.!?…]*", part)
            cue = list(_MARKER.finditer(part.split(spans[0], 1)[0])) if spans else []
            part = (cue[-1].group() + " " if cue else "") + " ".join(s.strip() for s in spans)
        kept.append(part.strip())
    return " ".join(kept).strip()


def delivery_instruction(style, language="Russian") -> str:
    if isinstance(style, str):
        return STYLE_INSTRUCTIONS.get(style, STYLE_INSTRUCTIONS["normal"]).replace("Russian", language)
    lines = ["Read the supplied Russian text once, in order, as one continuous utterance. "
             "Change delivery only at the following text boundaries. "
             "Quoted fragments identify the text; do not read these directions aloud."]
    for part in style["segments"]:
        instruction = STYLE_INSTRUCTIONS.get(part["style"], STYLE_INSTRUCTIONS["normal"])
        instruction = instruction.replace("throughout, ", "")
        lines.append(f'For this fragment only: {json.dumps(part["text"], ensure_ascii=False)}. {instruction}')
    return "\n".join(lines).replace("Russian", language)


def strip_markers(text: str) -> str:
    return _MARKER.sub("", text).strip()
