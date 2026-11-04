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


def delivery_instruction(style) -> str:
    if isinstance(style, str):
        return STYLE_INSTRUCTIONS.get(style, STYLE_INSTRUCTIONS["normal"])
    lines = ["Read the supplied Russian text once, in order, as one continuous utterance. "
             "Change delivery only at the following text boundaries. "
             "Quoted fragments identify the text; do not read these directions aloud."]
    for part in style["segments"]:
        instruction = STYLE_INSTRUCTIONS.get(part["style"], STYLE_INSTRUCTIONS["normal"])
        instruction = instruction.replace("throughout, ", "")
        lines.append(f'For this fragment only: {json.dumps(part["text"], ensure_ascii=False)}. {instruction}')
    return "\n".join(lines)


def strip_markers(text: str) -> str:
    return _MARKER.sub("", text).strip()
