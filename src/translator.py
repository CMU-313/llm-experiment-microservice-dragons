import os
from typing import Tuple

from ollama import Client

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "deepseek-r1:1.5b")

_client = Client(host=OLLAMA_HOST)

# Keep context modest on CPU
_OLLAMA_OPTIONS = {
    "num_ctx": 1024,
}

_LANG_SYSTEM = (
    "You are a language classifier. Detect the language of the input text and "
    "reply only with the English name of that language. If the input is empty, "
    "emojis, or gibberish, reply: Unknown"
)

_TRANS_SYSTEM = (
    "You are a translator. Translate the input text to English. "
    "Only output the translated text, nothing else. "
    "If the text is already English or unintelligible/emoji, return it unchanged."
)


def _safe_chat(system_prompt: str, user_text: str) -> str:
    try:
        resp = _client.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text or ""},
            ],
            options=_OLLAMA_OPTIONS,
        )
        content = (resp.message.content or "").strip()
        return content
    except Exception:
        return ""


def _detect_language(text: str) -> str:
    out = _safe_chat(_LANG_SYSTEM, text)
    if not out:
        return "Unknown"
    # Normalize common prefixes
    lowered = out.lower()
    for prefix in ("language:", "the language is", "detected language:", "output:"):
        if lowered.startswith(prefix):
            out = out[len(prefix) :].strip()
            break
    # Collapse lines
    if "\n" in out:
        out = out.split("\n", 1)[0].strip()
    return out or "Unknown"


def _translate_to_english(text: str) -> str:
    out = _safe_chat(_TRANS_SYSTEM, text)
    # If empty, fall back to original
    return out if out else (text or "")


def translate_content(content: str) -> Tuple[bool, str]:
    # Empty or whitespace-only → treat as non-English, return as-is
    if not content or not content.strip():
        return False, content if content else ""

    lang = _detect_language(content)
    is_english = (lang.strip().lower() == "english")

    if is_english:
        return True, content

    translated = _translate_to_english(content)

    # If the model failed or echoed errors, return original
    if not translated or translated.lower().startswith(("error", "i cannot")):
        return False, content

    # If output equals input, still treat as non-English per API contract
    return False, translated
