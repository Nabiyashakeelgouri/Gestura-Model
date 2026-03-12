from typing import List

try:
    import language_tool_python
except Exception:
    language_tool_python = None


_tool = None
_tool_init_failed = False


def _get_tool():
    global _tool, _tool_init_failed
    if _tool_init_failed or language_tool_python is None:
        return None
    if _tool is None:
        try:
            _tool = language_tool_python.LanguageTool("en-US")
        except Exception:
            _tool_init_failed = True
            return None
    return _tool


def correct_sentence(words: List[str]) -> str:
    if not words:
        return ""

    raw_sentence = " ".join(words).strip()
    raw_sentence = raw_sentence.replace("thank_u", "thank you")
    raw_sentence = raw_sentence.replace("are_you", "are you")
    raw_sentence = raw_sentence.replace("i_love_you", "I love you")

    if not raw_sentence:
        return ""

    raw_sentence = raw_sentence[0].upper() + raw_sentence[1:]
    corrected_sentence = raw_sentence

    tool = _get_tool()
    if tool:
        try:
            matches = tool.check(raw_sentence)
            corrected_sentence = language_tool_python.utils.correct(raw_sentence, matches)
        except Exception:
            corrected_sentence = raw_sentence

    if not corrected_sentence.endswith("."):
        corrected_sentence += "."

    return corrected_sentence
