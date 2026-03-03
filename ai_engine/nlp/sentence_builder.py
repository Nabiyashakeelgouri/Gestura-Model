import language_tool_python

tool = language_tool_python.LanguageTool('en-US')

def correct_sentence(words):
    if not words:
        return ""

    # Join words
    raw_sentence = " ".join(words)

    # Basic cleanup
    raw_sentence = raw_sentence.replace("thank_u", "thank you")

    # Capitalize
    raw_sentence = raw_sentence.capitalize()

    # Grammar correction
    matches = tool.check(raw_sentence)
    corrected_sentence = language_tool_python.utils.correct(raw_sentence, matches)

    # Add period if missing
    if not corrected_sentence.endswith("."):
        corrected_sentence += "."

    return corrected_sentence