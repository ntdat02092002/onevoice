from __future__ import annotations

import re
from collections.abc import Sequence


_WORD_OR_PUNCT = re.compile(r"\w+(?:['’-]\w+)*|[^\w\s]", re.UNICODE)
_CJK_OR_PUNCT = re.compile(r"[^\s]", re.UNICODE)
_NO_SPACE_BEFORE = set(",.!?;:%)]}、。，！？；：）】》")
_NO_SPACE_AFTER = set("([{（【《")


def tokenize_text(text: str, language: str | None) -> tuple[str, ...]:
    normalized = " ".join(text.strip().split())
    if not normalized:
        return ()
    matcher = _CJK_OR_PUNCT if language == "zh" else _WORD_OR_PUNCT
    return tuple(matcher.findall(normalized))


def detokenize(tokens: Sequence[str], language: str | None) -> str:
    if language == "zh":
        return "".join(tokens)
    output = ""
    previous = ""
    for token in tokens:
        if not output:
            output = token
        elif token in _NO_SPACE_BEFORE or previous in _NO_SPACE_AFTER:
            output += token
        else:
            output += " " + token
        previous = token
    return output


def longest_common_prefix(sequences: Sequence[Sequence[str]]) -> tuple[str, ...]:
    if not sequences:
        return ()
    shortest = min(len(item) for item in sequences)
    index = 0
    while index < shortest and all(item[index] == sequences[0][index] for item in sequences[1:]):
        index += 1
    return tuple(sequences[0][:index])


def ends_phrase(tokens: Sequence[str]) -> bool:
    return bool(tokens and tokens[-1] in {".", "!", "?", "。", "！", "？"})


def count_complete_sentences(tokens: Sequence[str]) -> int:
    """Count explicit sentence terminators in an ASR token sequence."""
    terminators = {".", "!", "?", "。", "！", "？"}
    count = 0
    previous_was_terminator = False
    for token in tokens:
        is_terminator = token in terminators
        if is_terminator and not previous_was_terminator:
            count += 1
        previous_was_terminator = is_terminator
    return count
