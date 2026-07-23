from __future__ import annotations

import re
from collections.abc import Sequence


_WORD_OR_PUNCT = re.compile(r"\w+(?:['’-]\w+)*|[^\w\s]", re.UNICODE)
_CJK_OR_PUNCT = re.compile(r"[^\s]", re.UNICODE)
_NO_SPACE_BEFORE = set(",.!?;:%)]}、。，！？；：）】》")
_NO_SPACE_AFTER = set("([{（【《")
_TERMINAL_MARKS = {".", "!", "?", "。", "！", "？"}
_CLOSING_MARKS = set("\"'”’)]}）】》")
_NON_TERMINAL_ABBREVIATIONS = {
    "mr",
    "mrs",
    "ms",
    "dr",
    "prof",
    "sr",
    "jr",
    "st",
    "pgs",
    "gs",
    "ts",
    "ths",
    "bs",
    "tp",
}


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


def _previous_word(text: str, period_index: int) -> str:
    start = period_index
    while start > 0 and text[start - 1].isalpha():
        start -= 1
    return text[start:period_index]


def _next_non_space(text: str, index: int) -> str:
    while index < len(text) and text[index].isspace():
        index += 1
    return text[index] if index < len(text) else ""


def _period_is_boundary(text: str, index: int) -> bool:
    previous = text[index - 1] if index else ""
    following = text[index + 1] if index + 1 < len(text) else ""
    if previous.isdigit() and following.isdigit():
        return False
    # Internal acronym/domain-like dots have no whitespace after them:
    # U.S.A., TP.HCM, p.m. and initials such as H.G.
    if following and not following.isspace() and following not in _CLOSING_MARKS:
        return False

    word = _previous_word(text, index)
    next_char = _next_non_space(text, index + 1)
    if word.lower() in _NON_TERMINAL_ABBREVIATIONS and next_char:
        return False

    # For multi-dot abbreviations, only the final dot can terminate, and only
    # at end-of-text or before a likely new sentence. This keeps p.m. and
    # U.S.A. intact while allowing "p.m. We left." to close a sentence.
    prefix = text[: index + 1]
    abbreviation = re.search(r"(?:[A-Za-z]\.){2,}$", prefix)
    if abbreviation:
        return not next_char or next_char.isupper()

    # A single capital followed by another capitalized name is an initial.
    if len(word) == 1 and word.isupper() and next_char.isupper():
        return False

    return True


def _sentence_boundaries(text: str) -> tuple[int, ...]:
    boundaries: list[int] = []
    index = 0
    while index < len(text):
        mark = text[index]
        boundary = mark in {"!", "?", "。", "！", "？"} or (
            mark == "." and _period_is_boundary(text, index)
        )
        if not boundary:
            index += 1
            continue
        end = index + 1
        while end < len(text) and text[end] in _TERMINAL_MARKS:
            end += 1
        while end < len(text) and text[end] in _CLOSING_MARKS:
            end += 1
        boundaries.append(end)
        index = end
    return tuple(boundaries)


def split_sentences(text: str, language: str | None = None) -> tuple[str, ...]:
    """Split on semantic sentence boundaries without breaking common abbreviations."""
    normalized = " ".join(text.strip().split())
    if not normalized:
        return ()
    output: list[str] = []
    start = 0
    for end in _sentence_boundaries(normalized):
        sentence = normalized[start:end].strip()
        if sentence:
            output.append(sentence)
        start = end
    tail = normalized[start:].strip()
    if tail:
        output.append(tail)
    return tuple(output)


def sentence_token_boundaries(text: str, language: str | None) -> tuple[int, ...]:
    """Return cumulative token offsets for complete sentences in text."""
    sentences = split_sentences(text, language)
    offsets: list[int] = []
    count = 0
    for sentence in sentences:
        sentence_tokens = tokenize_text(sentence, language)
        count += len(sentence_tokens)
        if sentence_tokens and ends_phrase(sentence, language):
            offsets.append(count)
    return tuple(offsets)


def ends_phrase(value: str | Sequence[str], language: str | None = None) -> bool:
    text = value if isinstance(value, str) else detokenize(value, language)
    normalized = text.rstrip()
    if not normalized:
        return False
    boundaries = _sentence_boundaries(normalized)
    return bool(boundaries and boundaries[-1] == len(normalized))


def restore_terminal_punctuation(
    source: str, translated: str, source_language: str | None, target_language: str | None
) -> str:
    """Restore a source sentence terminator when an MT model drops it."""
    source_tokens = tokenize_text(source, source_language)
    target_tokens = tokenize_text(translated, target_language)
    source_index = len(source_tokens) - 1
    while source_index >= 0 and source_tokens[source_index] in _CLOSING_MARKS:
        source_index -= 1
    target_index = len(target_tokens) - 1
    while target_index >= 0 and target_tokens[target_index] in _CLOSING_MARKS:
        target_index -= 1
    if source_index < 0 or source_tokens[source_index] not in _TERMINAL_MARKS:
        return translated.strip()
    if target_index >= 0 and target_tokens[target_index] in _TERMINAL_MARKS:
        return translated.strip()
    mark = source_tokens[source_index]
    if target_language == "zh":
        mark = {".": "。", "!": "！", "?": "？"}.get(mark, mark)
    output = translated.rstrip()
    insertion = len(output)
    while insertion > 0 and output[insertion - 1] in _CLOSING_MARKS:
        insertion -= 1
    return f"{output[:insertion]}{mark}{output[insertion:]}"


def count_complete_sentences(
    value: str | Sequence[str], language: str | None = None
) -> int:
    """Count complete sentences using the same splitter as MT and TTS."""
    text = value if isinstance(value, str) else detokenize(value, language)
    return len(_sentence_boundaries(text))
