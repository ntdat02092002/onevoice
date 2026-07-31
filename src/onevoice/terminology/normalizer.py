from __future__ import annotations

from dataclasses import dataclass
import unicodedata


_HYPHENS = frozenset("‐‑‒–—−")


def _is_hangul_lead(character: str) -> bool:
    value = ord(character)
    return 0x1100 <= value <= 0x115F or 0xA960 <= value <= 0xA97C


def _is_hangul_vowel(character: str) -> bool:
    value = ord(character)
    return 0x1160 <= value <= 0x11A7 or 0xD7B0 <= value <= 0xD7C6


def _is_hangul_tail(character: str) -> bool:
    value = ord(character)
    return 0x11A8 <= value <= 0x11FF or 0xD7CB <= value <= 0xD7FB


@dataclass(frozen=True, slots=True)
class NormalizedText:
    text: str
    original_spans: tuple[tuple[int, int], ...]

    def original_span(self, start: int, end: int) -> tuple[int, int]:
        if start < 0 or end < start or end > len(self.text):
            raise ValueError("normalized span is out of bounds")
        if start == end:
            if start == len(self.original_spans):
                edge = self.original_spans[-1][1] if self.original_spans else 0
                return edge, edge
            edge = self.original_spans[start][0]
            return edge, edge
        return self.original_spans[start][0], self.original_spans[end - 1][1]


def _clusters(text: str) -> tuple[tuple[str, int, int], ...]:
    output: list[tuple[str, int, int]] = []
    index = 0
    while index < len(text):
        start = index
        index += 1
        if (
            _is_hangul_lead(text[start])
            and index < len(text)
            and _is_hangul_vowel(text[index])
        ):
            index += 1
            if index < len(text) and _is_hangul_tail(text[index]):
                index += 1
        while index < len(text) and unicodedata.combining(text[index]):
            index += 1
        output.append((text[start:index], start, index))
    return tuple(output)


def normalize_with_alignment(
    text: str,
    language: str | None = None,
    *,
    case_sensitive: bool = False,
    normalize_hyphens: bool = True,
) -> NormalizedText:
    """Normalize text while mapping every output character to its source span."""
    del language  # Reserved for language-specific extensions without changing the API.
    characters: list[str] = []
    spans: list[tuple[int, int]] = []
    whitespace_start: int | None = None
    whitespace_end: int | None = None

    def flush_whitespace() -> None:
        nonlocal whitespace_start, whitespace_end
        if whitespace_start is not None and characters:
            characters.append(" ")
            spans.append((whitespace_start, whitespace_end or whitespace_start))
        whitespace_start = None
        whitespace_end = None

    for cluster, start, end in _clusters(text):
        value = unicodedata.normalize("NFC", cluster)
        if value.isspace():
            whitespace_start = start if whitespace_start is None else whitespace_start
            whitespace_end = end
            continue
        flush_whitespace()
        if normalize_hyphens:
            value = "".join("-" if character in _HYPHENS else character for character in value)
        if not case_sensitive:
            value = value.casefold()
        for character in value:
            characters.append(character)
            spans.append((start, end))

    # Trailing whitespace is intentionally discarded.
    return NormalizedText("".join(characters), tuple(spans))


def normalize_text(
    text: str,
    language: str | None = None,
    *,
    case_sensitive: bool = False,
    normalize_hyphens: bool = True,
) -> str:
    return normalize_with_alignment(
        text,
        language,
        case_sensitive=case_sensitive,
        normalize_hyphens=normalize_hyphens,
    ).text
