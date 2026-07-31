from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from onevoice.text import tokenize_text

from .normalizer import NormalizedText, normalize_text, normalize_with_alignment
from .schema import TerminologyEntry, TranslationPolicy


@dataclass(frozen=True, slots=True)
class TermMatch:
    term_id: str
    source_form: str
    canonical: str
    language: str
    normalized_span: tuple[int, int]
    original_span: tuple[int, int]
    priority: int
    declaration_order: int
    translation_policy: TranslationPolicy
    is_alias: bool

    @property
    def length(self) -> int:
        return self.normalized_span[1] - self.normalized_span[0]


@dataclass(frozen=True, slots=True)
class _Pattern:
    term_id: str
    source_form: str
    canonical: str
    priority: int
    declaration_order: int
    translation_policy: TranslationPolicy
    is_alias: bool
    case_sensitive: bool


@dataclass(slots=True)
class _Node:
    children: dict[str, "_Node"] = field(default_factory=dict)
    patterns: list[_Pattern] = field(default_factory=list)


def _looks_code_like(value: str) -> bool:
    letters = [character for character in value if character.isalpha()]
    return bool(
        any(character.isdigit() for character in value)
        or (any(character.isupper() for character in letters) and any(character.islower() for character in letters))
        or (letters and len(letters) <= 8 and all(character.isupper() for character in letters))
    )


def _is_word_character(value: str) -> bool:
    return bool(value) and (value.isalnum() or value == "_")


def _requires_boundaries(pattern: str, language: str) -> bool:
    if language == "zh":
        return False
    if language == "ko" and not any(character.isascii() and character.isalnum() for character in pattern):
        return False
    return True


class TerminologyMatcher:
    """Immutable character-trie matcher with normalized-to-original alignment."""

    def __init__(
        self,
        entries: Iterable[TerminologyEntry],
        language: str,
        *,
        case_sensitive_for_codes: bool = True,
    ) -> None:
        self.language = language
        self.case_sensitive_for_codes = case_sensitive_for_codes
        self._folded = _Node()
        self._sensitive = _Node()
        for entry in entries:
            form = entry.forms.get(language)
            if form is None:
                continue
            for form_order, value in enumerate(form.all_forms):
                sensitive = case_sensitive_for_codes and _looks_code_like(value)
                pattern = _Pattern(
                    term_id=entry.id,
                    source_form=value,
                    canonical=form.canonical,
                    priority=entry.priority,
                    declaration_order=entry.declaration_order * 1000 + form_order,
                    translation_policy=entry.translation_policy,
                    is_alias=form_order > 0,
                    case_sensitive=sensitive,
                )
                self._insert(self._sensitive if sensitive else self._folded, pattern)

    def _insert(self, root: _Node, pattern: _Pattern) -> None:
        value = normalize_text(
            pattern.source_form,
            self.language,
            case_sensitive=pattern.case_sensitive,
        )
        node = root
        for character in value:
            node = node.children.setdefault(character, _Node())
        node.patterns.append(pattern)

    def find_all(self, text: str) -> tuple[TermMatch, ...]:
        output = [
            *self._scan(text, self._folded, case_sensitive=False),
            *self._scan(text, self._sensitive, case_sensitive=True),
        ]
        unique: dict[tuple[str, tuple[int, int], str], TermMatch] = {}
        for match in output:
            key = (match.term_id, match.original_span, match.source_form)
            unique[key] = match
        return tuple(
            sorted(
                unique.values(),
                key=lambda item: (
                    item.normalized_span[0],
                    -item.length,
                    -item.priority,
                    item.declaration_order,
                ),
            )
        )

    def _scan(
        self, text: str, root: _Node, *, case_sensitive: bool
    ) -> tuple[TermMatch, ...]:
        normalized = normalize_with_alignment(
            text, self.language, case_sensitive=case_sensitive
        )
        output: list[TermMatch] = []
        for start in range(len(normalized.text)):
            node = root
            cursor = start
            while cursor < len(normalized.text):
                node = node.children.get(normalized.text[cursor])
                if node is None:
                    break
                cursor += 1
                for pattern in node.patterns:
                    if self._has_valid_boundaries(normalized, start, cursor, pattern):
                        output.append(
                            TermMatch(
                                term_id=pattern.term_id,
                                source_form=pattern.source_form,
                                canonical=pattern.canonical,
                                language=self.language,
                                normalized_span=(start, cursor),
                                original_span=normalized.original_span(start, cursor),
                                priority=pattern.priority,
                                declaration_order=pattern.declaration_order,
                                translation_policy=pattern.translation_policy,
                                is_alias=pattern.is_alias,
                            )
                        )
        return tuple(output)

    def _has_valid_boundaries(
        self, normalized: NormalizedText, start: int, end: int, pattern: _Pattern
    ) -> bool:
        if not _requires_boundaries(pattern.source_form, self.language):
            return True
        before = normalized.text[start - 1] if start else ""
        after = normalized.text[end] if end < len(normalized.text) else ""
        return not _is_word_character(before) and not _is_word_character(after)

    def find(self, text: str) -> tuple[TermMatch, ...]:
        return resolve_overlaps(self.find_all(text))


def resolve_overlaps(matches: Iterable[TermMatch]) -> tuple[TermMatch, ...]:
    candidates = sorted(
        matches,
        key=lambda item: (
            -item.length,
            -item.priority,
            item.declaration_order,
            item.normalized_span[0],
        ),
    )
    selected: list[TermMatch] = []
    for candidate in candidates:
        start, end = candidate.normalized_span
        if any(
            start < existing.normalized_span[1]
            and existing.normalized_span[0] < end
            for existing in selected
        ):
            continue
        selected.append(candidate)
    return tuple(sorted(selected, key=lambda item: item.normalized_span))


@dataclass(slots=True)
class _TokenNode:
    children: dict[str, "_TokenNode"] = field(default_factory=dict)
    terminal_ids: set[str] = field(default_factory=set)


class TermPrefixTrie:
    def __init__(self, entries: Iterable[TerminologyEntry], language: str) -> None:
        self.language = language
        self._root = _TokenNode()
        for entry in entries:
            form = entry.forms.get(language)
            if form is None:
                continue
            for value in form.all_forms:
                tokens = self._tokens(value)
                if not tokens:
                    continue
                node = self._root
                for token in tokens:
                    node = node.children.setdefault(token, _TokenNode())
                node.terminal_ids.add(entry.id)

    def _tokens(self, value: str | Iterable[str]) -> tuple[str, ...]:
        if isinstance(value, str):
            text = normalize_text(value, self.language)
            return tuple(token.casefold() for token in tokenize_text(text, self.language))
        return tuple(str(token).casefold() for token in value)

    def is_prefix(self, tokens: str | Iterable[str]) -> bool:
        node = self._walk(self._tokens(tokens))
        return node is not None

    def is_term(self, tokens: str | Iterable[str]) -> bool:
        node = self._walk(self._tokens(tokens))
        return bool(node and node.terminal_ids)

    def is_open_prefix(self, tokens: str | Iterable[str]) -> bool:
        node = self._walk(self._tokens(tokens))
        return bool(node and node.children)

    def _walk(self, tokens: tuple[str, ...]) -> _TokenNode | None:
        node = self._root
        for token in tokens:
            node = node.children.get(token)
            if node is None:
                return None
        return node

    def longest_suffix_open_prefix(
        self, tokens: str | Iterable[str]
    ) -> int | None:
        values = self._tokens(tokens)
        for start in range(len(values)):
            node = self._walk(values[start:])
            if node is not None and node.children:
                return start
        return None

    def term_spans(
        self, tokens: str | Iterable[str]
    ) -> tuple[tuple[int, int], ...]:
        """Return non-overlapping complete terms, preferring longest matches."""
        values = self._tokens(tokens)
        spans: list[tuple[int, int]] = []
        start = 0
        while start < len(values):
            node = self._root
            cursor = start
            longest_end: int | None = None
            while cursor < len(values):
                node = node.children.get(values[cursor])
                if node is None:
                    break
                cursor += 1
                if node.terminal_ids:
                    longest_end = cursor
            if longest_end is None:
                start += 1
                continue
            spans.append((start, longest_end))
            start = longest_end
        return tuple(spans)
