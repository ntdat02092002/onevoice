from __future__ import annotations

from dataclasses import dataclass

from onevoice.models import AsrUpdate, AsrWordTiming
from onevoice.text import tokenize_text

from ..manager import TerminologyManager
from ..profile import TerminologyProfile


@dataclass(frozen=True, slots=True)
class AsrCorrectionStats:
    corrections: int = 0
    timing_drops: int = 0


@dataclass(frozen=True, slots=True)
class AsrHotword:
    term_id: str
    text: str
    score: float
    token_count: int


class TerminologyAsrCorrector:
    """Canonicalize only exact aliases declared by the active ASR profile."""

    def __init__(self, profile: TerminologyProfile) -> None:
        self.profile = profile

    def correct(
        self, update: AsrUpdate
    ) -> tuple[AsrUpdate, AsrCorrectionStats]:
        matches = tuple(
            match
            for match in self.profile.source_matcher.find(update.text)
            if match.is_alias
            and update.text[slice(*match.original_span)] != match.canonical
        )
        if not matches:
            return update, AsrCorrectionStats()

        pieces: list[str] = []
        cursor = 0
        for match in matches:
            start, end = match.original_span
            pieces.extend((update.text[cursor:start], match.canonical))
            cursor = end
        pieces.append(update.text[cursor:])
        corrected_text = "".join(pieces)
        corrected_words = self._correct_words(
            update.text, update.words, matches
        )
        timing_drops = int(bool(update.words) and corrected_words is None)
        return (
            AsrUpdate(
                text=corrected_text,
                language=update.language,
                confidence=update.confidence,
                revision=update.revision,
                is_final=update.is_final,
                started_at=update.started_at,
                completed_at=update.completed_at,
                tokens=tokenize_text(
                    corrected_text, update.language or self.profile.source_language
                ),
                words=corrected_words or (),
                is_endpoint_cut=update.is_endpoint_cut,
            ),
            AsrCorrectionStats(len(matches), timing_drops),
        )

    @staticmethod
    def _correct_words(
        text: str,
        words: tuple[AsrWordTiming, ...],
        matches,
    ) -> tuple[AsrWordTiming, ...] | None:
        if not words:
            return ()
        folded = text.casefold()
        word_spans: list[tuple[int, int]] = []
        cursor = 0
        for word in words:
            value = word.text.strip()
            start = folded.find(value.casefold(), cursor)
            if start < 0:
                return None
            end = start + len(value)
            word_spans.append((start, end))
            cursor = end

        replacements: dict[int, tuple[int, AsrWordTiming]] = {}
        consumed: set[int] = set()
        for match in matches:
            start, end = match.original_span
            indices = [
                index
                for index, (word_start, word_end) in enumerate(word_spans)
                if word_start < end and start < word_end
            ]
            if not indices:
                return None
            first, last = indices[0], indices[-1]
            confidence_values = [
                words[index].confidence
                for index in indices
                if words[index].confidence is not None
            ]
            replacements[first] = (
                last,
                AsrWordTiming(
                    match.canonical,
                    words[first].start_seconds,
                    words[last].end_seconds,
                    (
                        min(confidence_values)
                        if confidence_values
                        else None
                    ),
                ),
            )
            consumed.update(indices[1:])

        output: list[AsrWordTiming] = []
        index = 0
        while index < len(words):
            replacement = replacements.get(index)
            if replacement is not None:
                last, timing = replacement
                output.append(timing)
                index = last + 1
                continue
            if index not in consumed:
                output.append(words[index])
            index += 1
        return tuple(output)


class TerminologyAsrRuntime:
    """Lazy language profiles for ASR prompt compilation and correction."""

    def __init__(
        self,
        manager: TerminologyManager,
        *,
        domain: str | None,
    ) -> None:
        self.manager = manager
        self.domain = domain
        self._profiles: dict[str, TerminologyProfile] = {}
        self._correctors: dict[str, TerminologyAsrCorrector] = {}

    def _profile(self, language: str) -> TerminologyProfile:
        profile = self._profiles.get(language)
        if profile is None:
            profile = self.manager.activate(
                domain=self.domain,
                source_language=language,
                target_language=language,
            )
            self._profiles[language] = profile
        return profile

    def prompt_terms(
        self,
        language: str,
        *,
        max_terms: int,
        max_tokens: int,
    ) -> tuple[str, ...]:
        output: list[str] = []
        token_count = 0
        for artifact in self._profile(language).asr_terms:
            term_tokens = tokenize_text(artifact.text, language)
            if not term_tokens:
                continue
            if len(output) >= max_terms:
                break
            if token_count + len(term_tokens) > max_tokens:
                continue
            output.append(artifact.text)
            token_count += len(term_tokens)
        return tuple(output)

    def hotwords(
        self,
        language: str,
        *,
        max_terms: int,
        max_tokens: int,
        text_case: str = "preserve",
    ) -> tuple[AsrHotword, ...]:
        output: list[AsrHotword] = []
        token_count = 0
        for artifact in self._profile(language).asr_terms:
            text = (
                artifact.text.upper()
                if text_case == "upper"
                else artifact.text
            )
            term_tokens = tokenize_text(text, language)
            if not term_tokens:
                continue
            if len(output) >= max_terms:
                break
            if token_count + len(term_tokens) > max_tokens:
                continue
            output.append(
                AsrHotword(
                    artifact.term_id,
                    text,
                    artifact.score,
                    len(term_tokens),
                )
            )
            token_count += len(term_tokens)
        return tuple(output)

    def correct(
        self, update: AsrUpdate
    ) -> tuple[AsrUpdate, AsrCorrectionStats]:
        language = update.language
        if not language:
            return update, AsrCorrectionStats()
        corrector = self._correctors.get(language)
        if corrector is None:
            corrector = TerminologyAsrCorrector(
                self._profile(language)
            )
            self._correctors[language] = corrector
        return corrector.correct(update)
