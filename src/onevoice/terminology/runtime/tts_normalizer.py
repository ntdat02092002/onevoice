from __future__ import annotations

from dataclasses import dataclass

from ..profile import TerminologyProfile


@dataclass(frozen=True, slots=True)
class TtsNormalizationResult:
    display_text: str
    spoken_text: str
    substitutions: int = 0

    @property
    def changed(self) -> bool:
        return self.display_text != self.spoken_text


class TerminologyTtsNormalizer:
    """Convert canonical display terms to locale-specific synthesis forms."""

    def __init__(self, profile: TerminologyProfile) -> None:
        self.profile = profile

    def normalize(self, display_text: str) -> TtsNormalizationResult:
        pieces: list[str] = []
        cursor = 0
        substitutions = 0
        for match in self.profile.target_matcher.find(display_text):
            artifact = self.profile.tts.terms.get(match.term_id)
            if artifact is None:
                continue
            start, end = match.original_span
            if display_text[start:end] == artifact.spoken_form:
                continue
            pieces.extend(
                (
                    display_text[cursor:start],
                    artifact.spoken_form,
                )
            )
            cursor = end
            substitutions += 1
        if not substitutions:
            return TtsNormalizationResult(
                display_text,
                display_text,
            )
        pieces.append(display_text[cursor:])
        return TtsNormalizationResult(
            display_text,
            "".join(pieces),
            substitutions,
        )
