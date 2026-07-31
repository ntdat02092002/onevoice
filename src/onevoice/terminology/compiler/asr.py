from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..normalizer import normalize_text
from ..schema import TerminologyEntry


@dataclass(frozen=True, slots=True)
class AsrTermArtifact:
    term_id: str
    text: str
    score: float


def compile_asr_terms(
    entries: Iterable[TerminologyEntry],
    language: str,
    *,
    default_score: float = 1.5,
) -> tuple[AsrTermArtifact, ...]:
    output: list[AsrTermArtifact] = []
    seen: set[str] = set()
    for entry in entries:
        form = entry.forms.get(language)
        if form is None:
            continue
        score = form.asr_boost or default_score
        for value in form.all_forms:
            key = normalize_text(value, language)
            if key in seen:
                continue
            seen.add(key)
            output.append(AsrTermArtifact(entry.id, value, score))
    return tuple(output)
