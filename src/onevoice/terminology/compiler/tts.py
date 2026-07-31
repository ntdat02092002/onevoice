from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterable, Mapping

from ..schema import TerminologyEntry


@dataclass(frozen=True, slots=True)
class TtsTermArtifact:
    term_id: str
    display_form: str
    spoken_form: str


@dataclass(frozen=True, slots=True)
class TtsArtifact:
    language: str
    terms: Mapping[str, TtsTermArtifact]


def compile_tts(
    entries: Iterable[TerminologyEntry], language: str
) -> TtsArtifact:
    terms = {}
    for entry in entries:
        form = entry.forms.get(language)
        if form is None:
            continue
        spoken = entry.tts.get(language)
        if spoken is None:
            continue
        terms[entry.id] = TtsTermArtifact(
            term_id=entry.id,
            display_form=form.canonical,
            spoken_form=spoken.spoken_form,
        )
    return TtsArtifact(language=language, terms=MappingProxyType(terms))
