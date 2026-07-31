from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterable, Mapping

from ..schema import TerminologyEntry, TranslationPolicy


@dataclass(frozen=True, slots=True)
class MtTermArtifact:
    term_id: str
    source_canonical: str
    target_canonical: str
    policy: TranslationPolicy


@dataclass(frozen=True, slots=True)
class MtHopArtifact:
    source_language: str
    target_language: str
    terms: Mapping[str, MtTermArtifact]


def compile_mt_hop(
    entries: Iterable[TerminologyEntry],
    source_language: str,
    target_language: str,
) -> MtHopArtifact:
    terms = {
        entry.id: MtTermArtifact(
            term_id=entry.id,
            source_canonical=entry.forms[source_language].canonical,
            target_canonical=entry.forms[target_language].canonical,
            policy=entry.translation_policy,
        )
        for entry in entries
    }
    return MtHopArtifact(
        source_language=source_language,
        target_language=target_language,
        terms=MappingProxyType(terms),
    )
