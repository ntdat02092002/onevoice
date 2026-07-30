from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Mapping, TypeVar


SUPPORTED_TERMINOLOGY_LANGUAGES = frozenset(("vi", "en", "zh", "ko"))
SUPPORTED_SCHEMA_VERSION = 1


class TranslationPolicy(StrEnum):
    PRESERVE = "preserve"
    PREFERRED_TERM = "preferred_term"
    TRANSLITERATE = "transliterate"
    EXPAND = "expand"
    SPELL_OUT = "spell_out"
    DISPLAY_PRESERVE_SPEECH_OVERRIDE = "display_preserve_speech_override"


@dataclass(frozen=True, slots=True)
class LanguageForm:
    canonical: str
    aliases: tuple[str, ...] = ()
    asr_boost: float | None = None

    @property
    def all_forms(self) -> tuple[str, ...]:
        return (self.canonical, *self.aliases)


@dataclass(frozen=True, slots=True)
class TtsForm:
    spoken_form: str


@dataclass(frozen=True, slots=True)
class TerminologyEntry:
    id: str
    domains: tuple[str, ...]
    priority: int
    translation_policy: TranslationPolicy
    forms: Mapping[str, LanguageForm]
    tts: Mapping[str, TtsForm]
    declaration_order: int


@dataclass(frozen=True, slots=True)
class TerminologyBundle:
    bundle_id: str
    schema_version: int
    description: str
    default_domains: tuple[str, ...]
    entries: tuple[TerminologyEntry, ...]

    def entry(self, term_id: str) -> TerminologyEntry:
        for item in self.entries:
            if item.id == term_id:
                return item
        raise KeyError(term_id)


T = TypeVar("T")


def immutable_mapping(values: Mapping[str, T]) -> Mapping[str, T]:
    return MappingProxyType(dict(values))
