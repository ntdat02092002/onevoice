from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .compiler import (
    AsrTermArtifact,
    MtHopArtifact,
    TtsArtifact,
    compile_asr_terms,
    compile_mt_hop,
    compile_tts,
)
from .errors import ProfileActivationError
from .matcher import TermPrefixTrie, TerminologyMatcher
from .schema import (
    SUPPORTED_TERMINOLOGY_LANGUAGES,
    TerminologyBundle,
    TerminologyEntry,
)


@dataclass(frozen=True, slots=True)
class TerminologyProfile:
    bundle_id: str
    schema_version: int
    domains: tuple[str, ...]
    source_language: str
    target_language: str
    mt_route: tuple[str, ...]
    asr_model_id: str | None
    tts_model_id: str | None
    entries: tuple[TerminologyEntry, ...]
    source_matcher: TerminologyMatcher
    target_matcher: TerminologyMatcher
    source_trie: TermPrefixTrie
    target_trie: TermPrefixTrie
    asr_terms: tuple[AsrTermArtifact, ...]
    mt_hops: tuple[MtHopArtifact, ...]
    tts: TtsArtifact


def build_profile(
    bundle: TerminologyBundle,
    *,
    domain: str | None,
    source_language: str,
    target_language: str,
    mt_route: Sequence[str] | None = None,
    asr_model_id: str | None = None,
    tts_model_id: str | None = None,
    case_sensitive_for_codes: bool = True,
) -> TerminologyProfile:
    for name, language in (
        ("source_language", source_language),
        ("target_language", target_language),
    ):
        if language not in SUPPORTED_TERMINOLOGY_LANGUAGES:
            raise ProfileActivationError(f"{name}: unsupported language {language!r}")
    domains = (domain,) if domain else bundle.default_domains
    if not domains:
        raise ProfileActivationError(
            "domain is required when the bundle has no default_domains"
        )
    route = tuple(mt_route) if mt_route is not None else (
        (source_language,)
        if source_language == target_language
        else (source_language, target_language)
    )
    if not route:
        raise ProfileActivationError("MT route must not be empty")
    if route[0] != source_language or route[-1] != target_language:
        raise ProfileActivationError(
            f"MT route {route!r} must start at {source_language!r} and end at "
            f"{target_language!r}"
        )
    if any(language not in SUPPORTED_TERMINOLOGY_LANGUAGES for language in route):
        raise ProfileActivationError(f"MT route contains unsupported language: {route!r}")
    entries = tuple(
        entry for entry in bundle.entries if set(entry.domains).intersection(domains)
    )
    missing = {
        entry.id: tuple(language for language in route if language not in entry.forms)
        for entry in entries
    }
    missing = {term_id: languages for term_id, languages in missing.items() if languages}
    if missing:
        raise ProfileActivationError(
            f"active terms do not cover MT route {route!r}: {missing}"
        )

    return TerminologyProfile(
        bundle_id=bundle.bundle_id,
        schema_version=bundle.schema_version,
        domains=tuple(domains),
        source_language=source_language,
        target_language=target_language,
        mt_route=route,
        asr_model_id=asr_model_id,
        tts_model_id=tts_model_id,
        entries=entries,
        source_matcher=TerminologyMatcher(
            entries,
            source_language,
            case_sensitive_for_codes=case_sensitive_for_codes,
        ),
        target_matcher=TerminologyMatcher(
            entries,
            target_language,
            case_sensitive_for_codes=case_sensitive_for_codes,
        ),
        source_trie=TermPrefixTrie(entries, source_language),
        target_trie=TermPrefixTrie(entries, target_language),
        asr_terms=compile_asr_terms(entries, source_language),
        mt_hops=tuple(
            compile_mt_hop(entries, route[index], route[index + 1])
            for index in range(len(route) - 1)
        ),
        tts=compile_tts(entries, target_language),
    )
