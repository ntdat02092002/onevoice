from __future__ import annotations

import hashlib
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .manager import TerminologyManager
from .schema import SUPPORTED_TERMINOLOGY_LANGUAGES

RouteResolver = Callable[[str, str], Sequence[tuple[str, str]]]

_LANGUAGE_ORDER = ("vi", "en", "zh", "ko")


@dataclass(frozen=True, slots=True)
class CompiledProfileInfo:
    source_language: str
    target_language: str
    mt_route: tuple[str, ...]
    domains: tuple[str, ...]
    entry_count: int
    asr_term_count: int
    mt_binding_count: int
    tts_spoken_form_count: int


@dataclass(frozen=True, slots=True)
class TerminologyBuildInfo:
    bundle_id: str
    schema_version: int
    description: str
    bundle_path: str
    bundle_sha256: str
    selected_domain: str | None
    profiles: tuple[CompiledProfileInfo, ...]

    @property
    def profile_count(self) -> int:
        return len(self.profiles)

    @property
    def entry_count(self) -> int:
        return sum(profile.entry_count for profile in self.profiles)

    @property
    def asr_term_count(self) -> int:
        return sum(profile.asr_term_count for profile in self.profiles)

    @property
    def mt_binding_count(self) -> int:
        return sum(profile.mt_binding_count for profile in self.profiles)

    @property
    def tts_spoken_form_count(self) -> int:
        return sum(
            profile.tts_spoken_form_count for profile in self.profiles
        )


def _display_path(path: Path) -> str:
    relative = os.path.relpath(path.resolve(), Path.cwd().resolve())
    return Path(relative).as_posix()


def _route_languages(
    source: str,
    target: str,
    resolver: RouteResolver | None,
) -> tuple[str, ...]:
    if source == target:
        return (source,)
    if resolver is None:
        return (source, target)
    pairs = tuple(resolver(source, target))
    if not pairs:
        return (source,)
    route = (pairs[0][0], *(pair[1] for pair in pairs))
    if route[0] != source or route[-1] != target:
        raise ValueError(
            f"Invalid translation route for terminology: {route!r}"
        )
    for left, right in zip(route, route[1:]):
        if (left, right) not in pairs:
            raise ValueError(
                f"Discontinuous translation route for terminology: {pairs!r}"
            )
    return route


def prepare_terminology_bundle(
    path: str | Path,
    *,
    domain: str | None,
    source_language: str,
    target_language: str,
    case_sensitive_for_codes: bool = True,
    route_resolver: RouteResolver | None = None,
) -> tuple[TerminologyManager, TerminologyBuildInfo]:
    selected = Path(path)
    manager = TerminologyManager.from_path(
        selected,
        case_sensitive_for_codes=case_sensitive_for_codes,
    )
    content = selected.read_bytes()
    if source_language == "auto":
        sources = tuple(
            language
            for language in _LANGUAGE_ORDER
            if language != target_language
        )
    else:
        sources = (source_language,)
    unsupported = set(sources).difference(
        SUPPORTED_TERMINOLOGY_LANGUAGES
    )
    if unsupported:
        raise ValueError(
            f"Unsupported terminology source languages: {sorted(unsupported)}"
        )

    compiled: list[CompiledProfileInfo] = []
    for source in sources:
        route = _route_languages(
            source,
            target_language,
            route_resolver,
        )
        profile = manager.activate(
            domain=domain,
            source_language=source,
            target_language=target_language,
            mt_route=route,
        )
        compiled.append(
            CompiledProfileInfo(
                source_language=source,
                target_language=target_language,
                mt_route=profile.mt_route,
                domains=profile.domains,
                entry_count=len(profile.entries),
                asr_term_count=len(profile.asr_terms),
                mt_binding_count=sum(
                    len(hop.terms) for hop in profile.mt_hops
                ),
                tts_spoken_form_count=len(profile.tts.terms),
            )
        )

    bundle = manager.bundle
    return manager, TerminologyBuildInfo(
        bundle_id=bundle.bundle_id,
        schema_version=bundle.schema_version,
        description=bundle.description,
        bundle_path=_display_path(selected),
        bundle_sha256=hashlib.sha256(content).hexdigest(),
        selected_domain=domain,
        profiles=tuple(compiled),
    )
