from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from onevoice.config import TerminologyMtConfig

from ..errors import TerminologyCoverageError
from ..manager import TerminologyManager
from ..profile import TerminologyProfile
from ..schema import TranslationPolicy


_PLACEHOLDER_LIKE = re.compile(
    r"__\s*term\s*_\s*\d+\s*__"
    r"|<\s*term\s*_\s*\d+\s*>"
    r"|zx\s*term\s*\d+\s*zx"
    r"|ovt\s*\d+\s*ovt",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class MtTermBinding:
    placeholder: str
    term_id: str
    source_language: str
    target_language: str
    source_span: tuple[int, int]
    source_form: str
    target_canonical: str
    translation_policy: TranslationPolicy


@dataclass(frozen=True, slots=True)
class MtProtectionResult:
    text: str
    bindings: tuple[MtTermBinding, ...]


@dataclass(frozen=True, slots=True)
class MtTerminologyStats:
    matches: int = 0
    hard_matches: int = 0
    expected_placeholders: int = 0
    retries: int = 0
    fallbacks: int = 0
    hops: int = 0

    def __add__(self, other: "MtTerminologyStats") -> "MtTerminologyStats":
        return MtTerminologyStats(
            matches=self.matches + other.matches,
            hard_matches=self.hard_matches + other.hard_matches,
            expected_placeholders=(
                self.expected_placeholders + other.expected_placeholders
            ),
            retries=self.retries + other.retries,
            fallbacks=self.fallbacks + other.fallbacks,
            hops=self.hops + other.hops,
        )


class TerminologyMtProtector:
    """Protect matched terms for one immutable source-to-target MT hop."""

    def __init__(self, profile: TerminologyProfile) -> None:
        if len(profile.mt_hops) != 1:
            raise ValueError("MT protector requires a direct, single-hop profile")
        self.profile = profile
        self.artifact = profile.mt_hops[0]

    def protect(self, text: str, placeholder_format: str) -> MtProtectionResult:
        matches = self.profile.source_matcher.find(text)
        bindings: list[MtTermBinding] = []
        pieces: list[str] = []
        cursor = 0
        for index, match in enumerate(matches, start=1):
            start, end = match.original_span
            placeholder = placeholder_format.format(id=index)
            artifact = self.artifact.terms[match.term_id]
            target_canonical = (
                artifact.source_canonical
                if artifact.policy == TranslationPolicy.PRESERVE
                else artifact.target_canonical
            )
            pieces.extend((text[cursor:start], placeholder))
            cursor = end
            bindings.append(
                MtTermBinding(
                    placeholder=placeholder,
                    term_id=match.term_id,
                    source_language=self.artifact.source_language,
                    target_language=self.artifact.target_language,
                    source_span=match.original_span,
                    source_form=text[start:end],
                    target_canonical=target_canonical,
                    translation_policy=artifact.policy,
                )
            )
        pieces.append(text[cursor:])
        return MtProtectionResult("".join(pieces), tuple(bindings))

    def restore_and_validate(
        self,
        translated_text: str,
        bindings: tuple[MtTermBinding, ...],
        *,
        validate_order: bool = False,
    ) -> str:
        expected = {binding.placeholder: binding for binding in bindings}
        positions: list[int] = []
        for placeholder in expected:
            count = translated_text.count(placeholder)
            if count == 0:
                raise TerminologyCoverageError(
                    f"MT lost or mutated placeholder {placeholder!r}"
                )
            if count > 1:
                raise TerminologyCoverageError(
                    f"MT duplicated placeholder {placeholder!r}"
                )
            positions.append(translated_text.index(placeholder))

        observed = tuple(match.group(0) for match in _PLACEHOLDER_LIKE.finditer(translated_text))
        unexpected = tuple(value for value in observed if value not in expected)
        if unexpected:
            raise TerminologyCoverageError(
                f"MT emitted corrupted or unexpected placeholders: {unexpected!r}"
            )
        if validate_order and positions != sorted(positions):
            raise TerminologyCoverageError("MT reordered terminology placeholders")

        restored = translated_text
        for binding in bindings:
            restored = restored.replace(
                binding.placeholder, binding.target_canonical, 1
            )
        if _PLACEHOLDER_LIKE.search(restored):
            raise TerminologyCoverageError(
                "MT terminology restore left a raw placeholder in output"
            )
        return restored


class TerminologyMtRuntime:
    """Request-local MT protection with retry and per-hop canonicalization."""

    def __init__(
        self,
        manager: TerminologyManager,
        *,
        domain: str | None,
        config: TerminologyMtConfig,
    ) -> None:
        self.manager = manager
        self.domain = domain
        self.config = config
        self._protectors: dict[tuple[str, str], TerminologyMtProtector] = {}

    def _protector(self, source: str, target: str) -> TerminologyMtProtector:
        key = (source, target)
        protector = self._protectors.get(key)
        if protector is None:
            profile = self.manager.activate(
                domain=self.domain,
                source_language=source,
                target_language=target,
                mt_route=(source, target),
            )
            protector = TerminologyMtProtector(profile)
            self._protectors[key] = protector
        return protector

    def translate_hop(
        self,
        text: str,
        source: str,
        target: str,
        translate_once: Callable[[str], str],
    ) -> tuple[str, MtTerminologyStats]:
        protector = self._protector(source, target)
        last_error: TerminologyCoverageError | None = None
        match_count = 0
        failed_outputs: list[
            tuple[str, tuple[MtTermBinding, ...]]
        ] = []
        for attempt, placeholder_format in enumerate(
            self.config.placeholder_formats
        ):
            protected = protector.protect(text, placeholder_format)
            match_count = len(protected.bindings)
            if not protected.bindings:
                return translate_once(text), MtTerminologyStats(hops=1)
            translated = translate_once(protected.text)
            try:
                restored = protector.restore_and_validate(
                    translated,
                    protected.bindings,
                    validate_order=self.config.validate_order,
                )
            except TerminologyCoverageError as exc:
                last_error = exc
                failed_outputs.append((translated, protected.bindings))
                continue
            return restored, MtTerminologyStats(
                matches=match_count,
                hard_matches=match_count,
                expected_placeholders=match_count,
                retries=attempt,
                hops=1,
            )
        assert last_error is not None
        if self.config.on_validation_error == "segment_fallback":
            repaired = self._repair_mutated_placeholders(
                protector, failed_outputs
            )
            if repaired is None:
                protected = protector.protect(
                    text, self.config.placeholder_formats[0]
                )
                repaired = self._translate_around_terms(
                    text,
                    protected.bindings,
                    translate_once,
                    target,
                )
            return repaired, MtTerminologyStats(
                matches=match_count,
                hard_matches=match_count,
                expected_placeholders=match_count,
                retries=max(0, len(self.config.placeholder_formats) - 1),
                fallbacks=1,
                hops=1,
            )
        raise TerminologyCoverageError(
            f"MT terminology validation failed after "
            f"{len(self.config.placeholder_formats)} placeholder formats: {last_error}"
        ) from last_error

    @staticmethod
    def _repair_mutated_placeholders(
        protector: TerminologyMtProtector,
        failed_outputs: list[
            tuple[str, tuple[MtTermBinding, ...]]
        ],
    ) -> str | None:
        """Recover only an unambiguous one-for-one placeholder mutation."""
        for translated, bindings in reversed(failed_outputs):
            observed = tuple(_PLACEHOLDER_LIKE.finditer(translated))
            if len(observed) != len(bindings):
                continue
            pieces: list[str] = []
            cursor = 0
            for match, binding in zip(observed, bindings, strict=True):
                pieces.extend(
                    (
                        translated[cursor : match.start()],
                        binding.placeholder,
                    )
                )
                cursor = match.end()
            pieces.append(translated[cursor:])
            candidate = "".join(pieces)
            try:
                return protector.restore_and_validate(
                    candidate,
                    bindings,
                    validate_order=True,
                )
            except TerminologyCoverageError:
                continue
        return None

    @staticmethod
    def _translate_around_terms(
        source_text: str,
        bindings: tuple[MtTermBinding, ...],
        translate_once: Callable[[str], str],
        target_language: str,
    ) -> str:
        """Last-resort MT that can never lose a matched canonical term."""
        parts: list[str] = []
        cursor = 0
        for binding in bindings:
            start, end = binding.source_span
            prefix = source_text[cursor:start].strip()
            if prefix:
                parts.append(translate_once(prefix).strip())
            parts.append(binding.target_canonical)
            cursor = end
        suffix = source_text[cursor:].strip()
        if suffix:
            parts.append(translate_once(suffix).strip())
        separator = "" if target_language == "zh" else " "
        return separator.join(part for part in parts if part).strip()
