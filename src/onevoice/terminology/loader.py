from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import re
from typing import Any

import yaml

from .errors import BundleValidationError
from .normalizer import normalize_text
from .schema import (
    LanguageForm,
    SUPPORTED_SCHEMA_VERSION,
    SUPPORTED_TERMINOLOGY_LANGUAGES,
    TerminologyBundle,
    TerminologyEntry,
    TranslationPolicy,
    TtsForm,
    immutable_mapping,
)


_ID = re.compile(r"^[a-z][a-z0-9_]*$")
_BUNDLE_KEYS = frozenset(
    ("bundle_id", "schema_version", "description", "default_domains", "entries")
)
_ENTRY_KEYS = frozenset(("id", "domain", "priority", "translation_policy", "forms", "tts"))
_FORM_KEYS = frozenset(("canonical", "aliases", "asr_boost"))
_TTS_KEYS = frozenset(("spoken_form",))


def _error(path: str, message: str) -> BundleValidationError:
    return BundleValidationError(f"{path}: {message}")


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _error(path, "must be a mapping")
    return value


def _strict_keys(value: Mapping[str, Any], allowed: frozenset[str], path: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise _error(path, f"unknown fields: {sorted(unknown)}")


def _required_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _error(path, "must be a non-empty string")
    return value.strip()


def _string_list(value: Any, path: str, *, allow_empty: bool = True) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise _error(path, "must be a list of strings")
    output = tuple(_required_string(item, f"{path}[{index}]") for index, item in enumerate(value))
    if not allow_empty and not output:
        raise _error(path, "must contain at least one item")
    if len(set(output)) != len(output):
        raise _error(path, "must not contain duplicates")
    return output


def _load_form(value: Any, path: str) -> LanguageForm:
    data = _mapping(value, path)
    _strict_keys(data, _FORM_KEYS, path)
    canonical = _required_string(data.get("canonical"), f"{path}.canonical")
    aliases = _string_list(data.get("aliases", []), f"{path}.aliases")
    normalized = [normalize_text(item) for item in (canonical, *aliases)]
    if len(set(normalized)) != len(normalized):
        raise _error(path, "canonical and aliases must be unique after normalization")
    boost = data.get("asr_boost")
    if boost is not None:
        if isinstance(boost, bool) or not isinstance(boost, (int, float)) or boost <= 0:
            raise _error(f"{path}.asr_boost", "must be a positive number")
        boost = float(boost)
    return LanguageForm(canonical=canonical, aliases=aliases, asr_boost=boost)


def _load_tts(value: Any, path: str) -> TtsForm:
    data = _mapping(value, path)
    _strict_keys(data, _TTS_KEYS, path)
    return TtsForm(
        spoken_form=_required_string(data.get("spoken_form"), f"{path}.spoken_form")
    )


def _load_entry(value: Any, index: int) -> TerminologyEntry:
    path = f"entries[{index}]"
    data = _mapping(value, path)
    _strict_keys(data, _ENTRY_KEYS, path)
    term_id = _required_string(data.get("id"), f"{path}.id")
    if not _ID.fullmatch(term_id):
        raise _error(f"{path}.id", "must use lowercase snake_case")
    domains = _string_list(data.get("domain"), f"{path}.domain", allow_empty=False)
    priority = data.get("priority")
    if isinstance(priority, bool) or not isinstance(priority, int):
        raise _error(f"{path}.priority", "must be an integer")
    try:
        policy = TranslationPolicy(data.get("translation_policy"))
    except (TypeError, ValueError) as exc:
        raise _error(
            f"{path}.translation_policy",
            f"must be one of {[item.value for item in TranslationPolicy]}",
        ) from exc

    raw_forms = _mapping(data.get("forms"), f"{path}.forms")
    if not raw_forms:
        raise _error(f"{path}.forms", "must define at least one language")
    invalid_languages = set(raw_forms) - SUPPORTED_TERMINOLOGY_LANGUAGES
    if invalid_languages:
        raise _error(f"{path}.forms", f"unsupported languages: {sorted(invalid_languages)}")
    forms = {
        language: _load_form(raw_forms[language], f"{path}.forms.{language}")
        for language in raw_forms
    }

    raw_tts = _mapping(data.get("tts", {}), f"{path}.tts")
    invalid_tts_languages = set(raw_tts) - SUPPORTED_TERMINOLOGY_LANGUAGES
    if invalid_tts_languages:
        raise _error(f"{path}.tts", f"unsupported languages: {sorted(invalid_tts_languages)}")
    undeclared = set(raw_tts) - set(forms)
    if undeclared:
        raise _error(
            f"{path}.tts", f"spoken forms require matching language forms: {sorted(undeclared)}"
        )
    tts = {
        language: _load_tts(raw_tts[language], f"{path}.tts.{language}")
        for language in raw_tts
    }
    return TerminologyEntry(
        id=term_id,
        domains=domains,
        priority=priority,
        translation_policy=policy,
        forms=immutable_mapping(forms),
        tts=immutable_mapping(tts),
        declaration_order=index,
    )


def _validate_alias_conflicts(entries: tuple[TerminologyEntry, ...]) -> None:
    seen: dict[tuple[str, str], list[TerminologyEntry]] = {}
    for entry in entries:
        for language, form in entry.forms.items():
            for value in form.all_forms:
                key = (language, normalize_text(value, language))
                for existing in seen.get(key, ()):
                    if (
                        set(existing.domains).intersection(entry.domains)
                        and existing.priority == entry.priority
                        and existing.id != entry.id
                    ):
                        raise _error(
                            "entries",
                            f"ambiguous alias {value!r} for {existing.id!r} and "
                            f"{entry.id!r}; separate domains or priorities",
                        )
                seen.setdefault(key, []).append(entry)


def load_bundle(path: str | Path) -> TerminologyBundle:
    selected = Path(path)
    try:
        raw = yaml.safe_load(selected.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise BundleValidationError(f"bundle file not found: {selected}") from exc
    except yaml.YAMLError as exc:
        raise BundleValidationError(f"invalid YAML in {selected}: {exc}") from exc

    data = _mapping(raw, "bundle")
    _strict_keys(data, _BUNDLE_KEYS, "bundle")
    bundle_id = _required_string(data.get("bundle_id"), "bundle.bundle_id")
    version = data.get("schema_version")
    if version != SUPPORTED_SCHEMA_VERSION:
        raise _error(
            "bundle.schema_version",
            f"must be {SUPPORTED_SCHEMA_VERSION}, got {version!r}",
        )
    description = data.get("description", "")
    if not isinstance(description, str):
        raise _error("bundle.description", "must be a string")
    default_domains = _string_list(
        data.get("default_domains", []), "bundle.default_domains"
    )
    raw_entries = data.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise _error("bundle.entries", "must be a non-empty list")
    entries = tuple(_load_entry(value, index) for index, value in enumerate(raw_entries))
    ids = [entry.id for entry in entries]
    if len(set(ids)) != len(ids):
        duplicates = sorted({term_id for term_id in ids if ids.count(term_id) > 1})
        raise _error("bundle.entries", f"duplicate IDs: {duplicates}")
    _validate_alias_conflicts(entries)
    return TerminologyBundle(
        bundle_id=bundle_id,
        schema_version=version,
        description=description.strip(),
        default_domains=default_domains,
        entries=entries,
    )
