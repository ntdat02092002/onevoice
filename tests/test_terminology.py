from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from onevoice.terminology import (
    BundleValidationError,
    ProfileActivationError,
    TermPrefixTrie,
    TerminologyManager,
    TerminologyMatcher,
    load_bundle,
    normalize_text,
    normalize_with_alignment,
)


SAMPLE_BUNDLE = Path("assets/terminology/factory-sample-v1/terminology.yaml")


def _write_bundle(tmp_path: Path, entries: str, **root: str) -> Path:
    path = tmp_path / "terminology.yaml"
    path.write_text(
        "bundle_id: test-v1\n"
        "schema_version: 1\n"
        f"{root.get('extra', '')}"
        "default_domains: [test]\n"
        "entries:\n"
        f"{entries}",
        encoding="utf-8",
    )
    return path


def _entry(
    term_id: str,
    canonical: str,
    *,
    priority: int = 10,
    domain: str = "test",
    aliases: tuple[str, ...] = (),
    languages: tuple[str, ...] = ("vi", "en"),
) -> str:
    aliases_yaml = (
        "[" + ", ".join(f'"{value}"' for value in aliases) + "]"
        if aliases
        else "[]"
    )
    forms = "".join(
        f"      {language}:\n"
        f'        canonical: "{canonical}"\n'
        f"        aliases: {aliases_yaml}\n"
        for language in languages
    )
    return (
        f"  - id: {term_id}\n"
        f"    domain: [{domain}]\n"
        f"    priority: {priority}\n"
        "    translation_policy: preferred_term\n"
        "    forms:\n"
        f"{forms}"
    )


def test_sample_bundle_loads_as_immutable_schema() -> None:
    bundle = load_bundle(SAMPLE_BUNDLE)

    assert bundle.bundle_id == "factory-sample-v1"
    assert bundle.schema_version == 1
    assert len(bundle.entries) == 11
    assert bundle.entry("m5stack").forms["en"].canonical == "M5Stack"
    assert (
        bundle.entry("plc_s7_1200").tts["vi"].spoken_form
        == "pi eo xi ét bảy một hai không không"
    )
    with pytest.raises(TypeError):
        bundle.entry("m5stack").forms["en"] = bundle.entry("m5stack").forms["vi"]
    with pytest.raises(FrozenInstanceError):
        bundle.entries[0].priority = 1


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (
            "bundle_id: test\nschema_version: 2\ndefault_domains: []\nentries: []\n",
            "schema_version",
        ),
        (
            "bundle_id: test\nschema_version: 1\nunknown: true\n"
            "default_domains: []\nentries: []\n",
            "unknown fields",
        ),
        (
            "bundle_id: test\nschema_version: 1\ndefault_domains: []\nentries: []\n",
            "non-empty list",
        ),
    ],
)
def test_bundle_rejects_invalid_root_schema(
    tmp_path: Path, content: str, message: str
) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(BundleValidationError, match=message):
        load_bundle(path)


def test_bundle_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = _write_bundle(
        tmp_path,
        _entry("duplicate", "alpha") + _entry("duplicate", "beta"),
    )

    with pytest.raises(BundleValidationError, match="duplicate IDs"):
        load_bundle(path)


def test_bundle_rejects_ambiguous_alias_in_same_domain_and_priority(
    tmp_path: Path,
) -> None:
    path = _write_bundle(
        tmp_path,
        _entry("first", "first", aliases=("shared",))
        + _entry("second", "second", aliases=("shared",)),
    )

    with pytest.raises(BundleValidationError, match="ambiguous alias"):
        load_bundle(path)


def test_bundle_allows_alias_conflict_when_priority_is_deterministic(
    tmp_path: Path,
) -> None:
    path = _write_bundle(
        tmp_path,
        _entry("first", "first", priority=20, aliases=("shared",))
        + _entry("second", "second", priority=10, aliases=("shared",)),
    )

    assert len(load_bundle(path).entries) == 2


def test_unicode_normalization_preserves_original_alignment() -> None:
    source = "  nu\u0301t\u00a0dừng  "
    normalized = normalize_with_alignment(source, "vi")

    assert normalized.text == "nút dừng"
    acute_index = normalized.text.index("ú")
    assert source[slice(*normalized.original_span(acute_index, acute_index + 1))] == "u\u0301"
    assert normalize_text("PLC S7–1200", "vi", case_sensitive=True) == "PLC S7-1200"
    assert normalize_text("\u1107\u1175\u1109\u1161\u11bc", "ko") == "비상"


def test_matcher_prefers_longest_overlap_and_maps_original_span() -> None:
    bundle = load_bundle(SAMPLE_BUNDLE)
    entries = (
        bundle.entry("emergency_stop_button"),
        bundle.entry("stop_button"),
    )
    matcher = TerminologyMatcher(entries, "vi")
    text = "Hãy nhấn nút dừng khẩn cấp ngay."

    all_ids = [item.term_id for item in matcher.find_all(text)]
    selected = matcher.find(text)

    assert all_ids == ["emergency_stop_button", "stop_button"]
    assert [item.term_id for item in selected] == ["emergency_stop_button"]
    start, end = selected[0].original_span
    assert text[start:end] == "nút dừng khẩn cấp"


def test_matcher_preserves_code_case_and_honors_case_sensitive_codes() -> None:
    entry = load_bundle(SAMPLE_BUNDLE).entry("m5stack")
    sensitive = TerminologyMatcher((entry,), "en", case_sensitive_for_codes=True)
    folded = TerminologyMatcher((entry,), "en", case_sensitive_for_codes=False)

    assert sensitive.find("Open M5Stack.")[0].canonical == "M5Stack"
    assert sensitive.find("Open m5stack.") == ()
    assert folded.find("Open m5stack.")[0].canonical == "M5Stack"


def test_matcher_handles_chinese_without_spaces_and_korean_particle() -> None:
    entry = load_bundle(SAMPLE_BUNDLE).entry("emergency_stop_button")

    zh = TerminologyMatcher((entry,), "zh").find("请按下紧急停止按钮。")
    ko = TerminologyMatcher((entry,), "ko").find("비상 정지 버튼을 누르세요.")

    assert [item.term_id for item in zh] == ["emergency_stop_button"]
    assert [item.term_id for item in ko] == ["emergency_stop_button"]


def test_term_prefix_trie_detects_open_suffix_and_full_term() -> None:
    bundle = load_bundle(SAMPLE_BUNDLE)
    trie = TermPrefixTrie(
        (
            bundle.entry("emergency_stop_button"),
            bundle.entry("stop_button"),
        ),
        "vi",
    )

    assert trie.is_term("nút dừng")
    assert trie.is_open_prefix("nút dừng")
    assert trie.is_term("nút dừng khẩn cấp")
    assert not trie.is_open_prefix("nút dừng khẩn cấp")
    assert trie.longest_suffix_open_prefix(("hãy", "nhấn", "nút", "dừng")) == 2


def test_manager_activates_domain_profile_and_compiles_artifacts() -> None:
    manager = TerminologyManager.from_path(SAMPLE_BUNDLE)
    profile = manager.activate(
        domain="factory-maintenance",
        source_language="vi",
        target_language="ko",
        mt_route=("vi", "en", "ko"),
        asr_model_id="moonshine-vi",
        tts_model_id="vits-ko",
    )

    ids = {entry.id for entry in profile.entries}
    assert "m5stack" in ids
    assert "plc_s7_1200" in ids
    assert "crane_biology" not in ids
    assert profile.mt_route == ("vi", "en", "ko")
    assert [(hop.source_language, hop.target_language) for hop in profile.mt_hops] == [
        ("vi", "en"),
        ("en", "ko"),
    ]
    assert (
        profile.mt_hops[0].terms["emergency_stop_button"].target_canonical
        == "emergency stop button"
    )
    assert profile.asr_terms
    assert profile.tts.terms["m5stack"].spoken_form == "엠 파이브 스택"


def test_manager_uses_default_domains_and_filters_profile_conflict() -> None:
    manager = TerminologyManager.from_path(SAMPLE_BUNDLE)
    default_profile = manager.activate(
        domain=None,
        source_language="en",
        target_language="vi",
    )
    biology = manager.activate(
        domain="biology",
        source_language="en",
        target_language="vi",
    )

    assert "crane_construction" not in {item.id for item in default_profile.entries}
    assert {item.id for item in biology.entries} == {"crane_biology"}
    assert biology.source_matcher.find("A crane is near the lake.")[0].term_id == (
        "crane_biology"
    )


def test_profile_rejects_missing_route_canonical(tmp_path: Path) -> None:
    path = _write_bundle(
        tmp_path,
        _entry("partial", "term", languages=("vi", "en")),
    )
    manager = TerminologyManager.from_path(path)

    with pytest.raises(ProfileActivationError, match="do not cover MT route"):
        manager.activate(
            domain="test",
            source_language="vi",
            target_language="ko",
            mt_route=("vi", "en", "ko"),
        )


def test_profile_rejects_route_endpoints() -> None:
    manager = TerminologyManager.from_path(SAMPLE_BUNDLE)

    with pytest.raises(ProfileActivationError, match="must start"):
        manager.activate(
            domain="factory-safety",
            source_language="vi",
            target_language="ko",
            mt_route=("en", "ko"),
        )


def test_same_language_profile_has_no_mt_hop() -> None:
    profile = TerminologyManager.from_path(SAMPLE_BUNDLE).activate(
        domain="factory-safety",
        source_language="vi",
        target_language="vi",
    )

    assert profile.mt_route == ("vi",)
    assert profile.mt_hops == ()
