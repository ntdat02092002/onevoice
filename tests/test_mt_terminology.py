from __future__ import annotations

from pathlib import Path

import pytest

from onevoice.backends.translation import (
    FakeTranslationBackend,
    M2M100Backend,
    OpusMtCTranslate2Backend,
)
from onevoice.config import (
    PipelineConfig,
    TerminologyMtConfig,
    TranslationConfig,
)
from onevoice.models import TranslationRequest
from onevoice.pipeline import RealtimePipeline
from onevoice.terminology import TerminologyCoverageError, TerminologyManager
from onevoice.terminology.runtime import TerminologyMtProtector, TerminologyMtRuntime


SAMPLE_BUNDLE = Path(
    "assets/terminology/factory-sample-v1/terminology.yaml"
)


def _manager() -> TerminologyManager:
    return TerminologyManager.from_path(SAMPLE_BUNDLE)


def _runtime(domain: str = "factory-safety") -> TerminologyMtRuntime:
    return TerminologyMtRuntime(
        _manager(),
        domain=domain,
        config=TerminologyMtConfig(),
    )


def test_protector_matches_alias_and_restores_target_canonical() -> None:
    profile = _manager().activate(
        domain="factory-safety",
        source_language="en",
        target_language="vi",
    )
    protector = TerminologyMtProtector(profile)

    protected = protector.protect(
        "Press the E-stop button now.", "__TERM_{id:04d}__"
    )

    assert protected.text == "Press the __TERM_0001__ now."
    assert protected.bindings[0].term_id == "emergency_stop_button"
    restored = protector.restore_and_validate(
        "Hãy nhấn __TERM_0001__ ngay.", protected.bindings
    )
    assert restored == "Hãy nhấn nút dừng khẩn cấp ngay."


@pytest.mark.parametrize(
    "translated",
    [
        "placeholder is missing",
        "__TERM_0001__ and __TERM_0001__",
        "__term_0001__",
        "__TERM_ 0001__",
        "__TERM_0001__ plus __TERM_9999__",
    ],
)
def test_validator_rejects_missing_duplicate_corrupted_and_extra(
    translated: str,
) -> None:
    profile = _manager().activate(
        domain="factory-safety",
        source_language="en",
        target_language="vi",
    )
    protector = TerminologyMtProtector(profile)
    protected = protector.protect(
        "Use the emergency stop button.", "__TERM_{id:04d}__"
    )

    with pytest.raises(TerminologyCoverageError):
        protector.restore_and_validate(translated, protected.bindings)


def test_validator_can_enforce_placeholder_order() -> None:
    profile = _manager().activate(
        domain="factory-maintenance",
        source_language="en",
        target_language="vi",
    )
    protector = TerminologyMtProtector(profile)
    protected = protector.protect(
        "EtherCAT and M5Stack", "__TERM_{id:04d}__"
    )

    with pytest.raises(TerminologyCoverageError, match="reordered"):
        protector.restore_and_validate(
            "__TERM_0002__ then __TERM_0001__",
            protected.bindings,
            validate_order=True,
        )


def test_runtime_retries_next_placeholder_format() -> None:
    calls: list[str] = []

    def translate_once(text: str) -> str:
        calls.append(text)
        if text.startswith("__TERM_"):
            return text.lower()
        return text

    translated, stats = _runtime().translate_hop(
        "emergency stop button",
        "en",
        "vi",
        translate_once,
    )

    assert translated == "nút dừng khẩn cấp"
    assert calls == ["__TERM_0001__", "OVT0001OVT"]
    assert stats.matches == 1
    assert stats.retries == 1


def test_runtime_never_returns_raw_placeholder_after_all_formats_fail() -> None:
    with pytest.raises(TerminologyCoverageError, match="3 placeholder formats"):
        _runtime().translate_hop(
            "emergency stop button",
            "en",
            "vi",
            lambda text: text.lower(),
        )


def test_runtime_repairs_unambiguous_mutated_placeholder_in_fallback_mode() -> None:
    config = TerminologyMtConfig(on_validation_error="segment_fallback")
    runtime = TerminologyMtRuntime(
        _manager(),
        domain="test",
        config=config,
    )

    translated, stats = runtime.translate_hop(
        "Use a windsurfing board.",
        "en",
        "vi",
        lambda text: (
            "Dùng bảng ZXTERM 10001ZX."
            if text.startswith("Use a ZXTERM")
            else text.lower()
        ),
    )

    assert translated == "Dùng bảng windsurfing."
    assert stats.fallbacks == 1
    assert "ZXTERM" not in translated


def test_runtime_segment_fallback_guarantees_term_when_placeholder_is_lost() -> None:
    config = TerminologyMtConfig(on_validation_error="segment_fallback")
    runtime = TerminologyMtRuntime(
        _manager(),
        domain="test",
        config=config,
    )

    translated, stats = runtime.translate_hop(
        "Try windsurfing today.",
        "en",
        "vi",
        lambda text: "bản dịch",
    )

    assert translated == "bản dịch windsurfing bản dịch"
    assert stats.fallbacks == 1
    assert stats.retries == 2


def test_opus_canonicalizes_terms_after_each_pivot_hop() -> None:
    backend = OpusMtCTranslate2Backend(TranslationConfig())
    backend.configure_terminology(
        _manager(),
        domain="factory-maintenance",
        config=TerminologyMtConfig(),
    )
    backend._ensure_route = lambda source, target: (
        (source, "en"),
        ("en", target),
    )
    calls: list[tuple[str, tuple[str, str]]] = []

    def translate_once(text: str, pair: tuple[str, str]) -> str:
        calls.append((text, pair))
        return f"use {text}" if pair[1] == "en" else f"{text} 확인"

    backend._translate_once = translate_once
    update = backend.translate(
        TranslationRequest("băng tải", "vi", "ko", 11, False)
    )

    assert calls == [
        ("__TERM_0001__", ("vi", "en")),
        ("use __TERM_0001__", ("en", "ko")),
    ]
    assert update.text == "use 컨베이어 확인"
    assert update.terminology_matches == 2
    assert update.terminology_hops == 2


def test_m2m100_direct_route_uses_same_protector() -> None:
    backend = M2M100Backend(
        TranslationConfig(
            backend="m2m100",
            model="facebook/m2m100_418M",
        )
    )
    backend.configure_terminology(
        _manager(),
        domain="factory-maintenance",
        config=TerminologyMtConfig(),
    )
    seen: list[str] = []
    backend._translate_once = (
        lambda text, source, target: seen.append(text)
        or f"检查{text.removeprefix('Check ')}"
    )

    update = backend.translate(
        TranslationRequest("Check EtherCAT.", "en", "zh", 3, True)
    )

    assert seen == ["Check __TERM_0001__."]
    assert update.text == "检查EtherCAT."
    assert update.terminology_matches == 1


def test_fake_backend_supports_app_smoke_testing_with_terminology() -> None:
    backend = FakeTranslationBackend(
        TranslationConfig(
            backend="fake",
            source_language="en",
            target_language="vi",
        )
    )
    backend.configure_terminology(
        _manager(),
        domain="factory-safety",
        config=TerminologyMtConfig(),
    )

    update = backend.translate(
        TranslationRequest("E-stop button", "en", "vi", 1, True)
    )

    assert update.text == "[vi] nút dừng khẩn cấp"
    assert update.terminology_matches == 1
    assert "__TERM" not in update.text


def test_sample_test_profile_preserves_windsurfing_and_outdoor_life() -> None:
    profile = _manager().activate(
        domain="test",
        source_language="en",
        target_language="vi",
    )
    protector = TerminologyMtProtector(profile)

    protected = protector.protect(
        "I tried winssurfing at Outdoor Life.",
        "__TERM_{id:04d}__",
    )

    assert protected.text == (
        "I tried __TERM_0001__ at __TERM_0002__."
    )
    assert [binding.term_id for binding in protected.bindings] == [
        "windsurfing",
        "outdoor_life",
    ]
    restored = protector.restore_and_validate(
        "Tôi thử __TERM_0001__ tại __TERM_0002__.",
        protected.bindings,
    )
    assert restored == "Tôi thử windsurfing tại Outdoor Life."


def test_opus_partial_splits_sentences_and_caches_only_complete_prefix() -> None:
    backend = OpusMtCTranslate2Backend(TranslationConfig())
    backend.configure_terminology(
        _manager(),
        domain="test",
        config=TerminologyMtConfig(),
    )
    backend._ensure_route = lambda source, target: ((source, target),)
    calls: list[str] = []
    backend._translate_once = (
        lambda text, pair: calls.append(text) or f"vi[{text}]"
    )
    stream_id = (3, 7)

    first = backend.translate(
        TranslationRequest(
            "I joined Outdoor Life. active tail",
            "en",
            "vi",
            1,
            False,
            stream_id=stream_id,
        )
    )
    second = backend.translate(
        TranslationRequest(
            "I joined Outdoor Life. active tail grows",
            "en",
            "vi",
            2,
            False,
            stream_id=stream_id,
        )
    )

    assert calls == [
        "I joined __TERM_0001__.",
        "active tail",
        "active tail grows",
    ]
    assert "Outdoor Life" in first.text
    assert "Outdoor Life" in second.text
    assert first.sentence_cache_hits == 0
    assert second.sentence_cache_hits == 1

    # An exact source change is a cache miss even inside the same stream.
    backend.translate(
        TranslationRequest(
            "I joined the Outdoor Life. active tail grows",
            "en",
            "vi",
            3,
            False,
            stream_id=stream_id,
        )
    )
    assert calls[-2:] == [
        "I joined the __TERM_0001__.",
        "active tail grows",
    ]


def test_opus_final_reuses_partial_sentence_then_drops_stream_cache() -> None:
    backend = OpusMtCTranslate2Backend(TranslationConfig())
    backend.configure_terminology(
        _manager(),
        domain="test",
        config=TerminologyMtConfig(),
    )
    backend._ensure_route = lambda source, target: ((source, target),)
    calls: list[str] = []
    backend._translate_once = (
        lambda text, pair: calls.append(text) or text
    )
    stream_id = (4, 9)
    source = "I joined Outdoor Life."

    backend.translate(
        TranslationRequest(
            source,
            "en",
            "vi",
            1,
            False,
            stream_id=stream_id,
        )
    )
    final = backend.translate(
        TranslationRequest(
            source,
            "en",
            "vi",
            2,
            True,
            stream_id=stream_id,
        )
    )
    next_stream = backend.translate(
        TranslationRequest(
            source,
            "en",
            "vi",
            1,
            False,
            stream_id=(4, 10),
        )
    )

    assert len(calls) == 2
    assert final.sentence_cache_hits == 1
    assert next_stream.sentence_cache_hits == 0


def test_m2m100_terminology_partial_uses_same_sentence_cache_policy() -> None:
    backend = M2M100Backend(
        TranslationConfig(
            backend="m2m100",
            model="facebook/m2m100_418M",
        )
    )
    backend.configure_terminology(
        _manager(),
        domain="test",
        config=TerminologyMtConfig(),
    )
    calls: list[str] = []
    backend._translate_once = (
        lambda text, source, target: calls.append(text) or text
    )
    stream_id = (5, 1)

    backend.translate(
        TranslationRequest(
            "Outdoor Life. mutable",
            "en",
            "vi",
            1,
            False,
            stream_id=stream_id,
        )
    )
    update = backend.translate(
        TranslationRequest(
            "Outdoor Life. mutable tail",
            "en",
            "vi",
            2,
            False,
            stream_id=stream_id,
        )
    )

    assert calls == [
        "__TERM_0001__.",
        "mutable",
        "mutable tail",
    ]
    assert update.sentence_cache_hits == 1


def test_pipeline_loads_bundle_and_injects_terminology_into_translator() -> None:
    config = PipelineConfig()
    config.asr.backend = "fake"
    config.vad.backend = "passthrough"
    config.translation.backend = "fake"
    config.translation.source_language = "en"
    config.translation.target_language = "vi"
    config.tts.backend = "fake"
    config.terminology.enabled = True
    config.terminology.bundle_path = str(SAMPLE_BUNDLE)
    config.terminology.domain = "factory-safety"

    pipeline = RealtimePipeline(config)
    update = pipeline.translator.translate(
        TranslationRequest("E-stop button", "en", "vi", 1, True)
    )

    assert pipeline.terminology_manager is not None
    assert update.text == "[vi] nút dừng khẩn cấp"
