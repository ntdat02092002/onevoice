from time import monotonic
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from onevoice.backends.tts import AUTO_TTS_VOICES, FakeTtsBackend, SherpaOnnxTtsBackend
from onevoice.config import TtsConfig
from onevoice.models import TranslationUpdate, TtsRequest
from onevoice.policy import PhraseTtsPolicy
from onevoice.terminology import TerminologyManager
from onevoice.terminology.runtime import TerminologyTtsNormalizer
from onevoice.text import tokenize_text


def translation(
    text: str, revision: int, *, final: bool = False, target: str = "en"
) -> TranslationUpdate:
    return TranslationUpdate(
        text=text,
        source_text="source",
        source_language="vi",
        target_language=target,
        source_revision=revision,
        is_final=final,
        started_at=monotonic(),
    )


SAMPLE_BUNDLE = Path(
    "assets/terminology/factory-sample-v1/terminology.yaml"
)


def terminology_tts_policy(
    *,
    min_tokens: int,
    max_tokens: int,
) -> PhraseTtsPolicy:
    policy = PhraseTtsPolicy(
        TtsConfig(
            min_chunk_tokens=min_tokens,
            max_chunk_tokens=max_tokens,
        )
    )
    policy.configure_terminology(
        TerminologyManager.from_path(SAMPLE_BUNDLE),
        domain="factory-safety",
    )
    return policy


def test_phrase_policy_waits_for_agreement_and_never_replays_prefix() -> None:
    policy = PhraseTtsPolicy(
        TtsConfig(
            min_chunk_tokens=2,
            max_chunk_tokens=4,
            agreement_updates=2,
            sentence_boundary_only=False,
            final_only=False,
        )
    )
    assert policy.requests_for(translation("wear a safety helmet", 1)) == []
    first = policy.requests_for(translation("wear a safety helmet now", 2))
    assert [request.text for request in first] == ["wear a safety helmet"]
    assert policy.mark_synthesized(first[0].phrase_id)
    final = policy.requests_for(translation("wear a safety helmet now.", 3, final=True))
    assert [request.text for request in final] == ["now."]
    assert final[-1].is_final
    assert policy.acknowledge(first[0].phrase_id) is not None
    assert policy.mark_synthesized(final[0].phrase_id)
    assert policy.acknowledge(final[0].phrase_id) == ((0, 0), True)


def test_phrase_is_not_emitted_until_synthesis_and_consumer_ack() -> None:
    policy = PhraseTtsPolicy(
        TtsConfig(
            min_chunk_tokens=2,
            max_chunk_tokens=4,
            agreement_updates=1,
            sentence_boundary_only=False,
            final_only=False,
        )
    )
    reserved = policy.requests_for(translation("one two three four", 1))
    assert len(reserved) == 1

    # Queue/synthesis failure releases the reservation, so a newer final can
    # reserve the complete text again instead of silently losing it.
    policy.cancel(reserved[0].phrase_id)
    retried = policy.requests_for(translation("one two three four.", 2, final=True))
    assert [request.text for request in retried] == ["one two three", "four."]


def test_phrase_policy_flushes_short_final_and_handles_chinese() -> None:
    policy = PhraseTtsPolicy(TtsConfig(min_chunk_tokens=4, max_chunk_tokens=12))
    requests = policy.requests_for(translation("请停机。", 1, final=True, target="zh"))
    assert [request.text for request in requests] == ["请停机。"]


def test_phrase_policy_waits_for_sentence_boundary_in_semantic_mode() -> None:
    policy = PhraseTtsPolicy(
        TtsConfig(
            min_chunk_tokens=2,
            max_chunk_tokens=4,
            agreement_updates=1,
            final_only=False,
        )
    )
    assert policy.requests_for(translation("one two three four", 1)) == []
    requests = policy.requests_for(translation("one two three four.", 2))
    assert [request.text for request in requests] == ["one two three", "four."]


def test_default_tts_policy_waits_for_final_translation() -> None:
    policy = PhraseTtsPolicy(TtsConfig(min_chunk_tokens=2, max_chunk_tokens=8))
    assert policy.requests_for(translation("complete sentence.", 1)) == []
    requests = policy.requests_for(translation("complete sentence.", 2, final=True))
    assert [request.text for request in requests] == ["complete sentence."]


def test_tts_final_never_exceeds_hard_max_without_punctuation() -> None:
    policy = PhraseTtsPolicy(TtsConfig(min_chunk_tokens=8, max_chunk_tokens=24))
    text = " ".join(f"token{index}" for index in range(46))

    requests = policy.requests_for(translation(text, 1, final=True))

    sizes = [len(request.text.split()) for request in requests]
    assert sizes == [24, 22]
    assert max(sizes) <= 24


def test_tts_chunker_short_sentence_long_sentence_and_comma_fallback() -> None:
    policy = PhraseTtsPolicy(TtsConfig(min_chunk_tokens=8, max_chunk_tokens=24))
    long_words = [f"word{index}" for index in range(30)]
    long_words[19] += ","
    text = "Short complete sentence. " + " ".join(long_words) + "."

    requests = policy.requests_for(translation(text, 1, final=True))

    assert requests[0].text == "Short complete sentence."
    assert all(len(tokenize_text(request.text, "en")) <= 24 for request in requests)
    assert requests[1].text.endswith(",")


def test_term_aware_tts_chunker_moves_boundary_before_whole_term() -> None:
    policy = terminology_tts_policy(min_tokens=2, max_tokens=4)

    requests = policy.requests_for(
        translation(
            "one two emergency stop button now",
            1,
            final=True,
        )
    )

    assert [request.text for request in requests] == [
        "one two",
        "emergency stop button now",
    ]
    assert policy.take_metrics()["tts_term_boundary_shifts"] == 1


def test_term_longer_than_hard_max_is_emitted_as_exceptional_chunk() -> None:
    policy = terminology_tts_policy(min_tokens=1, max_tokens=2)

    requests = policy.requests_for(
        translation("emergency stop button now", 1, final=True)
    )

    assert [request.text for request in requests] == [
        "emergency stop button",
        "now",
    ]
    metrics = policy.take_metrics()
    assert metrics["tts_term_boundary_shifts"] == 1
    assert metrics["tts_oversized_protected_spans"] == 1


def test_tts_normalizer_replaces_display_terms_and_preserves_missing_form() -> None:
    manager = TerminologyManager.from_path(SAMPLE_BUNDLE)
    profile = manager.activate(
        domain="factory-maintenance",
        source_language="vi",
        target_language="vi",
    )
    normalizer = TerminologyTtsNormalizer(profile)
    display = "Kiểm tra M5Stack, PLC S7-1200 và EtherCAT."

    result = normalizer.normalize(display)

    assert result.display_text == display
    assert result.spoken_text == (
        "Kiểm tra em năm stack, pi eo xi ét bảy một hai không không "
        "và EtherCAT."
    )
    assert result.substitutions == 2


@pytest.mark.parametrize(
    ("language", "display", "spoken"),
    [
        ("en", "Use M5Stack.", "Use M five stack."),
        ("zh", "使用 M5Stack。", "使用 M 五 Stack。"),
        ("ko", "M5Stack 사용.", "엠 파이브 스택 사용."),
    ],
)
def test_tts_normalizer_preserves_locale_spacing(
    language: str,
    display: str,
    spoken: str,
) -> None:
    profile = TerminologyManager.from_path(SAMPLE_BUNDLE).activate(
        domain="factory-maintenance",
        source_language=language,
        target_language=language,
    )

    assert TerminologyTtsNormalizer(profile).normalize(
        display
    ).spoken_text == spoken


def test_phrase_request_separates_display_and_spoken_text() -> None:
    policy = PhraseTtsPolicy(TtsConfig(min_chunk_tokens=1))
    policy.configure_terminology(
        TerminologyManager.from_path(SAMPLE_BUNDLE),
        domain="factory-maintenance",
    )

    request = policy.requests_for(
        translation(
            "Kiểm tra M5Stack.",
            1,
            final=True,
            target="vi",
        )
    )[0]

    assert request.text == "Kiểm tra M5Stack."
    assert request.spoken_text == "Kiểm tra em năm stack."
    assert request.synthesis_text == request.spoken_text
    metrics = policy.take_metrics()
    assert metrics["tts_spoken_form_requests"] == 1
    assert metrics["tts_spoken_form_substitutions"] == 1


def test_term_without_spoken_form_falls_back_to_display_text() -> None:
    policy = PhraseTtsPolicy(TtsConfig(min_chunk_tokens=1))
    policy.configure_terminology(
        TerminologyManager.from_path(SAMPLE_BUNDLE),
        domain="factory-maintenance",
    )

    request = policy.requests_for(
        translation("EtherCAT.", 1, final=True, target="vi")
    )[0]

    assert request.spoken_text is None
    assert request.synthesis_text == "EtherCAT."


def test_sample_profile_exposes_spoken_forms_for_app_testing() -> None:
    profile = TerminologyManager.from_path(SAMPLE_BUNDLE).activate(
        domain="test",
        source_language="vi",
        target_language="vi",
    )
    result = TerminologyTtsNormalizer(profile).normalize(
        "windsurfing tại Outdoor Life."
    )

    assert result.display_text == "windsurfing tại Outdoor Life."
    assert result.spoken_text == (
        "lướt ván buồm tại ao đờ lai-ph."
    )
    assert result.substitutions == 2


def test_cancel_later_final_phrase_does_not_orphan_earlier_reservation() -> None:
    policy = PhraseTtsPolicy(TtsConfig(min_chunk_tokens=2, max_chunk_tokens=4))
    requests = policy.requests_for(
        translation("one two three four five six", 1, final=True)
    )
    assert len(requests) == 2
    first, second = requests
    assert policy.mark_synthesized(first.phrase_id)

    policy.cancel(second.phrase_id)

    assert policy.is_reserved(first.phrase_id)
    assert not policy.is_reserved(second.phrase_id)
    assert policy.acknowledge(first.phrase_id) == ((0, 0), True)
    assert not policy.is_reserved(first.phrase_id)


def test_stable_sentence_mode_waits_for_two_identical_revisions() -> None:
    policy = PhraseTtsPolicy(
        TtsConfig(
            emission_mode="stable_sentence",
            agreement_updates=2,
            min_chunk_tokens=2,
            max_chunk_tokens=12,
            final_only=True,
        )
    )
    assert policy.requests_for(translation("A stable sentence.", 1)) == []
    requests = policy.requests_for(translation("A stable sentence.", 2))
    assert [request.text for request in requests] == ["A stable sentence."]


def test_stable_sentence_reservation_survives_new_revision_when_content_matches() -> None:
    policy = PhraseTtsPolicy(
        TtsConfig(
            emission_mode="stable_sentence",
            agreement_updates=1,
            min_chunk_tokens=2,
            max_chunk_tokens=12,
        )
    )
    request = policy.requests_for(translation("Keep this sentence.", 1))[0]

    policy.requests_for(translation("Keep this sentence. next draft", 2))

    assert policy.is_reserved(request.phrase_id)


def test_stable_sentence_reservation_is_cancelled_when_content_diverges() -> None:
    policy = PhraseTtsPolicy(
        TtsConfig(
            emission_mode="stable_sentence",
            agreement_updates=1,
            min_chunk_tokens=2,
            max_chunk_tokens=12,
        )
    )
    request = policy.requests_for(translation("Old stable sentence.", 1))[0]

    policy.requests_for(translation("Changed stable sentence.", 2))

    assert not policy.is_reserved(request.phrase_id)


def test_stable_sentence_does_not_emit_long_unfinished_partial() -> None:
    policy = PhraseTtsPolicy(
        TtsConfig(
            emission_mode="stable_sentence",
            agreement_updates=2,
            min_chunk_tokens=8,
            max_chunk_tokens=24,
        )
    )
    text = " ".join(f"token{index}" for index in range(30))
    assert policy.requests_for(translation(text, 1)) == []
    assert policy.requests_for(translation(text, 2)) == []


def test_stable_sentence_splits_completed_long_sentence_at_hard_max() -> None:
    policy = PhraseTtsPolicy(
        TtsConfig(
            emission_mode="stable_sentence",
            agreement_updates=2,
            min_chunk_tokens=8,
            max_chunk_tokens=24,
        )
    )
    text = " ".join(f"token{index}" for index in range(35)) + "."
    assert policy.requests_for(translation(text, 1)) == []
    requests = policy.requests_for(translation(text, 2))

    sizes = [len(tokenize_text(request.text, "en")) for request in requests]
    assert sizes == [24, 12]
    assert max(sizes) <= 24


def test_stable_sentence_final_flushes_unpunctuated_tail() -> None:
    policy = PhraseTtsPolicy(
        TtsConfig(
            emission_mode="stable_sentence",
            agreement_updates=2,
            min_chunk_tokens=8,
            max_chunk_tokens=24,
        )
    )
    requests = policy.requests_for(
        translation("unfinished final fragment", 1, final=True)
    )
    assert [request.text for request in requests] == ["unfinished final fragment"]


def test_stable_sentence_does_not_repeat_acknowledged_prefix_on_final() -> None:
    policy = PhraseTtsPolicy(
        TtsConfig(
            emission_mode="stable_sentence",
            agreement_updates=2,
            min_chunk_tokens=2,
            max_chunk_tokens=24,
        )
    )
    assert policy.requests_for(translation("Read this once.", 1)) == []
    partial = policy.requests_for(translation("Read this once.", 2))
    assert [request.text for request in partial] == ["Read this once."]
    assert policy.mark_synthesized(partial[0].phrase_id)
    assert policy.acknowledge(partial[0].phrase_id) is not None

    assert policy.requests_for(translation("Read this once.", 3, final=True)) == []
    assert policy.requests_for(translation("Read this once.", 3, final=True)) == []


def test_fake_tts_contract_and_lifecycle() -> None:
    backend = FakeTtsBackend(TtsConfig(backend="fake"))
    backend.load()
    output = backend.synthesize(TtsRequest("hello", "en", 1, True))
    assert output.samples.dtype == np.float32
    assert output.samples.ndim == 1
    assert output.sample_rate == 16_000
    assert output.duration_seconds > 0
    backend.reset()
    backend.close()


def test_fake_tts_synthesizes_spoken_text_but_reports_display_text() -> None:
    backend = FakeTtsBackend(TtsConfig(backend="fake"))
    request = TtsRequest(
        "M5Stack",
        "vi",
        1,
        True,
        spoken_text="em năm stack",
    )

    output = backend.synthesize(request)

    assert output.text == "M5Stack"
    assert output.spoken_text == "em năm stack"


def test_sherpa_tts_passes_spoken_text_to_native_generator(
    monkeypatch,
) -> None:
    seen: list[str] = []

    class GenerationConfig:
        def __init__(self) -> None:
            self.sid = 0
            self.speed = 1.0
            self.num_steps = 0
            self.extra = {}

    class NativeTts:
        def generate(self, text, generation):
            seen.append(text)
            return SimpleNamespace(
                samples=np.ones(160, dtype=np.float32),
                sample_rate=16_000,
            )

    monkeypatch.setitem(
        __import__("sys").modules,
        "sherpa_onnx",
        SimpleNamespace(GenerationConfig=GenerationConfig),
    )
    backend = SherpaOnnxTtsBackend(TtsConfig(language="vi"))
    backend._tts = NativeTts()

    output = backend.synthesize(
        TtsRequest(
            "M5Stack",
            "vi",
            1,
            True,
            spoken_text="em năm stack",
        )
    )

    assert seen == ["em năm stack"]
    assert output.text == "M5Stack"
    assert output.spoken_text == "em năm stack"


def test_sherpa_backend_fails_before_import_when_assets_are_missing(tmp_path) -> None:
    backend = SherpaOnnxTtsBackend(
        TtsConfig(model="model.onnx", model_dir=str(tmp_path))
    )
    with pytest.raises(FileNotFoundError, match="TTS model"):
        backend.load()


def test_auto_tts_catalog_covers_product_languages() -> None:
    assert set(AUTO_TTS_VOICES) == {"vi", "en", "zh", "ko"}


def test_auto_tts_offline_error_names_missing_cache(tmp_path) -> None:
    backend = SherpaOnnxTtsBackend(
        TtsConfig(language="vi", cache_dir=str(tmp_path), offline=True)
    )
    with pytest.raises(FileNotFoundError, match="disable offline mode"):
        backend._auto_assets()
