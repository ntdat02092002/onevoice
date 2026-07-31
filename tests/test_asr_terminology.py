from __future__ import annotations

from pathlib import Path
from time import monotonic

from onevoice.backends.asr import FasterWhisperBackend
from onevoice.backends.asr import FakeAsrBackend
from onevoice.config import AsrConfig, PipelineConfig
from onevoice.models import (
    AsrUpdate,
    AsrWordTiming,
    EventType,
    SpeechSegment,
)
from onevoice.pipeline import RealtimePipeline, _AsrJob
from onevoice.terminology import TerminologyManager
from onevoice.terminology.runtime import (
    TerminologyAsrCorrector,
    TerminologyAsrRuntime,
)


SAMPLE_BUNDLE = Path(
    "assets/terminology/factory-sample-v1/terminology.yaml"
)


def _manager() -> TerminologyManager:
    return TerminologyManager.from_path(SAMPLE_BUNDLE)


def _update(
    text: str,
    *,
    words: tuple[AsrWordTiming, ...] = (),
) -> AsrUpdate:
    return AsrUpdate(
        text=text,
        language="en",
        confidence=0.9,
        revision=3,
        is_final=False,
        started_at=monotonic(),
        words=words,
    )


def test_exact_alias_correction_remaps_multiword_timing() -> None:
    profile = _manager().activate(
        domain="test",
        source_language="en",
        target_language="en",
    )
    corrector = TerminologyAsrCorrector(profile)
    update = _update(
        "Try wind surfing today.",
        words=(
            AsrWordTiming("Try", 0.0, 0.2, 0.9),
            AsrWordTiming("wind", 0.3, 0.5, 0.8),
            AsrWordTiming("surfing", 0.5, 0.9, 0.7),
            AsrWordTiming("today", 1.0, 1.2, 0.9),
        ),
    )

    corrected, stats = corrector.correct(update)

    assert corrected.text == "Try windsurfing today."
    assert [word.text for word in corrected.words] == [
        "Try",
        "windsurfing",
        "today",
    ]
    assert corrected.words[1].start_seconds == 0.3
    assert corrected.words[1].end_seconds == 0.9
    assert corrected.words[1].confidence == 0.7
    assert stats.corrections == 1
    assert stats.timing_drops == 0


def test_declared_typo_is_corrected_but_near_acoustic_negative_is_not() -> None:
    runtime = TerminologyAsrRuntime(_manager(), domain="test")

    typo, typo_stats = runtime.correct(
        _update("Try winssurfing today.")
    )
    negative, negative_stats = runtime.correct(
        _update("Try wind serving today.")
    )

    assert typo.text == "Try windsurfing today."
    assert typo_stats.corrections == 1
    assert negative.text == "Try wind serving today."
    assert negative_stats.corrections == 0


def test_canonical_and_out_of_domain_terms_are_not_rewritten() -> None:
    test_runtime = TerminologyAsrRuntime(_manager(), domain="test")
    maintenance_runtime = TerminologyAsrRuntime(
        _manager(), domain="factory-maintenance"
    )

    canonical, canonical_stats = test_runtime.correct(
        _update("Outdoor Life and windsurfing.")
    )
    outside, outside_stats = test_runtime.correct(
        _update("Open M 5 Stack.")
    )
    in_domain, in_domain_stats = maintenance_runtime.correct(
        _update("Open M 5 Stack.")
    )

    assert canonical.text == "Outdoor Life and windsurfing."
    assert canonical_stats.corrections == 0
    assert outside.text == "Open M 5 Stack."
    assert outside_stats.corrections == 0
    assert in_domain.text == "Open M5Stack."
    assert in_domain_stats.corrections == 1


def test_prompt_compiler_obeys_term_and_token_budgets() -> None:
    runtime = TerminologyAsrRuntime(
        _manager(), domain="factory-maintenance"
    )

    terms = runtime.prompt_terms(
        "en",
        max_terms=2,
        max_tokens=4,
    )

    assert 0 < len(terms) <= 2
    assert sum(len(term.split()) for term in terms) <= 4


def test_hotword_compiler_applies_model_case_scores_and_budgets() -> None:
    runtime = TerminologyAsrRuntime(_manager(), domain="test")

    hotwords = runtime.hotwords(
        "en",
        max_terms=3,
        max_tokens=4,
        text_case="upper",
    )

    assert 0 < len(hotwords) <= 3
    assert sum(item.token_count for item in hotwords) <= 4
    assert all(item.text == item.text.upper() for item in hotwords)
    assert all(item.score > 0 for item in hotwords)
    assert {item.text for item in hotwords}.intersection(
        {"WINDSURFING", "WIND SURFING", "WINSSURFING"}
    )


def test_faster_whisper_receives_compiled_initial_prompt() -> None:
    backend = FasterWhisperBackend(
        AsrConfig(
            backend="faster_whisper",
            model="tiny",
            language="en",
        )
    )
    captured = {}

    class FakeWhisper:
        def transcribe(self, samples, **kwargs):
            captured.update(kwargs)
            return iter(()), type(
                "Info",
                (),
                {"language": "en", "language_probability": 1.0},
            )()

    backend._model = FakeWhisper()
    backend.configure_terminology_prompt(
        ("windsurfing", "Outdoor Life")
    )
    backend.transcribe(
        SpeechSegment(
            samples=__import__("numpy").zeros(160, dtype="float32"),
            sample_rate=16_000,
            started_at=monotonic(),
            ended_at=monotonic(),
        ),
        "en",
    )

    assert captured["initial_prompt"] == (
        "windsurfing; Outdoor Life"
    )
    metrics = backend.take_metrics()
    assert metrics["asr_prompt_term_count"] == 2


def test_pipeline_applies_post_correction_before_commit_and_mt() -> None:
    config = PipelineConfig()
    config.asr.backend = "fake"
    config.asr.model = "fake"
    config.asr.language = "en"
    config.vad.backend = "passthrough"
    config.commit.agreement_updates = 1
    config.commit.hold_tokens = 0
    config.translation.backend = "fake"
    config.translation.model = "fake"
    config.translation.source_language = "en"
    config.translation.target_language = "vi"
    config.tts.backend = "fake"
    config.terminology.enabled = True
    config.terminology.bundle_path = str(SAMPLE_BUNDLE)
    config.terminology.domain = "test"
    asr = FakeAsrBackend(config.asr, script=["wind surfing."])
    pipeline = RealtimePipeline(config, asr=asr)
    pipeline.start(load_models=False)
    try:
        pipeline._asr_queue.put(
            _AsrJob(
                0,
                1,
                SpeechSegment(
                    samples=__import__("numpy").zeros(
                        160, dtype="float32"
                    ),
                    sample_rate=16_000,
                    started_at=monotonic(),
                    ended_at=monotonic(),
                    is_final=True,
                ),
            )
        )
        assert pipeline.wait_until_idle(timeout=2)
        events = pipeline.poll_events(100)
    finally:
        pipeline.close()

    asr_final = next(
        event.payload
        for event in events
        if event.type == EventType.ASR_FINAL
    )
    mt_final = next(
        event.payload
        for event in events
        if event.type == EventType.TRANSLATION_FINAL
    )
    assert asr_final.text == "windsurfing."
    assert mt_final.source_text == "windsurfing."
    assert pipeline.metrics_snapshot()["asr_post_correction_count"] == 1


def test_pipeline_activates_sherpa_hotwords_and_modified_beam() -> None:
    config = PipelineConfig()
    config.asr.backend = "sherpa_onnx"
    config.asr.model = "auto"
    config.asr.language = "en"
    config.translation.backend = "fake"
    config.translation.model = "fake"
    config.translation.source_language = "en"
    config.translation.target_language = "vi"
    config.tts.backend = "fake"
    config.terminology.enabled = True
    config.terminology.bundle_path = str(SAMPLE_BUNDLE)
    config.terminology.domain = "test"

    pipeline = RealtimePipeline(config)

    assert (
        config.asr.sherpa.decoding_method
        == "modified_beam_search"
    )
    assert "WINDSURFING" in pipeline.asr._hotwords
    assert "OUTDOOR LIFE" in pipeline.asr._hotwords
    assert (
        pipeline.metrics_snapshot()["asr_hotword_auto_beam_switch"]
        == 1
    )
