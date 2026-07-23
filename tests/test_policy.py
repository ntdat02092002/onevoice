from onevoice.config import TranslationConfig
from onevoice.models import CommittedTranscript
from onevoice.policy import WaitKTranslationPolicy


def transcript(tokens: tuple[str, ...], revision: int, final: bool = False) -> CommittedTranscript:
    return CommittedTranscript(" ".join(tokens), "en", revision, final, tokens)


def test_wait_k_threshold_and_final_flush() -> None:
    policy = WaitKTranslationPolicy(
        TranslationConfig(
            wait_tokens=3,
            update_tokens=3,
            target_language="vi",
            sentence_boundary_only=False,
        )
    )
    assert policy.request_for(transcript(("one", "two"), 1), "vi") is None
    first = policy.request_for(transcript(("one", "two", "three"), 2), "vi")
    assert first is not None
    assert not first.is_final
    policy.mark_enqueued(first)
    assert policy.request_for(transcript(("one", "two", "three", "four"), 3), "vi") is None
    final = policy.request_for(transcript(("done",), 4, final=True), "vi")
    assert final is not None
    assert final.is_final


def test_wait_k_semantic_mode_skips_unfinished_partial_sentence() -> None:
    policy = WaitKTranslationPolicy(
        TranslationConfig(
            wait_tokens=3,
            update_tokens=3,
            target_language="vi",
            sentence_boundary_only=True,
        )
    )
    assert policy.request_for(transcript(("one", "two", "three"), 1), "vi") is None
    request = policy.request_for(transcript(("one", "two", "three", "."), 2), "vi")
    assert request is not None
    assert not request.is_final


def test_hybrid_wait_k_uses_interval_timeout_and_immediate_sentence_boundary(
    monkeypatch,
) -> None:
    clock = [0.0]
    monkeypatch.setattr("onevoice.policy.monotonic", lambda: clock[0])
    policy = WaitKTranslationPolicy(
        TranslationConfig(
            wait_tokens=6,
            update_tokens=4,
            min_request_interval_ms=500,
            timeout_ms=1200,
            sentence_boundary_only=False,
        )
    )
    first = policy.request_for(transcript(tuple("abcdef"), 1), "vi")
    assert first is not None
    policy.mark_enqueued(first)

    clock[0] = 0.2
    assert policy.request_for(transcript(tuple("abcdefghij"), 2), "vi") is None
    clock[0] = 0.5
    interval_request = policy.request_for(transcript(tuple("abcdefghij"), 2), "vi")
    assert interval_request is not None
    policy.mark_enqueued(interval_request)

    clock[0] = 0.6
    sentence_request = policy.request_for(
        transcript((*tuple("abcdefghij"), "next", "."), 3), "vi"
    )
    assert sentence_request is not None
    policy.mark_enqueued(sentence_request)

    clock[0] = 1.0
    assert policy.request_for(
        transcript((*tuple("abcdefghij"), "next", ".", "draft"), 4), "vi"
    ) is None
    clock[0] = 1.9
    timeout_request = policy.request_for(
        transcript((*tuple("abcdefghij"), "next", ".", "draft"), 4), "vi"
    )
    assert timeout_request is not None

