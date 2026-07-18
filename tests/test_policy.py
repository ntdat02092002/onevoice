from onevoice.config import TranslationConfig
from onevoice.models import CommittedTranscript
from onevoice.policy import WaitKTranslationPolicy


def transcript(tokens: tuple[str, ...], revision: int, final: bool = False) -> CommittedTranscript:
    return CommittedTranscript(" ".join(tokens), "en", revision, final, tokens)


def test_wait_k_threshold_and_final_flush() -> None:
    policy = WaitKTranslationPolicy(
        TranslationConfig(wait_tokens=3, update_tokens=3, target_language="vi")
    )
    assert policy.request_for(transcript(("one", "two"), 1), "vi") is None
    first = policy.request_for(transcript(("one", "two", "three"), 2), "vi")
    assert first is not None
    assert not first.is_final
    assert policy.request_for(transcript(("one", "two", "three", "four"), 3), "vi") is None
    final = policy.request_for(transcript(("done",), 4, final=True), "vi")
    assert final is not None
    assert final.is_final

