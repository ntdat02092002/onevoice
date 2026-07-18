from __future__ import annotations

from time import monotonic

from .config import TranslationConfig
from .models import CommittedTranscript, TranslationRequest
from .text import ends_phrase


class WaitKTranslationPolicy:
    def __init__(self, config: TranslationConfig) -> None:
        self.config = config
        self._last_token_count = 0
        self._last_request_at = monotonic()

    def reset(self) -> None:
        self._last_token_count = 0
        self._last_request_at = monotonic()

    def request_for(
        self, transcript: CommittedTranscript, target_language: str
    ) -> TranslationRequest | None:
        count = len(transcript.tokens)
        now = monotonic()
        enough_context = count >= self.config.wait_tokens
        enough_new = count - self._last_token_count >= self.config.update_tokens
        timed_out = (now - self._last_request_at) * 1000 >= self.config.timeout_ms
        should_translate = transcript.is_final or (
            enough_context and (enough_new or timed_out or ends_phrase(transcript.tokens))
        )
        if not should_translate or not transcript.text.strip():
            return None
        self._last_token_count = count
        self._last_request_at = now
        return TranslationRequest(
            text=transcript.text,
            source_language=transcript.language,
            target_language=target_language,
            source_revision=transcript.revision,
            is_final=transcript.is_final,
        )

