from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from time import monotonic

from .config import TranslationConfig, TtsConfig
from .models import CommittedTranscript, TranslationRequest, TranslationUpdate, TtsRequest
from .text import (
    detokenize,
    ends_phrase,
    longest_common_prefix,
    sentence_token_boundaries,
    tokenize_text,
)


class WaitKTranslationPolicy:
    def __init__(self, config: TranslationConfig) -> None:
        self.config = config
        self._last_token_count = 0
        self._last_request_at = monotonic()
        self._has_enqueued = False

    def reset(self) -> None:
        self._last_token_count = 0
        self._last_request_at = monotonic()
        self._has_enqueued = False

    def request_for(
        self, transcript: CommittedTranscript, target_language: str
    ) -> TranslationRequest | None:
        count = len(transcript.tokens)
        now = monotonic()
        if transcript.language == "zh":
            wait_tokens = self.config.zh_wait_tokens
            update_tokens = self.config.zh_update_tokens
            timeout_ms = self.config.zh_timeout_ms
        else:
            wait_tokens = self.config.wait_tokens
            update_tokens = self.config.update_tokens
            timeout_ms = self.config.timeout_ms
        new_tokens = max(0, count - self._last_token_count)
        elapsed_ms = (now - self._last_request_at) * 1000
        enough_context = count >= wait_tokens
        sentence_complete = ends_phrase(transcript.text, transcript.language)
        hybrid_trigger = (
            sentence_complete
            or (
                new_tokens >= update_tokens
                and (
                    not self._has_enqueued
                    or elapsed_ms >= self.config.min_request_interval_ms
                )
            )
            or (new_tokens > 0 and elapsed_ms >= timeout_ms)
        )
        if self.config.sentence_boundary_only:
            hybrid_trigger = sentence_complete
        should_translate = transcript.is_final or (
            enough_context and hybrid_trigger
        )
        if not should_translate or not transcript.text.strip():
            return None
        return TranslationRequest(
            text=transcript.text,
            source_language=transcript.language,
            target_language=target_language,
            source_revision=transcript.revision,
            is_final=transcript.is_final,
        )

    def mark_enqueued(self, request: TranslationRequest) -> None:
        """Publish scheduling state only after the queue accepted the request."""
        self._last_token_count = len(
            tokenize_text(request.text, request.source_language)
        )
        self._last_request_at = monotonic()
        self._has_enqueued = True


StreamId = tuple[int, int]


@dataclass(slots=True)
class _Reservation:
    phrase_id: int
    end_prefix: tuple[str, ...]
    synthesized: bool = False
    accepted: bool = False


@dataclass(slots=True)
class _PhraseState:
    history: deque[tuple[str, ...]]
    acknowledged: tuple[str, ...] = ()
    reservations: list[_Reservation] = field(default_factory=list)
    final_seen: bool = False
    last_reserve_at: float = field(default_factory=monotonic)

    @property
    def protected_prefix(self) -> tuple[str, ...]:
        if self.reservations:
            return self.reservations[-1].end_prefix
        return self.acknowledged


class PhraseTtsPolicy:
    """Reserve translated phrases and commit them only after consumer acknowledgment."""

    def __init__(self, config: TtsConfig) -> None:
        self.config = config
        self._states: dict[StreamId, _PhraseState] = {}
        self._phrase_streams: dict[int, StreamId] = {}
        self._completed_streams: set[StreamId] = set()
        self._completed_stream_order: deque[StreamId] = deque()
        self._next_phrase_id = 1
        self._lock = threading.RLock()

    def reset(self) -> None:
        with self._lock:
            self._states.clear()
            self._phrase_streams.clear()
            self._completed_streams.clear()
            self._completed_stream_order.clear()

    def reset_stream(self, stream_id: StreamId) -> None:
        with self._lock:
            self._completed_streams.discard(stream_id)
            try:
                self._completed_stream_order.remove(stream_id)
            except ValueError:
                pass
            state = self._states.pop(stream_id, None)
            if state is not None:
                for reservation in state.reservations:
                    self._phrase_streams.pop(reservation.phrase_id, None)

    def requests_for(
        self, update: TranslationUpdate, stream_id: StreamId = (0, 0)
    ) -> list[TtsRequest]:
        with self._lock:
            # Final delivery is idempotent. Queue retries or duplicate final MT
            # results must not reopen a stream whose audio was already accepted.
            if stream_id in self._completed_streams:
                return []
            state = self._states.get(stream_id)
            if state is None:
                state = _PhraseState(deque(maxlen=self.config.agreement_updates))
                self._states[stream_id] = state

            tokens = tokenize_text(update.text, update.target_language)
            # Revision numbers alone do not invalidate TTS. Keep a reservation
            # when its exact translated prefix is still present in the newer
            # revision; cancel only the first content-divergent unsynthesized
            # reservation and its dependent suffix.
            self._drop_content_divergent(state, tokens)
            state.history.append(tokens)
            if not tokens:
                if update.is_final and not state.reservations:
                    self._states.pop(stream_id, None)
                return []

            mode = self._emission_mode()
            if mode == "final_utterance" and not update.is_final:
                return []

            if update.is_final:
                stable = tokens
                state.final_seen = True
            elif len(state.history) < self.config.agreement_updates:
                return []
            else:
                stable = longest_common_prefix(tuple(state.history))

            protected = state.protected_prefix
            prefix_matches = stable[: len(protected)] == protected
            if not prefix_matches and not update.is_final:
                return []

            cursor = min(len(protected), len(stable))
            requests: list[TtsRequest] = []
            now = monotonic()
            timed_out = (now - state.last_reserve_at) * 1000 >= self.config.timeout_ms
            absolute_boundaries = sentence_token_boundaries(
                update.text, update.target_language
            )

            while cursor < len(stable):
                remaining = stable[cursor:]
                relative_boundaries = tuple(
                    boundary - cursor
                    for boundary in absolute_boundaries
                    if cursor < boundary <= len(stable)
                )
                endpoint = self._find_endpoint(
                    remaining,
                    update.is_final,
                    timed_out,
                    relative_boundaries,
                    mode,
                )
                if endpoint is None:
                    break
                chunk_tokens = remaining[:endpoint]
                cursor += endpoint
                phrase_id = self._next_phrase_id
                self._next_phrase_id += 1
                is_last_final = update.is_final and cursor == len(stable)
                reservation = _Reservation(
                    phrase_id=phrase_id,
                    end_prefix=stable[:cursor],
                )
                state.reservations.append(reservation)
                self._phrase_streams[phrase_id] = stream_id
                requests.append(
                    TtsRequest(
                        text=detokenize(chunk_tokens, update.target_language),
                        language=update.target_language,
                        source_revision=update.source_revision,
                        is_final=is_last_final,
                        phrase_id=phrase_id,
                        source_is_final=update.is_final,
                    )
                )
                state.last_reserve_at = now
                timed_out = False

            if update.is_final and not state.reservations:
                self._states.pop(stream_id, None)
                self._mark_completed(stream_id)
            return requests

    def is_reserved(self, phrase_id: int) -> bool:
        with self._lock:
            return phrase_id in self._phrase_streams

    def mark_synthesized(self, phrase_id: int) -> bool:
        with self._lock:
            reservation = self._reservation(phrase_id)
            if reservation is None:
                return False
            reservation.synthesized = True
            return True

    def acknowledge(self, phrase_id: int) -> tuple[StreamId, bool] | None:
        """Mark audio accepted by a consumer and advance the emitted prefix."""
        with self._lock:
            stream_id = self._phrase_streams.get(phrase_id)
            reservation = self._reservation(phrase_id)
            if stream_id is None or reservation is None or not reservation.synthesized:
                return None
            reservation.accepted = True
            state = self._states[stream_id]
            while state.reservations and state.reservations[0].accepted:
                completed = state.reservations.pop(0)
                state.acknowledged = completed.end_prefix
                self._phrase_streams.pop(completed.phrase_id, None)
            final_complete = state.final_seen and not state.reservations
            if final_complete:
                self._states.pop(stream_id, None)
                self._mark_completed(stream_id)
            return stream_id, final_complete

    def _mark_completed(self, stream_id: StreamId) -> None:
        if stream_id in self._completed_streams:
            return
        # Only recent stream ids are needed to absorb queue/final retries.
        # Bound the tombstones for long-running microphone sessions.
        if len(self._completed_stream_order) >= 256:
            self._completed_streams.discard(self._completed_stream_order.popleft())
        self._completed_stream_order.append(stream_id)
        self._completed_streams.add(stream_id)

    def cancel(self, phrase_id: int) -> StreamId | None:
        """Cancel a failed reservation and dependent unsynthesized suffixes."""
        with self._lock:
            stream_id = self._phrase_streams.get(phrase_id)
            if stream_id is None:
                return None
            state = self._states.get(stream_id)
            if state is None:
                return None
            index = next(
                (i for i, item in enumerate(state.reservations) if item.phrase_id == phrase_id),
                None,
            )
            if index is None:
                return None
            for item in state.reservations[index:]:
                self._phrase_streams.pop(item.phrase_id, None)
            del state.reservations[index:]
            if not state.reservations:
                self._states.pop(stream_id, None)
            return stream_id

    def _reservation(self, phrase_id: int) -> _Reservation | None:
        stream_id = self._phrase_streams.get(phrase_id)
        state = self._states.get(stream_id) if stream_id is not None else None
        if state is None:
            return None
        return next(
            (item for item in state.reservations if item.phrase_id == phrase_id), None
        )

    def _drop_content_divergent(
        self, state: _PhraseState, tokens: tuple[str, ...]
    ) -> None:
        first_invalid = next(
            (
                index
                for index, item in enumerate(state.reservations)
                if not item.synthesized
                and tokens[: len(item.end_prefix)] != item.end_prefix
            ),
            None,
        )
        if first_invalid is None:
            return
        for item in state.reservations[first_invalid:]:
            self._phrase_streams.pop(item.phrase_id, None)
        del state.reservations[first_invalid:]

    def _emission_mode(self) -> str:
        if self.config.emission_mode is not None:
            return self.config.emission_mode
        if self.config.final_only:
            return "final_utterance"
        if self.config.sentence_boundary_only:
            return "stable_sentence"
        return "stable_phrase"

    def _find_endpoint(
        self,
        tokens: tuple[str, ...],
        is_final: bool,
        timed_out: bool,
        sentence_boundaries: tuple[int, ...],
        mode: str,
    ) -> int | None:
        maximum = min(len(tokens), self.config.max_chunk_tokens)
        sentence_end = next(iter(sentence_boundaries), None)
        # A complete short sentence is meaningful and may be shorter than the
        # configured minimum. The configured maximum remains a hard limit.
        if sentence_end is not None and sentence_end <= maximum:
            return sentence_end
        if mode == "stable_sentence" and not is_final and sentence_end is None:
            return None
        if sentence_end == maximum + 1 and maximum - 1 >= self.config.min_chunk_tokens:
            return maximum - 1
        if len(tokens) > maximum:
            for index in range(maximum, self.config.min_chunk_tokens - 1, -1):
                if tokens[index - 1] in {",", ";", ":"}:
                    return index
            return maximum
        if len(tokens) == maximum and mode == "stable_phrase":
            return maximum
        if is_final:
            return len(tokens)
        if mode == "stable_sentence":
            return None
        if mode == "stable_phrase" and (
            timed_out and len(tokens) >= self.config.min_chunk_tokens
        ):
            return len(tokens)
        return None
