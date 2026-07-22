from __future__ import annotations

from collections import deque

from onevoice.config import CommitConfig
from onevoice.models import AsrUpdate, CommittedTranscript
from onevoice.text import (
    detokenize,
    ends_phrase,
    longest_common_prefix,
    sentence_token_boundaries,
    tokenize_text,
)


class LocalAgreementCommitter:
    """Lock completed sentences while allowing the active sentence to revise."""

    def __init__(self, config: CommitConfig) -> None:
        self.config = config
        self._history: deque[tuple[str, ...]] = deque(
            maxlen=max(1, config.agreement_updates)
        )
        self._locked: tuple[str, ...] = ()
        self._stable_current: tuple[str, ...] = ()
        self._last_hypothesis: tuple[str, ...] = ()
        self._last_emitted: tuple[str, ...] = ()
        self._revision = 0

    def load(self) -> None:
        pass

    def reset(self) -> None:
        self._history.clear()
        self._locked = ()
        self._stable_current = ()
        self._last_hypothesis = ()
        self._last_emitted = ()
        self._revision = 0

    def close(self) -> None:
        self.reset()

    def update(self, update: AsrUpdate) -> CommittedTranscript | None:
        language = update.language or "en"
        tokens = tuple(update.tokens or tokenize_text(update.text, language))
        current = self._extract_current(tokens)

        if update.is_final:
            final_tokens = self._final_tokens(tokens, current)
            return self._emit(final_tokens, language, is_final=True)

        self._last_hypothesis = tokens
        if current is None:
            return None
        self._history.append(current)
        if len(self._history) < self._history.maxlen:
            return None

        agreed = longest_common_prefix(tuple(self._history))
        stable = agreed
        # A terminal mark that itself survived Local Agreement is useful state,
        # not an unstable lexical tail. Publishing it creates a real endpoint
        # window before the speaker starts the next sentence.
        if self.config.hold_tokens and not ends_phrase(agreed, language):
            stable = stable[: max(0, len(stable) - self.config.hold_tokens)]

        # One disagreeing hypothesis commonly produces an empty LCP. Keep the
        # previous mutable fragment until N newer hypotheses agree instead of
        # retracting it immediately.
        if not stable and self._stable_current and current:
            return None

        self._stable_current = tuple(stable)
        self._lock_completed_current(language)
        output = self._locked + self._stable_current
        if output == self._last_emitted:
            return None
        return self._emit(output, language, is_final=False)

    def _extract_current(self, tokens: tuple[str, ...]) -> tuple[str, ...] | None:
        if not self._locked:
            return tokens
        if tokens[: len(self._locked)] == self._locked:
            return tokens[len(self._locked) :]

        # ASR may rewrite the beginning and even remove the old sentence mark.
        # Align on the longest lexical suffix of the immutable sentence(s).
        locked_lexical = [token for token in self._locked if self._is_lexical(token)]
        token_keys = [token.casefold() for token in tokens]
        for size in range(min(6, len(locked_lexical)), 0, -1):
            anchor = [token.casefold() for token in locked_lexical[-size:]]
            position = self._find_subsequence(token_keys, anchor)
            if position is not None:
                end = position + size
                while end < len(tokens) and tokens[end] in {".", "!", "?", "。", "！", "？"}:
                    end += 1
                return tokens[end:]

        # If the locked sentence has fallen out of the ASR window, retain an
        # anchor from the mutable sentence itself. Include the anchor so its
        # casing/wording can be revised by agreement.
        current_keys = [token.casefold() for token in self._stable_current]
        for size in range(min(4, len(current_keys)), 0, -1):
            position = self._find_subsequence(token_keys, current_keys[:size])
            if position is not None:
                return tokens[position:]
        return None

    def _lock_completed_current(self, language: str) -> None:
        text = detokenize(self._stable_current, language)
        boundaries = sentence_token_boundaries(text, language)
        if not boundaries:
            return
        boundary = boundaries[-1]
        newly_locked = self._stable_current[:boundary]
        self._locked += newly_locked
        self._stable_current = self._stable_current[boundary:]

        remapped: deque[tuple[str, ...]] = deque(maxlen=self._history.maxlen)
        for hypothesis in self._history:
            if hypothesis[:boundary] == newly_locked:
                remapped.append(hypothesis[boundary:])
        self._history = remapped

    def _final_tokens(
        self, tokens: tuple[str, ...], current: tuple[str, ...] | None
    ) -> tuple[str, ...]:
        published = self._locked + self._stable_current
        if tokens[: len(published)] == published:
            return tokens
        if self._last_hypothesis[: len(published)] == published:
            return self._last_hypothesis
        if self._locked and current is not None:
            return self._locked + current
        if len(tokens) >= len(published) and not published:
            return tokens
        if len(self._last_hypothesis) >= len(published):
            return self._last_hypothesis
        return published

    def _emit(
        self, tokens: tuple[str, ...], language: str, *, is_final: bool
    ) -> CommittedTranscript:
        self._last_emitted = tuple(tokens)
        self._revision += 1
        result = CommittedTranscript(
            text=detokenize(tokens, language),
            language=language,
            revision=self._revision,
            is_final=is_final,
            tokens=tuple(tokens),
        )
        if is_final:
            self.reset()
        return result

    @staticmethod
    def _is_lexical(token: str) -> bool:
        return any(character.isalnum() for character in token)

    @staticmethod
    def _find_subsequence(
        values: list[str], wanted: list[str]
    ) -> int | None:
        if not wanted or len(wanted) > len(values):
            return None
        for index in range(len(values) - len(wanted), -1, -1):
            if values[index : index + len(wanted)] == wanted:
                return index
        return None
