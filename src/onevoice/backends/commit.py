from __future__ import annotations

from collections import deque

from onevoice.config import CommitConfig
from onevoice.models import AsrUpdate, CommittedTranscript
from onevoice.text import detokenize, longest_common_prefix, tokenize_text


class LocalAgreementCommitter:
    def __init__(self, config: CommitConfig) -> None:
        self.config = config
        self._history: deque[tuple[str, ...]] = deque(maxlen=max(1, config.agreement_updates))
        self._committed: tuple[str, ...] = ()
        self._last_hypothesis: tuple[str, ...] = ()
        self._revision = 0

    def load(self) -> None:
        pass

    def reset(self) -> None:
        self._history.clear()
        self._committed = ()
        self._last_hypothesis = ()
        self._revision = 0

    def close(self) -> None:
        self.reset()

    def update(self, update: AsrUpdate) -> CommittedTranscript | None:
        language = update.language or "en"
        tokens = update.tokens or tokenize_text(update.text, language)
        self._history.append(tokens)

        if update.is_final:
            # A native streaming model can revise an early token during its
            # final pass. Partial commits remain monotonic, but falling back to
            # only the old prefix here would silently discard the entire
            # visible draft suffix. Prefer a full hypothesis that still extends
            # the prefix; otherwise allow the final to replace this utterance.
            if tokens[: len(self._committed)] == self._committed:
                candidate = tokens
            elif self._last_hypothesis[: len(self._committed)] == self._committed:
                candidate = self._last_hypothesis
            elif len(tokens) >= len(self._committed):
                candidate = tokens
            elif len(self._last_hypothesis) >= len(self._committed):
                candidate = self._last_hypothesis
            else:
                candidate = self._committed
        elif len(self._history) < self._history.maxlen:
            self._last_hypothesis = tokens
            return None
        else:
            candidate = longest_common_prefix(tuple(self._history))
            if self.config.hold_tokens:
                candidate = candidate[: max(0, len(candidate) - self.config.hold_tokens)]
            self._last_hypothesis = tokens

        if not update.is_final and candidate[: len(self._committed)] != self._committed:
            candidate = self._committed
        if candidate == self._committed and not update.is_final:
            return None

        self._committed = tuple(candidate)
        self._revision += 1
        result = CommittedTranscript(
            text=detokenize(self._committed, language),
            language=language,
            revision=self._revision,
            is_final=update.is_final,
            tokens=self._committed,
        )
        if update.is_final:
            self._history.clear()
            self._committed = ()
            self._last_hypothesis = ()
        return result
