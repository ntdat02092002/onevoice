from __future__ import annotations

from onevoice.models import AudioChunk


class PassthroughPreprocessor:
    def load(self) -> None:
        pass

    def reset(self) -> None:
        pass

    def close(self) -> None:
        pass

    def process(self, chunk: AudioChunk) -> AudioChunk:
        return chunk

