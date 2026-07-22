from __future__ import annotations

import io
import wave
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic
from typing import BinaryIO

import av
import numpy as np

from .models import AudioChunk


@dataclass(slots=True)
class RealtimePacer:
    """Pace media against an absolute timeline without accumulating sleep drift."""

    started_at: float = field(default_factory=monotonic)
    media_elapsed: float = 0.0

    def delay_after(self, duration_seconds: float, now: float | None = None) -> float:
        self.media_elapsed += max(0.0, duration_seconds)
        current = monotonic() if now is None else now
        return max(0.0, self.started_at + self.media_elapsed - current)


def encode_wav(samples: np.ndarray, sample_rate: int) -> bytes:
    """Encode mono float32 samples as a downloadable 16-bit PCM WAV."""
    if samples.ndim != 1:
        raise ValueError("WAV samples must be mono")
    if sample_rate <= 0:
        raise ValueError("WAV sample_rate must be positive")
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
    return output.getvalue()


def _frame_to_float32(frame: av.AudioFrame) -> np.ndarray:
    values = frame.to_ndarray().reshape(-1)
    if np.issubdtype(values.dtype, np.integer):
        maximum = float(max(abs(np.iinfo(values.dtype).min), np.iinfo(values.dtype).max))
        return values.astype(np.float32) / maximum
    return values.astype(np.float32, copy=False)


class AudioFrameNormalizer:
    """Stateful PyAV frame resampler used by the WebRTC callback."""

    def __init__(self, sample_rate: int = 16_000) -> None:
        self.sample_rate = sample_rate
        self._resampler = av.AudioResampler(format="s16", layout="mono", rate=sample_rate)
        self._sequence = 0

    def process(self, frame: av.AudioFrame) -> list[AudioChunk]:
        output: list[AudioChunk] = []
        for converted in self._resampler.resample(frame):
            samples = _frame_to_float32(converted)
            output.append(AudioChunk(samples, self.sample_rate, self._sequence, monotonic()))
            self._sequence += 1
        return output


def iter_audio_file(
    source: str | Path | BinaryIO,
    sample_rate: int = 16_000,
    chunk_ms: int = 20,
) -> Iterator[AudioChunk]:
    """Decode any PyAV-supported audio file into fixed mono float32 chunks."""

    resampler = av.AudioResampler(format="s16", layout="mono", rate=sample_rate)
    chunk_samples = sample_rate * chunk_ms // 1000
    pending = np.empty(0, dtype=np.float32)
    sequence = 0
    with av.open(source, mode="r") as container:
        for frame in container.decode(audio=0):
            for converted in resampler.resample(frame):
                pending = np.concatenate((pending, _frame_to_float32(converted)))
                while len(pending) >= chunk_samples:
                    values = pending[:chunk_samples].copy()
                    pending = pending[chunk_samples:]
                    yield AudioChunk(values, sample_rate, sequence)
                    sequence += 1
        for converted in resampler.resample(None):
            pending = np.concatenate((pending, _frame_to_float32(converted)))
    if pending.size:
        yield AudioChunk(pending.copy(), sample_rate, sequence)

