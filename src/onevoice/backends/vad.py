from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import threading
from time import monotonic

import numpy as np

from onevoice.config import AudioConfig, VadConfig
from onevoice.models import AudioChunk, SpeechSegment


def _float_to_pcm16(samples: np.ndarray) -> bytes:
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2", copy=False).tobytes()


@dataclass(frozen=True, slots=True)
class _EndpointRequest:
    started_at: float | None
    cut_sample: int | None


class WebRtcVadBackend:
    """WebRTC VAD plus utterance assembly and periodic partial snapshots."""

    def __init__(self, config: VadConfig, audio_config: AudioConfig) -> None:
        self.config = config
        self.audio_config = audio_config
        self._vad = None
        self._endpoint_requested = threading.Event()
        self._endpoint_lock = threading.Lock()
        self._endpoint_request: _EndpointRequest | None = None
        self._pending = np.empty(0, dtype=np.float32)
        self._silence_history: deque[np.ndarray] = deque()
        self._candidate: list[np.ndarray] = []
        self._utterance: list[np.ndarray] = []
        self._active = False
        self._silence_frames = 0
        self._frames_since_emit = 0
        self._started_at = monotonic()
        self._frame_samples = audio_config.sample_rate * audio_config.frame_ms // 1000
        self._padding_frames = max(1, config.speech_padding_ms // audio_config.frame_ms)
        self._min_speech_frames = max(1, config.min_speech_ms // audio_config.frame_ms)
        self._end_silence_frames = max(1, config.end_silence_ms // audio_config.frame_ms)
        self._emit_frames = max(1, audio_config.asr_chunk_ms // audio_config.frame_ms)
        self._max_frames = max(1, config.max_utterance_seconds * 1000 // audio_config.frame_ms)

    def load(self) -> None:
        try:
            import webrtcvad
        except ImportError as exc:
            raise RuntimeError("Install webrtcvad-wheels to use the WebRTC VAD backend") from exc
        self._vad = webrtcvad.Vad(self.config.aggressiveness)
        self.reset()

    def reset(self) -> None:
        with self._endpoint_lock:
            self._endpoint_request = None
            self._endpoint_requested.clear()
        self._pending = np.empty(0, dtype=np.float32)
        self._silence_history = deque(maxlen=self._padding_frames)
        self._candidate.clear()
        self._utterance.clear()
        self._active = False
        self._silence_frames = 0
        self._frames_since_emit = 0
        self._started_at = monotonic()

    def close(self) -> None:
        self.reset()
        self._vad = None

    def request_endpoint(
        self,
        *,
        started_at: float | None = None,
        cut_sample: int | None = None,
    ) -> None:
        """Ask the audio thread to cut an utterance at a recognized snapshot."""
        if cut_sample is not None and cut_sample <= 0:
            raise ValueError("cut_sample must be positive")
        with self._endpoint_lock:
            self._endpoint_request = _EndpointRequest(started_at, cut_sample)
            self._endpoint_requested.set()

    def _take_endpoint_request(self) -> _EndpointRequest | None:
        if not self._endpoint_requested.is_set():
            return None
        with self._endpoint_lock:
            request = self._endpoint_request
            self._endpoint_request = None
            self._endpoint_requested.clear()
            return request

    def process(self, chunk: AudioChunk) -> list[SpeechSegment]:
        if self._vad is None:
            self.load()
            self.reset()
        if chunk.sample_rate != self.audio_config.sample_rate:
            raise ValueError("Audio must be resampled before VAD")
        if chunk.samples.size:
            self._pending = np.concatenate((self._pending, chunk.samples.astype(np.float32, copy=False)))
        output: list[SpeechSegment] = []
        while len(self._pending) >= self._frame_samples:
            frame = self._pending[: self._frame_samples].copy()
            self._pending = self._pending[self._frame_samples :]
            output.extend(self._process_frame(frame, chunk.captured_at))
        if chunk.end_of_stream:
            output.extend(self.flush())
        return output

    def _process_frame(self, frame: np.ndarray, captured_at: float) -> list[SpeechSegment]:
        assert self._vad is not None
        voiced = self._vad.is_speech(_float_to_pcm16(frame), self.audio_config.sample_rate)
        if not self._active:
            if voiced:
                if not self._candidate:
                    self._started_at = captured_at
                self._candidate.append(frame)
                if len(self._candidate) >= self._min_speech_frames:
                    self._active = True
                    self._utterance = [*self._silence_history, *self._candidate]
                    self._candidate.clear()
                    self._silence_frames = 0
                    self._frames_since_emit = len(self._utterance)
            else:
                self._candidate.clear()
                self._silence_history.append(frame)
            return []

        self._utterance.append(frame)
        self._frames_since_emit += 1
        self._silence_frames = 0 if voiced else self._silence_frames + 1

        endpoint = self._take_endpoint_request()
        if endpoint is not None:
            if endpoint.started_at is None or endpoint.started_at == self._started_at:
                if endpoint.cut_sample is None:
                    return [self._finish(captured_at)]
                return [self._finish_at(endpoint.cut_sample)]
        if len(self._utterance) >= self._max_frames:
            return [self._finish(captured_at)]
        if self._silence_frames >= self._end_silence_frames:
            keep_silence = min(self._padding_frames, self._silence_frames)
            if self._silence_frames > keep_silence:
                del self._utterance[-(self._silence_frames - keep_silence) :]
            return [self._finish(captured_at)]
        if self._frames_since_emit >= self._emit_frames:
            self._frames_since_emit = 0
            return [self._snapshot(captured_at, is_final=False)]
        return []

    def _snapshot(self, ended_at: float, is_final: bool) -> SpeechSegment:
        samples = np.concatenate(self._utterance) if self._utterance else np.empty(0, dtype=np.float32)
        return SpeechSegment(samples, self.audio_config.sample_rate, self._started_at, ended_at, is_final)

    def _finish(self, ended_at: float) -> SpeechSegment:
        segment = self._snapshot(ended_at, is_final=True)
        self.reset()
        return segment

    def _finish_at(self, cut_sample: int) -> SpeechSegment:
        samples = (
            np.concatenate(self._utterance)
            if self._utterance
            else np.empty(0, dtype=np.float32)
        )
        cut_sample = min(max(1, cut_sample), len(samples))
        prefix = samples[:cut_sample].copy()
        suffix = samples[cut_sample:].copy()
        old_started_at = self._started_at
        cut_ended_at = old_started_at + cut_sample / self.audio_config.sample_rate
        segment = SpeechSegment(
            prefix,
            self.audio_config.sample_rate,
            old_started_at,
            cut_ended_at,
            True,
            True,
        )

        self.reset()
        if suffix.size:
            self._active = True
            self._started_at = cut_ended_at
            self._utterance = [
                suffix[index : index + self._frame_samples].copy()
                for index in range(0, len(suffix), self._frame_samples)
            ]
            self._frames_since_emit = len(self._utterance)
        return segment

    def flush(self) -> list[SpeechSegment]:
        if self._active and self._utterance:
            return [self._finish(monotonic())]
        self.reset()
        return []


class PassthroughVad:
    """Test/file backend treating every received chunk as speech."""

    def __init__(self, config: VadConfig, audio_config: AudioConfig) -> None:
        self.audio_config = audio_config
        self._endpoint_requested = threading.Event()
        self._endpoint_lock = threading.Lock()
        self._endpoint_request: _EndpointRequest | None = None
        self._samples: list[np.ndarray] = []
        self._started_at = monotonic()

    def load(self) -> None:
        pass

    def reset(self) -> None:
        with self._endpoint_lock:
            self._endpoint_request = None
            self._endpoint_requested.clear()
        self._samples.clear()
        self._started_at = monotonic()

    def close(self) -> None:
        self.reset()

    def request_endpoint(
        self,
        *,
        started_at: float | None = None,
        cut_sample: int | None = None,
    ) -> None:
        if cut_sample is not None and cut_sample <= 0:
            raise ValueError("cut_sample must be positive")
        with self._endpoint_lock:
            self._endpoint_request = _EndpointRequest(started_at, cut_sample)
            self._endpoint_requested.set()

    def _take_endpoint_request(self) -> _EndpointRequest | None:
        if not self._endpoint_requested.is_set():
            return None
        with self._endpoint_lock:
            request = self._endpoint_request
            self._endpoint_request = None
            self._endpoint_requested.clear()
            return request

    def process(self, chunk: AudioChunk) -> list[SpeechSegment]:
        if chunk.samples.size:
            self._samples.append(chunk.samples)
        samples = np.concatenate(self._samples) if self._samples else np.empty(0, dtype=np.float32)
        endpoint = self._take_endpoint_request()
        if endpoint is not None and samples.size:
            if endpoint.started_at is None or endpoint.started_at == self._started_at:
                if endpoint.cut_sample is None:
                    return self.flush()
                cut_sample = min(max(1, endpoint.cut_sample), len(samples))
                prefix = samples[:cut_sample].copy()
                suffix = samples[cut_sample:].copy()
                old_started_at = self._started_at
                cut_ended_at = (
                    old_started_at + cut_sample / self.audio_config.sample_rate
                )
                result = SpeechSegment(
                    prefix,
                    self.audio_config.sample_rate,
                    old_started_at,
                    cut_ended_at,
                    True,
                    True,
                )
                self.reset()
                if suffix.size:
                    self._samples = [suffix]
                    self._started_at = cut_ended_at
                return [result]
        if chunk.end_of_stream:
            return self.flush()
        return [SpeechSegment(samples, chunk.sample_rate, self._started_at, monotonic(), False)] if samples.size else []

    def flush(self) -> list[SpeechSegment]:
        if not self._samples:
            return []
        samples = np.concatenate(self._samples)
        result = SpeechSegment(samples, self.audio_config.sample_rate, self._started_at, monotonic(), True)
        self.reset()
        return [result]
