import io
import wave

import numpy as np
import pytest

from onevoice.audio import (
    PlaybackDeadline,
    RealtimePacer,
    encode_wav,
    iter_audio_file,
)
from onevoice.models import AudioChunk


def test_audio_chunk_duration_and_shape_validation() -> None:
    chunk = AudioChunk(np.zeros(8000, dtype=np.float32), 16_000, 0)
    assert chunk.duration_seconds == 0.5


def test_audio_chunk_must_be_mono() -> None:
    try:
        AudioChunk(np.zeros((2, 10), dtype=np.float32), 16_000, 0)
    except ValueError as exc:
        assert "mono" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_wav_file_is_decoded_to_fixed_mono_chunks(tmp_path) -> None:
    path = tmp_path / "sample.wav"
    samples = (np.sin(np.linspace(0, 20, 1600)) * 1000).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16_000)
        output.writeframes(samples.tobytes())
    chunks = list(iter_audio_file(path, sample_rate=16_000, chunk_ms=20))
    assert chunks
    assert sum(len(chunk.samples) for chunk in chunks) == 1600
    assert all(chunk.samples.ndim == 1 for chunk in chunks)


def test_encode_wav_creates_mono_pcm_with_expected_duration() -> None:
    payload = encode_wav(np.zeros(8_000, dtype=np.float32), 16_000)

    with wave.open(io.BytesIO(payload), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == 16_000
        assert wav.getnframes() == 8_000


def test_realtime_pacer_compensates_previous_sleep_drift() -> None:
    pacer = RealtimePacer(started_at=10.0)

    assert pacer.delay_after(0.02, now=10.0) == pytest.approx(0.02)
    # The first frame ran 5 ms late, so the second sleep is shortened rather
    # than adding another full 20 ms and accumulating drift across the file.
    assert pacer.delay_after(0.02, now=10.025) == pytest.approx(0.015)


def test_playback_deadline_tracks_media_finish_and_poll_guard() -> None:
    deadline = PlaybackDeadline.start(
        2.5,
        monotonic_now=10.0,
        wall_now=1_000.0,
    )

    assert deadline.finish_monotonic == pytest.approx(12.5)
    assert deadline.finish_wall_at == pytest.approx(1_002.5)
    assert not deadline.reached(12.64)
    assert deadline.reached(12.65)


def test_playback_deadline_clamps_negative_duration_and_guard() -> None:
    deadline = PlaybackDeadline.start(
        -1.0,
        monotonic_now=10.0,
        wall_now=1_000.0,
    )

    assert deadline.finish_wall_at == pytest.approx(1_000.0)
    assert deadline.reached(10.0, guard_seconds=-1.0)
