import wave

import numpy as np

from onevoice.audio import iter_audio_file
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
