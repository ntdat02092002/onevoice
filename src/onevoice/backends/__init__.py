from __future__ import annotations

from onevoice.registry import registry

from .asr import DolphinAsrBackend, FakeAsrBackend, FasterWhisperBackend, MoonshineAsrBackend
from .asr_sherpa import SherpaOnnxZipformerAsrBackend
from .commit import LocalAgreementCommitter
from .preprocess import PassthroughPreprocessor
from .translation import FakeTranslationBackend, M2M100Backend, OpusMtCTranslate2Backend
from .tts import FakeTtsBackend, SherpaOnnxTtsBackend
from .vad import PassthroughVad, WebRtcVadBackend


def register_builtin_backends() -> None:
    registrations = (
        ("preprocessor", "passthrough", PassthroughPreprocessor),
        ("vad", "webrtc", WebRtcVadBackend),
        ("vad", "passthrough", PassthroughVad),
        ("asr", "moonshine", MoonshineAsrBackend),
        ("asr", "dolphin", DolphinAsrBackend),
        ("asr", "faster_whisper", FasterWhisperBackend),
        ("asr", "sherpa_onnx", SherpaOnnxZipformerAsrBackend),
        ("asr", "fake", FakeAsrBackend),
        ("commit", "local_agreement", LocalAgreementCommitter),
        ("translation", "m2m100", M2M100Backend),
        ("translation", "opus_ct2", OpusMtCTranslate2Backend),
        ("translation", "fake", FakeTranslationBackend),
        ("tts", "sherpa_onnx", SherpaOnnxTtsBackend),
        ("tts", "fake", FakeTtsBackend),
    )
    for kind, name, factory in registrations:
        if name not in registry.names(kind):
            registry.register(kind, name, factory)


register_builtin_backends()

__all__ = ["register_builtin_backends"]
