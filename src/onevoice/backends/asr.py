from __future__ import annotations

from math import isfinite
from pathlib import Path
from time import monotonic
from typing import Any, Iterable

import numpy as np

from onevoice.config import AsrConfig
from onevoice.models import AsrUpdate, AsrWordTiming, SpeechSegment
from onevoice.text import tokenize_text


SUPPORTED_MOONSHINE_LANGUAGES = frozenset(("vi", "en", "zh", "ko"))
SUPPORTED_DOLPHIN_LANGUAGES = frozenset(("vi", "zh", "ko"))

MOONSHINE_MODELS_BY_LANGUAGE = {
    "en": ("auto", "tiny", "base", "tiny_streaming", "small_streaming", "medium_streaming"),
    "vi": ("auto", "base"),
    "zh": ("auto", "base"),
    "ko": ("auto", "tiny"),
}
DOLPHIN_MODELS = ("base", "small")
FASTER_WHISPER_MODELS = ("tiny", "base", "small")


def asr_model_options(backend: str, language: str) -> tuple[str, ...]:
    """Return built-in model choices without importing any model runtime."""
    if backend == "moonshine":
        return MOONSHINE_MODELS_BY_LANGUAGE.get(language, ("auto",))
    if backend == "dolphin":
        return DOLPHIN_MODELS
    if backend == "faster_whisper":
        return FASTER_WHISPER_MODELS
    if backend == "fake":
        return ("fake",)
    return ()


def validate_asr_selection(backend: str, model: str, language: str) -> None:
    """Fail before model loading when a built-in ASR combination is unsupported."""
    normalized_model = model.lower().replace("-", "_")
    if backend == "moonshine":
        if language not in SUPPORTED_MOONSHINE_LANGUAGES:
            raise ValueError(
                "Moonshine uses per-language models and cannot auto-detect language; "
                "select one of: en, vi, zh, ko"
            )
        supported = MOONSHINE_MODELS_BY_LANGUAGE[language]
        if normalized_model not in supported:
            raise ValueError(
                f"Moonshine model {model!r} is not available for language {language!r}; "
                f"supported models: {', '.join(supported)}"
            )
    elif backend == "dolphin":
        if language not in (*SUPPORTED_DOLPHIN_LANGUAGES, "auto"):
            raise ValueError(
                "Dolphin does not support English; select auto, vi, zh, or ko"
            )
        if normalized_model not in DOLPHIN_MODELS:
            raise ValueError(
                f"Dolphin model {model!r} is unsupported; supported models: "
                f"{', '.join(DOLPHIN_MODELS)}"
            )
    elif backend in ("faster_whisper", "fake") and not model.strip():
        raise ValueError(f"{backend} requires a non-empty model name")


def _selected_language(language: str | None, configured: str) -> str | None:
    value = configured if language in (None, "auto") else language
    return None if value in (None, "auto") else value


def _transcript_text(transcript: Any) -> str:
    """Convert Moonshine's newest-first Transcript lines to readable text."""
    if transcript is None:
        return ""
    lines = list(getattr(transcript, "lines", ()) or ())
    if lines and all(hasattr(line, "start_time") for line in lines):
        lines.sort(key=lambda line: line.start_time)
    else:
        lines.reverse()
    texts = [str(getattr(line, "text", "")).strip() for line in lines]
    return " ".join(text for text in texts if text).strip()


def _transcript_word_timings(transcript: Any) -> tuple[AsrWordTiming, ...]:
    """Flatten Moonshine line words into utterance-relative timing order."""
    lines = list(getattr(transcript, "lines", ()) or ())
    lines.sort(key=lambda line: float(getattr(line, "start_time", 0.0)))
    output: list[AsrWordTiming] = []
    for line in lines:
        words = list(getattr(line, "words", ()) or ())
        words.sort(key=lambda word: float(getattr(word, "start", 0.0)))
        for word in words:
            text = str(getattr(word, "word", "")).strip()
            start = float(getattr(word, "start", -1.0))
            end = float(getattr(word, "end", -1.0))
            if (
                not text
                or not isfinite(start)
                or not isfinite(end)
                or start < 0
                or end <= start
            ):
                continue
            confidence = getattr(word, "confidence", None)
            output.append(
                AsrWordTiming(
                    text=text,
                    start_seconds=start,
                    end_seconds=end,
                    confidence=float(confidence) if confidence is not None else None,
                )
            )
    return tuple(output)


class MoonshineAsrBackend:
    """Native incremental ASR using Moonshine's cached streaming state.

    SpeechSegment partials are growing snapshots. Only the unseen tail is added
    to Moonshine, preventing the repeated full-utterance work done by Whisper.
    """

    def __init__(self, config: AsrConfig) -> None:
        validate_asr_selection("moonshine", config.model, config.language)
        self.config = config
        self._transcriber: Any = None
        self._transcriber_type: Any = None
        self._get_model: Any = None
        self._model_arch_type: Any = None
        self._loaded_language: str | None = None
        self._stream_started = False
        self._processed_samples = 0
        self._revision = 0
        self._last_text = ""

    def load(self) -> None:
        try:
            from moonshine_voice import ModelArch, Transcriber, get_model_for_language
        except ImportError as exc:
            raise RuntimeError(
                "Install the 'moonshine' extra to use Moonshine ASR: "
                "python -m pip install -e \".[moonshine]\""
            ) from exc
        self._transcriber_type = Transcriber
        self._get_model = get_model_for_language
        self._model_arch_type = ModelArch
        language = _selected_language(self.config.language, self.config.language)
        if language is not None:
            self._load_language(language)

    def _model_arch(self) -> Any:
        if self.config.model.lower() == "auto":
            return None
        name = self.config.model.upper().replace("-", "_")
        try:
            return getattr(self._model_arch_type, name)
        except AttributeError as exc:
            raise ValueError(f"Unknown Moonshine model architecture: {self.config.model}") from exc

    def _load_language(self, language: str) -> None:
        if language not in SUPPORTED_MOONSHINE_LANGUAGES:
            raise ValueError(f"Moonshine backend does not support language '{language}'")
        if self._transcriber is not None and self._loaded_language == language:
            return
        self._close_transcriber()
        architecture = self._model_arch()
        if self.config.offline:
            if not self.config.model_dir:
                raise RuntimeError(
                    "Moonshine offline mode requires asr.model_dir pointing to downloaded model files"
                )
            model_path = Path(self.config.model_dir)
            if not model_path.is_dir():
                raise RuntimeError(f"Moonshine model directory not found: {model_path}")
            if architecture is None:
                raise RuntimeError("Moonshine offline mode requires an explicit asr.model architecture")
        else:
            cache_root = Path(self.config.model_dir) if self.config.model_dir else None
            model_path, architecture = self._get_model(
                wanted_language=language,
                wanted_model_arch=architecture,
                cache_root=cache_root,
            )
        options = {
            "identify_speakers": "false",
            "return_audio_data": "false",
            # Semantic endpointing aligns a stable sentence boundary to the
            # exact word end instead of waiting for a whole-snapshot boundary.
            "word_timestamps": "true",
            # Segmentation is already owned by the pipeline's WebRTC VAD.
            "vad_threshold": "0",
        }
        if language != "en":
            options["max_tokens_per_second"] = "13.0"
        if self.config.device.lower() == "cuda":
            options["ort_providers"] = "CUDA,CPU"
        self._transcriber = self._transcriber_type(
            model_path=str(model_path),
            model_arch=architecture,
            update_interval=self.config.update_interval,
            options=options,
        )
        self._loaded_language = language

    def _stop_stream(self) -> Any:
        if self._stream_started and self._transcriber is not None:
            try:
                return self._transcriber.stop()
            finally:
                self._stream_started = False
        return None

    def _close_transcriber(self) -> None:
        self._stop_stream()
        transcriber, self._transcriber = self._transcriber, None
        if transcriber is not None:
            close = getattr(transcriber, "close", None)
            if callable(close):
                close()
        self._loaded_language = None

    def reset(self) -> None:
        self._stop_stream()
        self._processed_samples = 0
        self._revision = 0
        self._last_text = ""

    def close(self) -> None:
        self._close_transcriber()
        self._processed_samples = 0
        self._revision = 0
        self._last_text = ""

    def transcribe(self, segment: SpeechSegment, language: str | None) -> AsrUpdate:
        if self._transcriber_type is None:
            self.load()
        selected = _selected_language(language, self.config.language)
        if selected is None:
            raise ValueError(
                "Moonshine uses per-language models; select vi, en, zh, or ko instead of auto"
            )
        self._load_language(selected)
        started = monotonic()

        # A shorter final can occur when VAD trims trailing padding. Recreate the
        # native stream so samples are never silently omitted or duplicated.
        if len(segment.samples) < self._processed_samples:
            self.reset()
        if not self._stream_started:
            self._transcriber.start()
            self._stream_started = True

        tail = np.asarray(segment.samples[self._processed_samples :], dtype=np.float32)
        if tail.size:
            self._transcriber.add_audio(tail.tolist(), segment.sample_rate)
            self._processed_samples = len(segment.samples)
        transcript = self._transcriber.update_transcription()
        if segment.is_final:
            # Moonshine flushes buffered encoder/decoder state during stop() and
            # returns the final Transcript; do not discard that final revision.
            transcript = self._stop_stream() or transcript
        text = _transcript_text(transcript) or self._last_text
        self._last_text = text
        self._revision += 1
        return AsrUpdate(
            text=text,
            language=selected,
            confidence=None,
            revision=self._revision,
            is_final=segment.is_final,
            started_at=started,
            tokens=tokenize_text(text, selected),
            words=_transcript_word_timings(transcript),
            is_endpoint_cut=segment.is_endpoint_cut,
        )


class DolphinAsrBackend:
    """Dolphin Base/Small adapter for Vietnamese, Mandarin and Korean."""

    def __init__(self, config: AsrConfig) -> None:
        validate_asr_selection("dolphin", config.model, config.language)
        self.config = config
        self._model: Any = None
        self._transcribe: Any = None
        self._torch: Any = None
        self._revision = 0

    def load(self) -> None:
        try:
            import dolphin
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "Install the 'dolphin' extra to use Dolphin ASR: "
                "python -m pip install -e \".[dolphin]\""
            ) from exc
        if self.config.offline and not self.config.model_dir:
            raise RuntimeError("Dolphin offline mode requires asr.model_dir")
        model_dir = self.config.model_dir
        if model_dir is not None and not Path(model_dir).is_dir():
            raise RuntimeError(f"Dolphin model directory not found: {model_dir}")
        self._torch = torch
        self._transcribe = dolphin.transcribe
        self._model = dolphin.load_model(
            self.config.model,
            model_dir=model_dir,
            device=self.config.device,
        )

    def reset(self) -> None:
        self._revision = 0

    def close(self) -> None:
        self._model = None
        self._transcribe = None
        self._torch = None
        self.reset()

    def transcribe(self, segment: SpeechSegment, language: str | None) -> AsrUpdate:
        if self._model is None:
            self.load()
        selected = _selected_language(language, self.config.language)
        if selected == "en":
            raise ValueError(
                "The official Dolphin model does not support English; use Moonshine for source=en"
            )
        if selected is not None and selected not in SUPPORTED_DOLPHIN_LANGUAGES:
            raise ValueError(f"Dolphin backend does not support language '{selected}'")
        started = monotonic()
        waveform = self._torch.from_numpy(
            np.asarray(segment.samples, dtype=np.float32)
        ).unsqueeze(0)
        result = self._transcribe(
            self._model,
            waveform,
            lang_sym=selected,
            decoding_method=self.config.decoding_method,
            beam_size=self.config.beam_size,
            predict_time=False,
            word_timestamp=False,
        )
        text = str(
            getattr(result, "text_nospecial", None) or getattr(result, "text", "")
        ).strip()
        detected = selected or getattr(result, "language", None)
        self._revision += 1
        return AsrUpdate(
            text=text,
            language=detected,
            confidence=None,
            revision=self._revision,
            is_final=segment.is_final,
            started_at=started,
            tokens=tokenize_text(text, detected),
            is_endpoint_cut=segment.is_endpoint_cut,
        )


class FasterWhisperBackend:
    def __init__(self, config: AsrConfig) -> None:
        validate_asr_selection("faster_whisper", config.model, config.language)
        self.config = config
        self._model = None
        self._revision = 0

    def load(self) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError("Install the 'models' extra to use faster-whisper") from exc
        self._model = WhisperModel(
            self.config.model,
            device=self.config.device,
            compute_type=self.config.compute_type,
            local_files_only=self.config.offline,
        )

    def reset(self) -> None:
        self._revision = 0

    def close(self) -> None:
        self._model = None
        self.reset()

    def transcribe(self, segment: SpeechSegment, language: str | None) -> AsrUpdate:
        if self._model is None:
            self.load()
        started = monotonic()
        selected = None if language in (None, "auto") else language
        segments, info = self._model.transcribe(
            segment.samples,
            language=selected,
            beam_size=self.config.beam_size,
            condition_on_previous_text=False,
            vad_filter=False,
            word_timestamps=True,
        )
        segments = list(segments)
        text = " ".join(item.text.strip() for item in segments if item.text.strip()).strip()
        words = tuple(
            AsrWordTiming(
                text=str(word.word).strip(),
                start_seconds=float(word.start),
                end_seconds=float(word.end),
                confidence=(
                    float(word.probability)
                    if getattr(word, "probability", None) is not None
                    else None
                ),
            )
            for item in segments
            for word in (getattr(item, "words", None) or ())
            if str(getattr(word, "word", "")).strip()
            and float(getattr(word, "end", 0.0))
            > float(getattr(word, "start", 0.0))
        )
        detected = selected or getattr(info, "language", None)
        confidence = getattr(info, "language_probability", None)
        self._revision += 1
        return AsrUpdate(
            text=text,
            language=detected,
            confidence=confidence,
            revision=self._revision,
            is_final=segment.is_final,
            started_at=started,
            tokens=tokenize_text(text, detected),
            words=words,
            is_endpoint_cut=segment.is_endpoint_cut,
        )


class FakeAsrBackend:
    def __init__(self, config: AsrConfig, script: Iterable[str] | None = None) -> None:
        validate_asr_selection("fake", config.model, config.language)
        self.config = config
        self.script = list(script or ["hello", "hello world", "hello world."])
        self._index = 0

    def load(self) -> None:
        pass

    def reset(self) -> None:
        self._index = 0

    def close(self) -> None:
        self.reset()

    def transcribe(self, segment: SpeechSegment, language: str | None) -> AsrUpdate:
        started = monotonic()
        index = min(self._index, len(self.script) - 1)
        text = self.script[index] if self.script else ""
        if not segment.is_final:
            self._index += 1
        elif self.script:
            text = self.script[-1]
        selected = "en" if language in (None, "auto") else language
        return AsrUpdate(
            text=text,
            language=selected,
            confidence=1.0,
            revision=self._index + 1,
            is_final=segment.is_final,
            started_at=started,
            tokens=tokenize_text(text, selected),
            is_endpoint_cut=segment.is_endpoint_cut,
        )
