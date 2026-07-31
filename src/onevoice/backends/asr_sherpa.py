from __future__ import annotations

import shutil
import tarfile
import tempfile
import threading
import urllib.request
from collections import Counter
from pathlib import Path
from time import monotonic
from typing import Any

import numpy as np

from onevoice.config import AsrConfig
from onevoice.models import AsrUpdate, AsrWordTiming, SpeechSegment
from onevoice.sherpa_models import (
    DEFAULT_SHERPA_STREAMING_MODEL_BY_LANGUAGE,
    EN_ONLINE_PUNCTUATION,
    SHERPA_STREAMING_MODELS,
    SherpaStreamingModelSpec,
)
from onevoice.text import tokenize_text


_DOWNLOAD_LOCK = threading.Lock()


class SherpaOnnxZipformerAsrBackend:
    """True-streaming Zipformer adapter for Vietnamese, English, Chinese and Korean."""
    terminology_capability = "native_hotwords"

    def __init__(self, config: AsrConfig) -> None:
        self.config = config
        self._validate_selection()
        self._recognizer: Any = None
        self._stream: Any = None
        self._sherpa: Any = None
        self._punctuator: Any = None
        self._processed_samples = 0
        self._segment_started_at: float | None = None
        self._revision = 0
        self._hotwords = ""
        self._configured_hotwords: tuple[Any, ...] = ()
        self._hotword_score = 1.5
        self._metrics: Counter[str] = Counter()

    def configure_native_hotwords(
        self,
        hotwords: tuple[Any, ...],
        *,
        global_score: float,
    ) -> None:
        if hotwords and (
            self.config.sherpa.decoding_method
            != "modified_beam_search"
        ):
            raise ValueError(
                "Sherpa native hotwords require "
                "asr.sherpa.decoding_method=modified_beam_search"
            )
        self._configured_hotwords = hotwords
        self._hotwords = "\n".join(
            f"{item.text} :{item.score:g}" for item in hotwords
        )
        self._hotword_score = global_score

    def take_metrics(self) -> dict[str, int]:
        output = dict(self._metrics)
        self._metrics.clear()
        return output

    def _validate_selection(self) -> None:
        if self.config.language not in DEFAULT_SHERPA_STREAMING_MODEL_BY_LANGUAGE:
            raise ValueError(
                "sherpa_onnx streaming Zipformer requires an explicit "
                "language: vi, en, zh, or ko"
            )
        if self.config.sherpa.recognizer_mode != "online_transducer":
            raise ValueError(
                "Streaming Zipformer requires "
                "asr.sherpa.recognizer_mode=online_transducer"
            )
        if self.config.model != "auto":
            try:
                spec = SHERPA_STREAMING_MODELS[self.config.model]
            except KeyError as exc:
                raise ValueError(
                    f"Unsupported sherpa_onnx streaming model "
                    f"{self.config.model!r}; available: "
                    f"{sorted(SHERPA_STREAMING_MODELS)}"
                ) from exc
            if spec.language != self.config.language:
                raise ValueError(
                    f"Model {spec.id!r} supports {spec.language!r}, not "
                    f"{self.config.language!r}"
                )

    @property
    def model_spec(self) -> SherpaStreamingModelSpec:
        model_id = (
            DEFAULT_SHERPA_STREAMING_MODEL_BY_LANGUAGE[self.config.language]
            if self.config.model == "auto"
            else self.config.model
        )
        return SHERPA_STREAMING_MODELS[model_id]

    @property
    def native_hotword_text_case(self) -> str:
        return self.model_spec.hotword_case

    @staticmethod
    def _required_paths(
        root: Path, spec: SherpaStreamingModelSpec
    ) -> dict[str, Path]:
        return {
            "encoder": root / spec.encoder,
            "decoder": root / spec.decoder,
            "joiner": root / spec.joiner,
            "tokens": root / spec.tokens,
        }

    def _validate_assets(self, root: Path) -> None:
        missing = [
            path.name
            for path in self._required_paths(root, self.model_spec).values()
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"Streaming Zipformer assets are incomplete in {root}: "
                f"{missing}"
            )

    def _resolve_model_dir(self) -> Path:
        if self.config.model_dir:
            root = Path(self.config.model_dir)
            self._validate_assets(root)
            return root
        return self._ensure_cached(self.model_spec)

    def _ensure_cached(self, spec: SherpaStreamingModelSpec) -> Path:
        cache_root = Path(self.config.sherpa.cache_dir).resolve()
        model_dir = cache_root / spec.directory
        try:
            self._validate_assets(model_dir)
            return model_dir
        except FileNotFoundError:
            pass
        if self.config.offline:
            raise FileNotFoundError(
                f"Streaming Zipformer model {spec.id!r} is not cached in "
                f"{cache_root}; disable offline mode once to download it"
            )

        with _DOWNLOAD_LOCK:
            try:
                self._validate_assets(model_dir)
                return model_dir
            except FileNotFoundError:
                pass
            cache_root.mkdir(parents=True, exist_ok=True)
            staging = Path(
                tempfile.mkdtemp(prefix=f".{spec.directory}-", dir=cache_root)
            )
            try:
                if spec.archive:
                    self._download_archive(spec, staging)
                else:
                    self._download_files(spec, staging)
                extracted = staging / spec.directory
                self._validate_assets(extracted)
                if model_dir.exists():
                    shutil.rmtree(model_dir)
                shutil.move(str(extracted), str(model_dir))
                source = spec.archive_url or spec.source_url
                (model_dir / ".complete").write_text(
                    source, encoding="utf-8"
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Could not download streaming Zipformer model "
                    f"{spec.id}: {exc}"
                ) from exc
            finally:
                shutil.rmtree(staging, ignore_errors=True)
        return model_dir

    def _ensure_punctuation_cached(self) -> Path:
        spec = EN_ONLINE_PUNCTUATION
        cache_root = Path(self.config.sherpa.cache_dir).resolve()
        model_dir = cache_root / spec.directory
        required = (model_dir / spec.model, model_dir / spec.bpe_vocab)
        if all(path.is_file() for path in required):
            return model_dir
        if self.config.offline:
            raise FileNotFoundError(
                f"English punctuation model {spec.id!r} is not cached in "
                f"{cache_root}; disable offline mode once to download it"
            )

        with _DOWNLOAD_LOCK:
            if all(path.is_file() for path in required):
                return model_dir
            cache_root.mkdir(parents=True, exist_ok=True)
            staging = Path(
                tempfile.mkdtemp(prefix=f".{spec.directory}-", dir=cache_root)
            )
            try:
                archive_path = staging / spec.archive
                urllib.request.urlretrieve(spec.archive_url, archive_path)
                self._safe_extract(archive_path, staging)
                extracted = staging / spec.directory
                extracted_required = (
                    extracted / spec.model,
                    extracted / spec.bpe_vocab,
                )
                if not all(path.is_file() for path in extracted_required):
                    raise FileNotFoundError(
                        f"Punctuation assets are incomplete in {extracted}"
                    )
                if model_dir.exists():
                    shutil.rmtree(model_dir)
                shutil.move(str(extracted), str(model_dir))
                (model_dir / ".complete").write_text(
                    spec.archive_url, encoding="utf-8"
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Could not download English punctuation model "
                    f"{spec.id}: {exc}"
                ) from exc
            finally:
                shutil.rmtree(staging, ignore_errors=True)
        return model_dir

    def _ensure_hotword_assets(
        self, root: Path
    ) -> tuple[str, str]:
        spec = self.model_spec
        if spec.modeling_unit == "cjkchar":
            return spec.modeling_unit, ""
        if not spec.bpe_model or not spec.bpe_vocab:
            raise RuntimeError(
                f"Model {spec.id!r} does not declare BPE hotword assets"
            )
        model_path = root / spec.bpe_model
        if not model_path.is_file():
            if self.config.offline or not spec.bpe_model_url:
                raise FileNotFoundError(
                    f"Native hotwords for {spec.id!r} require "
                    f"{model_path.name}; disable offline mode once to "
                    "download it"
                )
            temporary = model_path.with_suffix(".download")
            urllib.request.urlretrieve(spec.bpe_model_url, temporary)
            temporary.replace(model_path)
        vocab_path = root / spec.bpe_vocab
        if not vocab_path.is_file():
            try:
                import sentencepiece as spm
            except ImportError as exc:
                raise RuntimeError(
                    "BPE hotword export requires sentencepiece; install "
                    "the 'sherpa-asr' extra"
                ) from exc
            processor = spm.SentencePieceProcessor(
                model_file=str(model_path)
            )
            content = "".join(
                f"{processor.id_to_piece(index)}\t"
                f"{processor.get_score(index)}\n"
                for index in range(processor.get_piece_size())
            )
            temporary = vocab_path.with_suffix(".tmp")
            temporary.write_text(content, encoding="utf-8")
            temporary.replace(vocab_path)
        return spec.modeling_unit, str(vocab_path.resolve())

    def _prepare_hotwords(self, root: Path) -> str:
        if not self._configured_hotwords:
            return ""
        token_values = {
            line.rsplit(maxsplit=1)[0]
            for line in (
                root / self.model_spec.tokens
            ).read_text(encoding="utf-8").splitlines()
            if len(line.rsplit(maxsplit=1)) == 2
        }
        accepted: list[Any] = []
        if self.model_spec.modeling_unit == "cjkchar":
            for item in self._configured_hotwords:
                units = tuple(
                    character
                    for character in item.text
                    if not character.isspace()
                )
                if units and all(unit in token_values for unit in units):
                    accepted.append(item)
        else:
            assert self.model_spec.bpe_model is not None
            try:
                import sentencepiece as spm
            except ImportError as exc:
                raise RuntimeError(
                    "BPE hotword validation requires sentencepiece; "
                    "install the 'sherpa-asr' extra"
                ) from exc
            processor = spm.SentencePieceProcessor(
                model_file=str(root / self.model_spec.bpe_model)
            )
            for item in self._configured_hotwords:
                pieces = tuple(
                    processor.encode(item.text, out_type=str)
                )
                if (
                    pieces
                    and "<unk>" not in pieces
                    and all(piece in token_values for piece in pieces)
                ):
                    accepted.append(item)
        rejected = len(self._configured_hotwords) - len(accepted)
        self._metrics["asr_hotword_term_count"] = len(accepted)
        self._metrics["asr_hotword_token_count"] = sum(
            item.token_count for item in accepted
        )
        self._metrics["asr_hotword_rejection_count"] = rejected
        return "\n".join(
            f"{item.text} :{item.score:g}" for item in accepted
        )

    def _download_archive(
        self, spec: SherpaStreamingModelSpec, staging: Path
    ) -> None:
        assert spec.archive is not None
        assert spec.archive_url is not None
        archive_path = staging / spec.archive
        urllib.request.urlretrieve(spec.archive_url, archive_path)
        self._safe_extract(archive_path, staging)
        archive_path.unlink(missing_ok=True)

    @staticmethod
    def _download_files(
        spec: SherpaStreamingModelSpec, staging: Path
    ) -> None:
        destination = staging / spec.directory
        destination.mkdir(parents=True, exist_ok=True)
        for filename, url in spec.remote_files:
            urllib.request.urlretrieve(url, destination / filename)

    @staticmethod
    def _safe_extract(archive_path: Path, destination: Path) -> None:
        destination = destination.resolve()
        with tarfile.open(archive_path, mode="r:bz2") as archive:
            for member in archive.getmembers():
                target = (destination / member.name).resolve()
                if target != destination and destination not in target.parents:
                    raise RuntimeError(
                        f"Unsafe path in Zipformer archive: {member.name}"
                    )
                if member.issym() or member.islnk():
                    raise RuntimeError(
                        f"Links are not allowed in Zipformer archive: "
                        f"{member.name}"
                    )
            archive.extractall(destination)

    def load(self) -> None:
        if self._recognizer is not None:
            return
        try:
            if self._sherpa is None:
                import sherpa_onnx

                self._sherpa = sherpa_onnx
        except ImportError as exc:
            raise RuntimeError(
                "Install the sherpa ASR extra: "
                "python -m pip install -e \".[sherpa-asr]\""
            ) from exc

        root = self._resolve_model_dir()
        paths = self._required_paths(root, self.model_spec)
        settings = self.config.sherpa
        modeling_unit = self.model_spec.modeling_unit
        bpe_vocab = ""
        if self._configured_hotwords:
            modeling_unit, bpe_vocab = self._ensure_hotword_assets(root)
            self._hotwords = self._prepare_hotwords(root)
        self._recognizer = self._sherpa.OnlineRecognizer.from_transducer(
            tokens=str(paths["tokens"].resolve()),
            encoder=str(paths["encoder"].resolve()),
            decoder=str(paths["decoder"].resolve()),
            joiner=str(paths["joiner"].resolve()),
            num_threads=settings.num_threads,
            sample_rate=16_000,
            feature_dim=80,
            enable_endpoint_detection=False,
            decoding_method=settings.decoding_method,
            max_active_paths=settings.max_active_paths,
            hotwords_score=self._hotword_score,
            modeling_unit=modeling_unit,
            bpe_vocab=bpe_vocab,
            provider=settings.provider,
        )
        if (
            self.model_spec.language == "en"
            and settings.punctuation_enabled
        ):
            punctuation_root = self._ensure_punctuation_cached()
            spec = EN_ONLINE_PUNCTUATION
            model_config = self._sherpa.OnlinePunctuationModelConfig(
                cnn_bilstm=str((punctuation_root / spec.model).resolve()),
                bpe_vocab=str(
                    (punctuation_root / spec.bpe_vocab).resolve()
                ),
                num_threads=settings.num_threads,
                provider=settings.provider,
            )
            punctuation_config = self._sherpa.OnlinePunctuationConfig(
                model_config
            )
            self._punctuator = self._sherpa.OnlinePunctuation(
                punctuation_config
            )

    def _ensure_stream(self) -> Any:
        if self._recognizer is None:
            self.load()
        if self._stream is None:
            self._stream = (
                self._recognizer.create_stream(self._hotwords)
                if self._hotwords
                else self._recognizer.create_stream()
            )
            if self._hotwords:
                self._metrics["asr_hotword_stream_count"] += 1
            self._processed_samples = 0
        return self._stream

    def _decode_ready(self, stream: Any) -> None:
        while self._recognizer.is_ready(stream):
            self._recognizer.decode_stream(stream)

    def reset(self) -> None:
        self._stream = None
        self._processed_samples = 0
        self._segment_started_at = None
        self._revision = 0

    def close(self) -> None:
        self._stream = None
        self._recognizer = None
        self._sherpa = None
        self._punctuator = None
        self._processed_samples = 0
        self._segment_started_at = None
        self._revision = 0

    @staticmethod
    def _sentence_case_all_caps(text: str) -> str:
        if not text.isupper():
            return text
        lowered = text.lower()
        output: list[str] = []
        capitalize_next = True
        for character in lowered:
            if capitalize_next and character.isalpha():
                character = character.upper()
                capitalize_next = False
            output.append(character)
            if character in ".!?。！？":
                capitalize_next = True
        return "".join(output)

    def _normalize_text(self, text: str, language: str) -> str:
        if language == "en" and self._punctuator is not None and text:
            return str(
                self._punctuator.add_punctuation_with_case(text.lower())
            ).strip()
        if language in {"en", "vi"}:
            return self._sentence_case_all_caps(text)
        return text

    def _word_timings(
        self,
        stream: Any,
        *,
        language: str,
        segment_samples: int,
        sample_rate: int,
    ) -> tuple[AsrWordTiming, ...]:
        read_tokens = getattr(self._recognizer, "tokens", None)
        read_timestamps = getattr(self._recognizer, "timestamps", None)
        if not callable(read_tokens) or not callable(read_timestamps):
            return ()
        pieces = tuple(str(piece) for piece in read_tokens(stream))
        timestamps = tuple(float(value) for value in read_timestamps(stream))
        if not pieces or len(pieces) != len(timestamps):
            return ()

        units: list[tuple[str, float, float]] = []
        current = ""
        current_start = 0.0
        current_last = 0.0
        pending_start: float | None = None

        def flush() -> None:
            nonlocal current
            text = current.strip()
            if text and any(character.isalnum() for character in text):
                units.append((text, current_start, current_last))
            current = ""

        for raw_piece, timestamp in zip(pieces, timestamps, strict=True):
            piece = raw_piece.replace("▁", " ")
            if not piece or piece.startswith("<"):
                continue
            timestamp = max(0.0, timestamp)
            if language == "zh":
                text = piece.strip()
                if text and any(character.isalnum() for character in text):
                    units.append((text, timestamp, timestamp))
                continue
            if piece[0].isspace():
                flush()
                pending_start = timestamp
                piece = piece.lstrip()
                if not piece:
                    continue
            if not current:
                current_start = (
                    pending_start if pending_start is not None else timestamp
                )
                pending_start = None
            current += piece
            current_last = timestamp
        flush()

        duration = segment_samples / sample_rate
        output: list[AsrWordTiming] = []
        for index, (text, start, last_token_start) in enumerate(units):
            if index + 1 < len(units):
                next_start = units[index + 1][1]
                guard = min(
                    0.06,
                    max(0.0, next_start - last_token_start) / 2,
                )
                end = next_start - guard
            else:
                end = duration
            start = min(max(0.0, start), duration)
            end = min(max(start, end), duration)
            output.append(AsrWordTiming(text, start, end))
        return tuple(output)

    @staticmethod
    def _strip_acoustic_context(
        raw_text: str,
        words: tuple[AsrWordTiming, ...],
        *,
        language: str,
        context_seconds: float,
    ) -> tuple[str, tuple[AsrWordTiming, ...]]:
        if context_seconds <= 0 or not words:
            return raw_text, words
        visible = tuple(
            word
            for word in words
            if word.start_seconds >= context_seconds - 0.02
        )
        if not visible:
            return "", ()
        separator = "" if language == "zh" else " "
        visible_text = separator.join(word.text for word in visible)
        adjusted = tuple(
            AsrWordTiming(
                word.text,
                max(0.0, word.start_seconds - context_seconds),
                max(0.0, word.end_seconds - context_seconds),
                word.confidence,
            )
            for word in visible
        )
        return visible_text, adjusted

    def transcribe(
        self, segment: SpeechSegment, language: str | None
    ) -> AsrUpdate:
        selected = (
            self.config.language if language in (None, "auto") else language
        )
        if selected != self.model_spec.language:
            raise ValueError(
                f"Active streaming Zipformer model supports "
                f"{self.model_spec.language!r}, not {selected!r}"
        )
        started = monotonic()
        if (
            self._segment_started_at is not None
            and segment.started_at != self._segment_started_at
        ):
            # A new VAD utterance can arrive before the caller has observed
            # the preceding final. Never carry native Zipformer state across
            # utterance identities.
            self._stream = None
            self._processed_samples = 0
        stream = self._ensure_stream()
        self._segment_started_at = segment.started_at
        samples = np.asarray(segment.samples, dtype=np.float32)
        if len(samples) < self._processed_samples:
            if not segment.is_final:
                raise ValueError(
                    "SpeechSegment snapshot shrank within an active ASR "
                    "utterance"
                )
            # A semantic endpoint deliberately cuts at an earlier sentence
            # boundary. Online transducers cannot un-feed their tail, so
            # decode the authoritative final prefix once on a fresh stream.
            self._stream = None
            self._processed_samples = 0
            stream = self._ensure_stream()
            self._segment_started_at = segment.started_at
        unseen = samples[self._processed_samples :]
        if unseen.size:
            stream.accept_waveform(segment.sample_rate, unseen)
            self._processed_samples = len(samples)
        self._decode_ready(stream)
        if segment.is_final:
            padding_samples = (
                segment.sample_rate
                * self.config.sherpa.final_padding_ms
                // 1_000
            )
            if padding_samples:
                stream.accept_waveform(
                    segment.sample_rate,
                    np.zeros(padding_samples, dtype=np.float32),
                )
                self._decode_ready(stream)
            stream.input_finished()
            self._decode_ready(stream)
        raw_text = str(self._recognizer.get_result(stream)).strip()
        words = self._word_timings(
            stream,
            language=selected,
            segment_samples=len(samples),
            sample_rate=segment.sample_rate,
        )
        raw_text, words = self._strip_acoustic_context(
            raw_text,
            words,
            language=selected,
            context_seconds=segment.context_samples / segment.sample_rate,
        )
        text = self._normalize_text(raw_text, selected)
        self._revision += 1
        update = AsrUpdate(
            text=text,
            language=selected,
            confidence=None,
            revision=self._revision,
            is_final=segment.is_final,
            started_at=started,
            tokens=tokenize_text(text, selected),
            words=words,
            is_endpoint_cut=segment.is_endpoint_cut,
        )
        if segment.is_final:
            self._stream = None
            self._processed_samples = 0
            self._segment_started_at = None
        return update
