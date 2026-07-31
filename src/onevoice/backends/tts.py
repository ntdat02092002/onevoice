from __future__ import annotations

import shutil
import tarfile
import tempfile
import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from time import monotonic

import numpy as np

from onevoice.config import TtsConfig
from onevoice.models import TtsRequest, TtsUpdate


_TTS_RELEASE_BASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models"
_DOWNLOAD_LOCK = threading.Lock()


@dataclass(frozen=True, slots=True)
class _VoiceSpec:
    kind: str
    directory: str
    archive: str
    model: str = ""
    tokens: str = ""
    lexicon: str = ""
    data_dir: str = ""
    rule_fsts: tuple[str, ...] = ()

    @property
    def url(self) -> str:
        return f"{_TTS_RELEASE_BASE}/{self.archive}"


AUTO_TTS_VOICES: dict[str, _VoiceSpec] = {
    "vi": _VoiceSpec(
        kind="vits",
        directory="vits-piper-vi_VN-25hours_single-low",
        archive="vits-piper-vi_VN-25hours_single-low.tar.bz2",
        model="vi_VN-25hours_single-low.onnx",
        tokens="tokens.txt",
        data_dir="espeak-ng-data",
    ),
    "en": _VoiceSpec(
        kind="vits",
        directory="vits-piper-en_US-amy-low",
        archive="vits-piper-en_US-amy-low.tar.bz2",
        model="en_US-amy-low.onnx",
        tokens="tokens.txt",
        data_dir="espeak-ng-data",
    ),
    "zh": _VoiceSpec(
        kind="vits",
        directory="vits-piper-zh_CN-chaowen-medium",
        archive="vits-piper-zh_CN-chaowen-medium.tar.bz2",
        model="zh_CN-chaowen-medium.onnx",
        tokens="tokens.txt",
        lexicon="lexicon.txt",
        rule_fsts=("phone.fst", "date.fst", "number.fst"),
    ),
    "ko": _VoiceSpec(
        kind="supertonic",
        directory="sherpa-onnx-supertonic-3-tts-int8-2026-05-11",
        archive="sherpa-onnx-supertonic-3-tts-int8-2026-05-11.tar.bz2",
    ),
}


class SherpaOnnxTtsBackend:
    """Offline sherpa-onnx TTS with automatic voice download and caching."""

    def __init__(self, config: TtsConfig) -> None:
        self.config = config
        self._tts = None
        self._model_kind = "vits"

    @staticmethod
    def _existing(path: Path, label: str) -> str:
        if not path.exists():
            raise FileNotFoundError(f"TTS {label} does not exist: {path}")
        return str(path.resolve())

    def _manual_assets(self) -> dict[str, object]:
        if not self.config.model_dir:
            raise ValueError("tts.model_dir is required when tts.model is not 'auto'")
        root = Path(self.config.model_dir)

        def optional(name: str | None, label: str) -> str:
            return self._existing(root / name, label) if name else ""

        return {
            "kind": "vits",
            "model": optional(self.config.model, "model"),
            "tokens": optional(self.config.tokens, "tokens"),
            "lexicon": optional(self.config.lexicon, "lexicon"),
            "data_dir": optional(self.config.data_dir, "data directory"),
            "rule_fsts": [optional(item, "rule FST") for item in self.config.rule_fsts],
        }

    def _auto_assets(self) -> dict[str, object]:
        try:
            spec = AUTO_TTS_VOICES[self.config.language]
        except KeyError as exc:
            raise ValueError(
                f"No automatic TTS voice for language {self.config.language!r}; "
                f"available: {sorted(AUTO_TTS_VOICES)}"
            ) from exc
        root = self._ensure_cached(spec)
        if spec.kind == "supertonic":
            names = {
                "duration_predictor": "duration_predictor.int8.onnx",
                "text_encoder": "text_encoder.int8.onnx",
                "vector_estimator": "vector_estimator.int8.onnx",
                "vocoder": "vocoder.int8.onnx",
                "tts_json": "tts.json",
                "unicode_indexer": "unicode_indexer.bin",
                "voice_style": "voice.bin",
            }
            return {
                "kind": spec.kind,
                **{key: self._existing(root / name, key) for key, name in names.items()},
            }
        return {
            "kind": spec.kind,
            "model": self._existing(root / spec.model, "model"),
            "tokens": self._existing(root / spec.tokens, "tokens") if spec.tokens else "",
            "lexicon": self._existing(root / spec.lexicon, "lexicon") if spec.lexicon else "",
            "data_dir": self._existing(root / spec.data_dir, "data directory") if spec.data_dir else "",
            "rule_fsts": [self._existing(root / item, "rule FST") for item in spec.rule_fsts],
        }

    def _ensure_cached(self, spec: _VoiceSpec) -> Path:
        cache_root = Path(self.config.cache_dir).resolve()
        voice_dir = cache_root / spec.directory
        marker = voice_dir / ".complete"
        if marker.exists():
            return voice_dir
        if self.config.offline:
            raise FileNotFoundError(
                f"TTS voice {spec.directory!r} is not cached in {cache_root}; "
                "disable offline mode once to download it"
            )

        with _DOWNLOAD_LOCK:
            if marker.exists():
                return voice_dir
            cache_root.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                prefix=f".{spec.directory}-", suffix=".tar.bz2", dir=cache_root, delete=False
            ) as archive_file:
                archive_path = Path(archive_file.name)
            extract_dir = Path(tempfile.mkdtemp(prefix=f".{spec.directory}-", dir=cache_root))
            try:
                urllib.request.urlretrieve(spec.url, archive_path)
                self._safe_extract(archive_path, extract_dir)
                extracted = extract_dir / spec.directory
                if not extracted.is_dir():
                    raise RuntimeError(
                        f"TTS archive {spec.archive} did not contain {spec.directory}"
                    )
                if voice_dir.exists():
                    shutil.rmtree(voice_dir)
                shutil.move(str(extracted), str(voice_dir))
                marker.write_text(spec.url, encoding="utf-8")
            except Exception as exc:
                raise RuntimeError(f"Could not download TTS voice {spec.directory}: {exc}") from exc
            finally:
                archive_path.unlink(missing_ok=True)
                shutil.rmtree(extract_dir, ignore_errors=True)
        return voice_dir

    @staticmethod
    def _safe_extract(archive_path: Path, destination: Path) -> None:
        destination = destination.resolve()
        with tarfile.open(archive_path, mode="r:bz2") as archive:
            for member in archive.getmembers():
                target = (destination / member.name).resolve()
                if target != destination and destination not in target.parents:
                    raise RuntimeError(f"Unsafe path in TTS archive: {member.name}")
                if member.issym() or member.islnk():
                    raise RuntimeError(f"Links are not allowed in TTS archive: {member.name}")
            archive.extractall(destination)

    def load(self) -> None:
        if self._tts is not None:
            return
        assets = self._manual_assets() if self.config.model != "auto" else None
        try:
            import sherpa_onnx
        except ImportError as exc:
            raise RuntimeError(
                "Install the optional TTS dependency with: pip install -e \".[tts]\""
            ) from exc

        if assets is None:
            assets = self._auto_assets()
        self._model_kind = str(assets["kind"])
        if self._model_kind == "supertonic":
            model_config = sherpa_onnx.OfflineTtsModelConfig(
                supertonic=sherpa_onnx.OfflineTtsSupertonicModelConfig(
                    duration_predictor=assets["duration_predictor"],
                    text_encoder=assets["text_encoder"],
                    vector_estimator=assets["vector_estimator"],
                    vocoder=assets["vocoder"],
                    tts_json=assets["tts_json"],
                    unicode_indexer=assets["unicode_indexer"],
                    voice_style=assets["voice_style"],
                ),
                provider=self.config.device,
                num_threads=self.config.num_threads,
                debug=False,
            )
            rule_fsts = ""
        else:
            model_config = sherpa_onnx.OfflineTtsModelConfig(
                vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                    model=assets["model"],
                    lexicon=assets["lexicon"],
                    tokens=assets["tokens"],
                    data_dir=assets["data_dir"],
                ),
                provider=self.config.device,
                num_threads=self.config.num_threads,
                debug=False,
            )
            rule_fsts = ",".join(assets["rule_fsts"])

        tts_config = sherpa_onnx.OfflineTtsConfig(
            model=model_config,
            rule_fsts=rule_fsts,
            max_num_sentences=1,
        )
        if not tts_config.validate():
            raise ValueError("Invalid sherpa-onnx TTS configuration; check cached model assets")
        self._tts = sherpa_onnx.OfflineTts(tts_config)

    def reset(self) -> None:
        pass

    def close(self) -> None:
        self._tts = None

    def synthesize(self, request: TtsRequest) -> TtsUpdate:
        if self._tts is None:
            self.load()
        import sherpa_onnx

        generation = sherpa_onnx.GenerationConfig()
        generation.sid = self.config.speaker_id
        generation.speed = self.config.speed
        if self._model_kind == "supertonic":
            generation.num_steps = self.config.num_steps
            generation.extra["lang"] = request.language
        started = monotonic()
        audio = self._tts.generate(request.synthesis_text, generation)
        samples = np.asarray(audio.samples, dtype=np.float32)
        if samples.size == 0:
            raise RuntimeError("sherpa-onnx generated empty TTS audio")
        return TtsUpdate(
            samples=samples,
            sample_rate=int(audio.sample_rate),
            text=request.text,
            language=request.language,
            source_revision=request.source_revision,
            is_final=request.is_final,
            phrase_id=request.phrase_id,
            started_at=started,
            source_is_final=request.source_is_final,
            spoken_text=request.spoken_text,
        )


class FakeTtsBackend:
    """Small deterministic tone generator for contract and pipeline tests."""

    def __init__(self, config: TtsConfig) -> None:
        self.config = config

    def load(self) -> None:
        pass

    def reset(self) -> None:
        pass

    def close(self) -> None:
        pass

    def synthesize(self, request: TtsRequest) -> TtsUpdate:
        started = monotonic()
        sample_rate = 16_000
        duration = max(
            0.08,
            min(0.5, len(request.synthesis_text) * 0.015),
        )
        time_axis = np.arange(int(sample_rate * duration), dtype=np.float32) / sample_rate
        samples = (0.02 * np.sin(2 * np.pi * 440 * time_axis)).astype(np.float32)
        return TtsUpdate(
            samples=samples,
            sample_rate=sample_rate,
            text=request.text,
            language=request.language,
            source_revision=request.source_revision,
            is_final=request.is_final,
            phrase_id=request.phrase_id,
            started_at=started,
            source_is_final=request.source_is_final,
            spoken_text=request.spoken_text,
        )
