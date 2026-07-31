from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any

from onevoice.config import TerminologyMtConfig, TranslationConfig
from onevoice.models import TranslationRequest, TranslationUpdate
from onevoice.terminology import TerminologyManager
from onevoice.terminology.runtime import MtTerminologyStats, TerminologyMtRuntime
from onevoice.text import (
    ends_phrase,
    restore_terminal_punctuation,
    split_sentences,
)


OPUS_PAIR_MODELS = {
    ("vi", "en"): "Helsinki-NLP/opus-mt-vi-en",
    ("en", "vi"): "Helsinki-NLP/opus-mt-en-vi",
    ("zh", "en"): "Helsinki-NLP/opus-mt-zh-en",
    ("en", "zh"): "Helsinki-NLP/opus-mt-en-zh",
    ("ko", "en"): "Helsinki-NLP/opus-mt-ko-en",
    ("en", "ko"): "Helsinki-NLP/opus-mt-tc-big-en-ko",
    ("zh", "vi"): "Helsinki-NLP/opus-mt-zh-vi",
}
SUPPORTED_TRANSLATION_LANGUAGES = frozenset(("vi", "en", "zh", "ko"))


def validate_translation_selection(config: TranslationConfig, backend: str | None = None) -> None:
    selected_backend = backend or config.backend
    source = config.source_language
    target = config.target_language
    if source not in (*SUPPORTED_TRANSLATION_LANGUAGES, "auto"):
        raise ValueError(f"Unsupported translation source language: {source!r}")
    if target not in SUPPORTED_TRANSLATION_LANGUAGES:
        raise ValueError(f"Unsupported translation target language: {target!r}")
    if source != "auto" and source == target:
        raise ValueError(f"Translation source and target must differ, got {source!r}")
    if not config.model.strip():
        raise ValueError(f"Translation backend {selected_backend!r} requires a model name")
    if selected_backend == "opus_ct2" and config.model != "opus-auto":
        raise ValueError(
            "OPUS CTranslate2 uses its built-in pair router; model must be 'opus-auto'"
        )

_TARGET_PREFIXES = {
    "vi": (">>vie<<", ">>vie_Hani<<"),
    "zh": (">>cmn_Hans<<", ">>zho_Hans<<", ">>cmn<<", ">>zho<<"),
    "ko": (">>kor_Hang<<", ">>kor<<"),
}


@dataclass(slots=True)
class _OpusEngine:
    tokenizer: Any
    translator: Any


@dataclass(frozen=True, slots=True)
class _CachedSentence:
    text: str
    stats: MtTerminologyStats


class _SentenceTranslationCache:
    """Small exact-match cache scoped by pipeline utterance identity."""

    def __init__(self, max_entries: int = 512) -> None:
        self.max_entries = max_entries
        self._items: OrderedDict[
            tuple[tuple[int, int], str, str, str],
            _CachedSentence,
        ] = OrderedDict()

    def get(
        self,
        stream_id: tuple[int, int] | None,
        source_language: str,
        target_language: str,
        source_sentence: str,
    ) -> _CachedSentence | None:
        if stream_id is None:
            return None
        key = (
            stream_id,
            source_language,
            target_language,
            source_sentence,
        )
        cached = self._items.get(key)
        if cached is not None:
            self._items.move_to_end(key)
        return cached

    def put(
        self,
        stream_id: tuple[int, int] | None,
        source_language: str,
        target_language: str,
        source_sentence: str,
        value: _CachedSentence,
    ) -> None:
        if stream_id is None:
            return
        key = (
            stream_id,
            source_language,
            target_language,
            source_sentence,
        )
        self._items[key] = value
        self._items.move_to_end(key)
        while len(self._items) > self.max_entries:
            self._items.popitem(last=False)

    def discard_stream(self, stream_id: tuple[int, int] | None) -> None:
        if stream_id is None:
            return
        for key in tuple(self._items):
            if key[0] == stream_id:
                del self._items[key]

    def clear(self) -> None:
        self._items.clear()


class OpusMtCTranslate2Backend:
    """INT8 OPUS-MT routing backend optimized for low-latency CPU inference.

    English pairs use one Marian model. Other language pairs route through
    English and therefore use two small models while preserving all 12 product
    directions. Models are converted once and reused from the local CT2 cache.
    """

    def __init__(self, config: TranslationConfig) -> None:
        validate_translation_selection(config, "opus_ct2")
        self.config = config
        self._ct2: Any = None
        self._auto_tokenizer: Any = None
        self._snapshot_download: Any = None
        self._converter_type: Any = None
        self._engines: dict[tuple[str, str], _OpusEngine] = {}
        self._terminology: TerminologyMtRuntime | None = None
        self._sentence_cache = _SentenceTranslationCache()

    def configure_terminology(
        self,
        manager: TerminologyManager,
        *,
        domain: str | None,
        config: TerminologyMtConfig,
    ) -> None:
        self._sentence_cache.clear()
        self._terminology = TerminologyMtRuntime(
            manager,
            domain=domain,
            config=config,
        )

    @staticmethod
    def route(source: str, target: str) -> tuple[tuple[str, str], ...]:
        if source == target:
            return ()
        pair = (source, target)
        if pair in OPUS_PAIR_MODELS:
            return (pair,)
        if source not in ("vi", "en", "zh", "ko") or target not in ("vi", "en", "zh", "ko"):
            raise ValueError(f"Unsupported OPUS-MT direction: {source}->{target}")
        return ((source, "en"), ("en", target))

    def _import_dependencies(self) -> None:
        if self._ct2 is not None:
            return
        try:
            import ctranslate2
            from ctranslate2.converters import TransformersConverter
            from huggingface_hub import snapshot_download
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Install the 'opus' extra to use fast OPUS-MT: "
                "python -m pip install -e \".[opus]\""
            ) from exc
        self._ct2 = ctranslate2
        self._converter_type = TransformersConverter
        self._snapshot_download = snapshot_download
        self._auto_tokenizer = AutoTokenizer

    def load(self) -> None:
        self._import_dependencies()
        source = self.config.source_language
        target = self.config.target_language
        if source != "auto":
            self._ensure_route(source, target)

    def _cache_root(self) -> Path:
        root = Path(self.config.model_dir) if self.config.model_dir else Path(".cache/onevoice/opus_ct2")
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _download_snapshot(self, model_id: str) -> Path:
        # Download into a real project-local directory. Hugging Face's default
        # cache uses symlinks which require Windows Developer Mode/admin rights.
        source_dir = self._cache_root().parent / "opus_sources" / model_id.replace("/", "--")
        source_dir.mkdir(parents=True, exist_ok=True)
        if self.config.offline:
            has_weights = any(source_dir.glob("*.safetensors")) or (
                source_dir / "pytorch_model.bin"
            ).is_file() or any(source_dir.glob("*.bin.index.json"))
            if has_weights and (source_dir / "config.json").is_file():
                return source_dir
            raise RuntimeError(f"OPUS-MT offline cache is incomplete: {source_dir}")
        common_files = [
            "config.json",
            "generation_config.json",
            "source.spm",
            "target.spm",
            "vocab.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ]
        try:
            snapshot = self._snapshot_download(
                repo_id=model_id,
                local_dir=source_dir,
                allow_patterns=common_files + ["*.safetensors", "*.safetensors.index.json"],
                local_files_only=self.config.offline,
                max_workers=4,
            )
            snapshot_path = Path(snapshot)
            has_weights = any(snapshot_path.glob("*.safetensors"))
            if not has_weights:
                snapshot = self._snapshot_download(
                    repo_id=model_id,
                    local_dir=source_dir,
                    allow_patterns=common_files + ["pytorch_model.bin", "*.bin.index.json"],
                    local_files_only=self.config.offline,
                    max_workers=4,
                )
                snapshot_path = Path(snapshot)
                has_weights = (snapshot_path / "pytorch_model.bin").is_file() or any(
                    snapshot_path.glob("*.bin.index.json")
                )
            if not has_weights:
                raise RuntimeError("model snapshot has no supported weights")
            return snapshot_path
        except Exception as exc:
            mode = "offline cache" if self.config.offline else "Hugging Face Hub"
            raise RuntimeError(f"Could not resolve {model_id} from {mode}: {exc}") from exc

    def _conversion_quantization(self) -> str | None:
        value = self.config.compute_type.lower()
        return None if value in ("auto", "default") else value

    def _load_pair(self, pair: tuple[str, str]) -> _OpusEngine:
        cached = self._engines.get(pair)
        if cached is not None:
            return cached
        self._import_dependencies()
        model_id = OPUS_PAIR_MODELS[pair]
        snapshot = self._download_snapshot(model_id)
        suffix = self._conversion_quantization() or "default"
        output_dir = self._cache_root() / f"{model_id.replace('/', '--')}--{suffix}"
        if not self._ct2.contains_model(str(output_dir)):
            converter = self._converter_type(str(snapshot))
            converter.convert(
                str(output_dir),
                quantization=self._conversion_quantization(),
                force=output_dir.exists(),
            )
        tokenizer = self._auto_tokenizer.from_pretrained(snapshot, local_files_only=True)
        translator = self._ct2.Translator(
            str(output_dir),
            device=self.config.device,
            compute_type=self.config.compute_type,
            inter_threads=1,
        )
        engine = _OpusEngine(tokenizer=tokenizer, translator=translator)
        self._engines[pair] = engine
        return engine

    def _ensure_route(self, source: str, target: str) -> tuple[tuple[str, str], ...]:
        route = self.route(source, target)
        for pair in route:
            self._load_pair(pair)
        return route

    @staticmethod
    def _with_target_prefix(tokenizer: Any, text: str, target: str) -> str:
        supported = set(getattr(tokenizer, "supported_language_codes", ()) or ())
        for prefix in _TARGET_PREFIXES.get(target, ()):
            if prefix in supported:
                return f"{prefix} {text}"
        return text

    def _translate_once(self, text: str, pair: tuple[str, str]) -> str:
        engine = self._engines[pair]
        source_text = self._with_target_prefix(engine.tokenizer, text, pair[1])
        token_ids = engine.tokenizer.encode(source_text)
        source_tokens = engine.tokenizer.convert_ids_to_tokens(token_ids)
        results = engine.translator.translate_batch(
            [source_tokens],
            beam_size=1,
            max_decoding_length=self.config.max_new_tokens,
            return_scores=False,
        )
        target_tokens = results[0].hypotheses[0]
        target_ids = engine.tokenizer.convert_tokens_to_ids(target_tokens)
        return engine.tokenizer.decode(target_ids, skip_special_tokens=True).strip()

    def reset(self) -> None:
        self._sentence_cache.clear()

    def close(self) -> None:
        for engine in self._engines.values():
            unload = getattr(engine.translator, "unload_model", None)
            if callable(unload):
                unload()
        self._engines.clear()
        self._sentence_cache.clear()

    def translate(self, request: TranslationRequest) -> TranslationUpdate:
        route = self._ensure_route(request.source_language, request.target_language)
        started = monotonic()
        translated_sentences: list[str] = []
        terminology_stats = MtTerminologyStats()
        sentence_cache_hits = 0
        source_sentences = (
            split_sentences(request.text, request.source_language)
            if request.is_final or self._terminology is not None
            else (request.text,)
        ) or (request.text,)
        try:
            for source_sentence in source_sentences:
                is_complete = ends_phrase(
                    source_sentence, request.source_language
                )
                cached = (
                    self._sentence_cache.get(
                        request.stream_id,
                        request.source_language,
                        request.target_language,
                        source_sentence,
                    )
                    if self._terminology is not None and is_complete
                    else None
                )
                if cached is not None:
                    translated_sentences.append(cached.text)
                    terminology_stats += cached.stats
                    sentence_cache_hits += 1
                    continue

                text = source_sentence
                chunk_stats = MtTerminologyStats()
                for pair in route:
                    if self._terminology is None:
                        text = self._translate_once(text, pair)
                    else:
                        text, hop_stats = self._terminology.translate_hop(
                            text,
                            pair[0],
                            pair[1],
                            lambda value, active_pair=pair: self._translate_once(
                                value, active_pair
                            ),
                        )
                        chunk_stats += hop_stats
                if request.is_final or is_complete:
                    text = restore_terminal_punctuation(
                        source_sentence,
                        text,
                        request.source_language,
                        request.target_language,
                    )
                translated_sentences.append(text)
                terminology_stats += chunk_stats
                if self._terminology is not None and is_complete:
                    self._sentence_cache.put(
                        request.stream_id,
                        request.source_language,
                        request.target_language,
                        source_sentence,
                        _CachedSentence(text, chunk_stats),
                    )
        finally:
            if request.is_final:
                self._sentence_cache.discard_stream(request.stream_id)
        separator = "" if request.target_language == "zh" else " "
        text = separator.join(translated_sentences)
        return TranslationUpdate(
            text=text,
            source_text=request.text,
            source_language=request.source_language,
            target_language=request.target_language,
            source_revision=request.source_revision,
            is_final=request.is_final,
            started_at=started,
            terminology_matches=terminology_stats.matches,
            terminology_hard_matches=terminology_stats.hard_matches,
            terminology_expected_placeholders=(
                terminology_stats.expected_placeholders
            ),
            terminology_retries=terminology_stats.retries,
            terminology_fallbacks=terminology_stats.fallbacks,
            terminology_hops=terminology_stats.hops,
            sentence_cache_hits=sentence_cache_hits,
        )


@dataclass(slots=True)
class _M2M100Engine:
    tokenizer: Any
    translator: Any


class M2M100Backend:
    """Multilingual M2M100 inference through a converted CTranslate2 model."""

    def __init__(self, config: TranslationConfig) -> None:
        validate_translation_selection(config, "m2m100")
        self.config = config
        self._ct2: Any = None
        self._auto_tokenizer: Any = None
        self._snapshot_download: Any = None
        self._converter_type: Any = None
        self._engine: _M2M100Engine | None = None
        self._terminology: TerminologyMtRuntime | None = None
        self._sentence_cache = _SentenceTranslationCache()

    def configure_terminology(
        self,
        manager: TerminologyManager,
        *,
        domain: str | None,
        config: TerminologyMtConfig,
    ) -> None:
        self._sentence_cache.clear()
        self._terminology = TerminologyMtRuntime(
            manager,
            domain=domain,
            config=config,
        )

    def _import_dependencies(self) -> None:
        if self._ct2 is not None:
            return
        try:
            import ctranslate2
            from ctranslate2.converters import TransformersConverter
            from huggingface_hub import snapshot_download
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Install the 'models' extra to use M2M100 with CTranslate2: "
                "python -m pip install -e \".[models]\""
            ) from exc
        self._ct2 = ctranslate2
        self._converter_type = TransformersConverter
        self._snapshot_download = snapshot_download
        self._auto_tokenizer = AutoTokenizer

    def load(self) -> None:
        self._load_engine()

    def _cache_root(self) -> Path:
        root = (
            Path(self.config.model_dir)
            if self.config.model_dir
            else Path(".cache/onevoice/m2m100_ct2")
        )
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _download_snapshot(self) -> Path:
        model_id = self.config.model
        source_dir = (
            self._cache_root().parent
            / "m2m100_sources"
            / model_id.replace("/", "--")
        )
        source_dir.mkdir(parents=True, exist_ok=True)
        common_files = [
            "config.json",
            "generation_config.json",
            "sentencepiece.bpe.model",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
        ]
        if self.config.offline:
            has_weights = any(source_dir.glob("*.safetensors")) or (
                source_dir / "pytorch_model.bin"
            ).is_file() or any(source_dir.glob("*.bin.index.json"))
            has_tokenizer = (source_dir / "sentencepiece.bpe.model").is_file()
            if has_weights and has_tokenizer and (source_dir / "config.json").is_file():
                return source_dir
            raise RuntimeError(f"M2M100 offline cache is incomplete: {source_dir}")
        try:
            snapshot = self._snapshot_download(
                repo_id=model_id,
                local_dir=source_dir,
                allow_patterns=common_files
                + ["*.safetensors", "*.safetensors.index.json"],
                local_files_only=False,
                max_workers=4,
            )
            snapshot_path = Path(snapshot)
            has_weights = any(snapshot_path.glob("*.safetensors"))
            if not has_weights:
                snapshot = self._snapshot_download(
                    repo_id=model_id,
                    local_dir=source_dir,
                    allow_patterns=common_files
                    + ["pytorch_model.bin", "*.bin.index.json"],
                    local_files_only=False,
                    max_workers=4,
                )
                snapshot_path = Path(snapshot)
                has_weights = (snapshot_path / "pytorch_model.bin").is_file() or any(
                    snapshot_path.glob("*.bin.index.json")
                )
            if not has_weights:
                raise RuntimeError("model snapshot has no supported weights")
            return snapshot_path
        except Exception as exc:
            raise RuntimeError(
                f"Could not resolve {model_id} from Hugging Face Hub: {exc}"
            ) from exc

    def _conversion_quantization(self) -> str | None:
        value = self.config.compute_type.lower()
        return None if value in ("auto", "default") else value

    def _load_engine(self) -> _M2M100Engine:
        if self._engine is not None:
            return self._engine
        self._import_dependencies()
        snapshot = self._download_snapshot()
        suffix = self._conversion_quantization() or "default"
        output_dir = self._cache_root() / (
            f"{self.config.model.replace('/', '--')}--{suffix}"
        )
        if not self._ct2.contains_model(str(output_dir)):
            converter = self._converter_type(str(snapshot))
            converter.convert(
                str(output_dir),
                quantization=self._conversion_quantization(),
                force=output_dir.exists(),
            )
        tokenizer = self._auto_tokenizer.from_pretrained(
            snapshot, local_files_only=True
        )
        translator = self._ct2.Translator(
            str(output_dir),
            device=self.config.device,
            compute_type=self.config.compute_type,
            inter_threads=1,
        )
        self._engine = _M2M100Engine(tokenizer=tokenizer, translator=translator)
        return self._engine

    def _translate_once(self, text: str, source: str, target: str) -> str:
        engine = self._load_engine()
        engine.tokenizer.src_lang = source
        source_ids = engine.tokenizer.encode(text)
        source_tokens = engine.tokenizer.convert_ids_to_tokens(source_ids)
        try:
            target_token = engine.tokenizer.lang_code_to_token[target]
        except (AttributeError, KeyError) as exc:
            raise ValueError(f"M2M100 tokenizer does not support target language: {target}") from exc
        results = engine.translator.translate_batch(
            [source_tokens],
            target_prefix=[[target_token]],
            beam_size=1,
            max_decoding_length=self.config.max_new_tokens + 1,
            return_scores=False,
        )
        # CTranslate2 includes the forced language token in the hypothesis.
        target_tokens = results[0].hypotheses[0][1:]
        target_ids = engine.tokenizer.convert_tokens_to_ids(target_tokens)
        return engine.tokenizer.decode(target_ids, skip_special_tokens=True).strip()

    def reset(self) -> None:
        self._sentence_cache.clear()

    def close(self) -> None:
        if self._engine is not None:
            unload = getattr(self._engine.translator, "unload_model", None)
            if callable(unload):
                unload()
        self._engine = None
        self._sentence_cache.clear()

    def translate(self, request: TranslationRequest) -> TranslationUpdate:
        started = monotonic()
        translated_sentences: list[str] = []
        terminology_stats = MtTerminologyStats()
        sentence_cache_hits = 0
        source_sentences = (
            split_sentences(request.text, request.source_language)
            if request.is_final or self._terminology is not None
            else (request.text,)
        ) or (request.text,)
        try:
            for source_sentence in source_sentences:
                is_complete = ends_phrase(
                    source_sentence, request.source_language
                )
                cached = (
                    self._sentence_cache.get(
                        request.stream_id,
                        request.source_language,
                        request.target_language,
                        source_sentence,
                    )
                    if self._terminology is not None and is_complete
                    else None
                )
                if cached is not None:
                    translated_sentences.append(cached.text)
                    terminology_stats += cached.stats
                    sentence_cache_hits += 1
                    continue

                if self._terminology is None:
                    text = self._translate_once(
                        source_sentence,
                        request.source_language,
                        request.target_language,
                    )
                    chunk_stats = MtTerminologyStats()
                else:
                    text, chunk_stats = self._terminology.translate_hop(
                        source_sentence,
                        request.source_language,
                        request.target_language,
                        lambda value: self._translate_once(
                            value,
                            request.source_language,
                            request.target_language,
                        ),
                    )
                if request.is_final or is_complete:
                    text = restore_terminal_punctuation(
                        source_sentence,
                        text,
                        request.source_language,
                        request.target_language,
                    )
                translated_sentences.append(text)
                terminology_stats += chunk_stats
                if self._terminology is not None and is_complete:
                    self._sentence_cache.put(
                        request.stream_id,
                        request.source_language,
                        request.target_language,
                        source_sentence,
                        _CachedSentence(text, chunk_stats),
                    )
        finally:
            if request.is_final:
                self._sentence_cache.discard_stream(request.stream_id)
        separator = "" if request.target_language == "zh" else " "
        text = separator.join(translated_sentences)
        return TranslationUpdate(
            text=text,
            source_text=request.text,
            source_language=request.source_language,
            target_language=request.target_language,
            source_revision=request.source_revision,
            is_final=request.is_final,
            started_at=started,
            terminology_matches=terminology_stats.matches,
            terminology_hard_matches=terminology_stats.hard_matches,
            terminology_expected_placeholders=(
                terminology_stats.expected_placeholders
            ),
            terminology_retries=terminology_stats.retries,
            terminology_fallbacks=terminology_stats.fallbacks,
            terminology_hops=terminology_stats.hops,
            sentence_cache_hits=sentence_cache_hits,
        )


class FakeTranslationBackend:
    def __init__(self, config: TranslationConfig) -> None:
        validate_translation_selection(config, "fake")
        self.config = config
        self._terminology: TerminologyMtRuntime | None = None

    def configure_terminology(
        self,
        manager: TerminologyManager,
        *,
        domain: str | None,
        config: TerminologyMtConfig,
    ) -> None:
        self._terminology = TerminologyMtRuntime(
            manager,
            domain=domain,
            config=config,
        )

    def load(self) -> None:
        pass

    def reset(self) -> None:
        pass

    def close(self) -> None:
        pass

    def translate(self, request: TranslationRequest) -> TranslationUpdate:
        started = monotonic()
        stats = MtTerminologyStats()
        if self._terminology is None:
            text = f"[{request.target_language}] {request.text}"
        else:
            text, stats = self._terminology.translate_hop(
                request.text,
                request.source_language,
                request.target_language,
                lambda value: f"[{request.target_language}] {value}",
            )
        return TranslationUpdate(
            text=text,
            source_text=request.text,
            source_language=request.source_language,
            target_language=request.target_language,
            source_revision=request.source_revision,
            is_final=request.is_final,
            started_at=started,
            terminology_matches=stats.matches,
            terminology_hard_matches=stats.hard_matches,
            terminology_expected_placeholders=stats.expected_placeholders,
            terminology_retries=stats.retries,
            terminology_fallbacks=stats.fallbacks,
            terminology_hops=stats.hops,
        )
