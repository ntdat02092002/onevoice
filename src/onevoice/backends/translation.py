from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any

from onevoice.config import TranslationConfig
from onevoice.models import TranslationRequest, TranslationUpdate
from onevoice.text import restore_terminal_punctuation, split_sentences


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
        # OPUS/Marian translation is stateless between requests.
        pass

    def close(self) -> None:
        for engine in self._engines.values():
            unload = getattr(engine.translator, "unload_model", None)
            if callable(unload):
                unload()
        self._engines.clear()

    def translate(self, request: TranslationRequest) -> TranslationUpdate:
        route = self._ensure_route(request.source_language, request.target_language)
        started = monotonic()
        translated_sentences: list[str] = []
        source_sentences = (
            split_sentences(request.text, request.source_language)
            if request.is_final
            else (request.text,)
        ) or (request.text,)
        for source_sentence in source_sentences:
            text = source_sentence
            for pair in route:
                text = self._translate_once(text, pair)
            translated_sentences.append(
                restore_terminal_punctuation(
                    source_sentence,
                    text,
                    request.source_language,
                    request.target_language,
                )
                if request.is_final
                else text
            )
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
        )


class M2M100Backend:
    def __init__(self, config: TranslationConfig) -> None:
        validate_translation_selection(config, "m2m100")
        self.config = config
        self._model = None
        self._tokenizer = None
        self._torch = None
        self._device = config.device

    def load(self) -> None:
        try:
            import torch
            from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
        except ImportError as exc:
            raise RuntimeError("Install the 'models' extra to use M2M100") from exc
        if self._device == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tokenizer = M2M100Tokenizer.from_pretrained(
            self.config.model, local_files_only=self.config.offline
        )
        self._model = M2M100ForConditionalGeneration.from_pretrained(
            self.config.model, local_files_only=self.config.offline
        ).to(self._device)
        self._model.eval()
        self._torch = torch

    def reset(self) -> None:
        pass

    def close(self) -> None:
        self._model = None
        self._tokenizer = None
        self._torch = None

    def translate(self, request: TranslationRequest) -> TranslationUpdate:
        if self._model is None or self._tokenizer is None or self._torch is None:
            self.load()
        started = monotonic()
        translated_sentences: list[str] = []
        source_sentences = (
            split_sentences(request.text, request.source_language)
            if request.is_final
            else (request.text,)
        ) or (request.text,)
        for source_sentence in source_sentences:
            self._tokenizer.src_lang = request.source_language
            encoded = self._tokenizer(source_sentence, return_tensors="pt", truncation=True)
            encoded = {key: value.to(self._device) for key, value in encoded.items()}
            with self._torch.inference_mode():
                generated = self._model.generate(
                    **encoded,
                    forced_bos_token_id=self._tokenizer.get_lang_id(request.target_language),
                    max_new_tokens=self.config.max_new_tokens,
                    num_beams=1,
                )
            text = self._tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
            translated_sentences.append(
                restore_terminal_punctuation(
                    source_sentence,
                    text,
                    request.source_language,
                    request.target_language,
                )
                if request.is_final
                else text
            )
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
        )


class FakeTranslationBackend:
    def __init__(self, config: TranslationConfig) -> None:
        validate_translation_selection(config, "fake")
        self.config = config

    def load(self) -> None:
        pass

    def reset(self) -> None:
        pass

    def close(self) -> None:
        pass

    def translate(self, request: TranslationRequest) -> TranslationUpdate:
        started = monotonic()
        return TranslationUpdate(
            text=f"[{request.target_language}] {request.text}",
            source_text=request.text,
            source_language=request.source_language,
            target_language=request.target_language,
            source_revision=request.source_revision,
            is_final=request.is_final,
            started_at=started,
        )
