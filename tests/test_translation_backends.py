from __future__ import annotations

from types import SimpleNamespace

import pytest

from onevoice.backends.translation import (
    M2M100Backend,
    OpusMtCTranslate2Backend,
    _M2M100Engine,
    _OpusEngine,
)
from onevoice.config import TranslationConfig
from onevoice.models import TranslationRequest


def test_opus_routes_all_twelve_directions() -> None:
    languages = ("vi", "en", "zh", "ko")
    routes = {
        (source, target): OpusMtCTranslate2Backend.route(source, target)
        for source in languages
        for target in languages
        if source != target
    }
    assert len(routes) == 12
    assert routes[("vi", "en")] == (("vi", "en"),)
    assert routes[("vi", "zh")] == (("vi", "en"), ("en", "zh"))
    assert routes[("ko", "vi")] == (("ko", "en"), ("en", "vi"))


def test_opus_pivot_preserves_request_metadata() -> None:
    backend = OpusMtCTranslate2Backend(TranslationConfig(backend="opus_ct2"))
    calls: list[tuple[str, tuple[str, str]]] = []
    backend._ensure_route = lambda source, target: ((source, "en"), ("en", target))

    def translate_once(text: str, pair: tuple[str, str]) -> str:
        calls.append((text, pair))
        return f"{text}:{pair[1]}"

    backend._translate_once = translate_once
    request = TranslationRequest(
        text="xin chào",
        source_language="vi",
        target_language="ko",
        source_revision=7,
        is_final=True,
    )
    update = backend.translate(request)

    assert calls == [
        ("xin chào", ("vi", "en")),
        ("xin chào:en", ("en", "ko")),
    ]
    assert update.text == "xin chào:en:ko"
    assert update.source_revision == 7
    assert update.is_final


class _Tokenizer:
    supported_language_codes = [">>vie<<"]

    def __init__(self) -> None:
        self.encoded = ""

    def encode(self, text: str):
        self.encoded = text
        return [1, 2]

    def convert_ids_to_tokens(self, values):
        return [f"t{value}" for value in values]

    def convert_tokens_to_ids(self, values):
        return [3]

    def decode(self, values, skip_special_tokens: bool):
        assert skip_special_tokens
        return "xin chào"


class _Translator:
    def __init__(self) -> None:
        self.kwargs = {}

    def translate_batch(self, sources, **kwargs):
        self.kwargs = kwargs
        assert sources == [["t1", "t2"]]
        return [SimpleNamespace(hypotheses=[["result"]])]


def test_opus_uses_language_prefix_and_greedy_decode() -> None:
    backend = OpusMtCTranslate2Backend(TranslationConfig(max_new_tokens=64))
    tokenizer = _Tokenizer()
    translator = _Translator()
    backend._engines[("en", "vi")] = _OpusEngine(tokenizer, translator)

    assert backend._translate_once("hello", ("en", "vi")) == "xin chào"
    assert tokenizer.encoded == ">>vie<< hello"
    assert translator.kwargs["beam_size"] == 1
    assert translator.kwargs["max_decoding_length"] == 64


def test_opus_translates_final_sentence_by_sentence_and_restores_punctuation() -> None:
    backend = OpusMtCTranslate2Backend(TranslationConfig())
    calls: list[str] = []
    backend._ensure_route = lambda source, target: ((source, target),)

    def translate_once(text: str, pair: tuple[str, str]) -> str:
        calls.append(text)
        return {"First sentence.": "Câu đầu", "Second sentence?": "Câu sau"}[text]

    backend._translate_once = translate_once
    result = backend.translate(
        TranslationRequest("First sentence. Second sentence?", "en", "vi", 3, True)
    )

    assert calls == ["First sentence.", "Second sentence?"]
    assert result.text == "Câu đầu. Câu sau?"


def test_opus_translates_partial_prefix_once_without_sentence_loop() -> None:
    backend = OpusMtCTranslate2Backend(TranslationConfig())
    calls: list[str] = []
    backend._ensure_route = lambda source, target: ((source, target),)
    backend._translate_once = lambda text, pair: calls.append(text) or "partial result"

    result = backend.translate(
        TranslationRequest("Sentence one. Sentence two. active", "en", "vi", 2, False)
    )

    assert calls == ["Sentence one. Sentence two. active"]
    assert result.text == "partial result"


class _M2MTokenizer:
    lang_code_to_token = {"vi": "__vi__"}

    def __init__(self) -> None:
        self.src_lang = None
        self.encoded = None

    def encode(self, text: str):
        self.encoded = text
        return [1, 2]

    def convert_ids_to_tokens(self, values):
        return [f"s{value}" for value in values]

    def convert_tokens_to_ids(self, values):
        assert values == ["translated"]
        return [3]

    def decode(self, values, skip_special_tokens: bool):
        assert values == [3]
        assert skip_special_tokens
        return "đã dịch"


class _M2MTranslator:
    def __init__(self) -> None:
        self.sources = None
        self.kwargs = None

    def translate_batch(self, sources, **kwargs):
        self.sources = sources
        self.kwargs = kwargs
        return [SimpleNamespace(hypotheses=[["__vi__", "translated"]])]


def test_m2m100_uses_ct2_target_prefix_and_strips_forced_language_token() -> None:
    backend = M2M100Backend(
        TranslationConfig(backend="m2m100", model="facebook/m2m100_418M", max_new_tokens=64)
    )
    tokenizer = _M2MTokenizer()
    translator = _M2MTranslator()
    backend._engine = _M2M100Engine(tokenizer, translator)

    assert backend._translate_once("hello", "en", "vi") == "đã dịch"
    assert tokenizer.src_lang == "en"
    assert tokenizer.encoded == "hello"
    assert translator.sources == [["s1", "s2"]]
    assert translator.kwargs["target_prefix"] == [["__vi__"]]
    assert translator.kwargs["beam_size"] == 1
    assert translator.kwargs["max_decoding_length"] == 65


def test_m2m100_preserves_final_sentence_policy_and_metadata() -> None:
    backend = M2M100Backend(
        TranslationConfig(backend="m2m100", model="facebook/m2m100_418M")
    )
    calls: list[tuple[str, str, str]] = []

    def translate_once(text: str, source: str, target: str) -> str:
        calls.append((text, source, target))
        return {"First.": "Đầu", "Second?": "Sau"}[text]

    backend._translate_once = translate_once
    update = backend.translate(
        TranslationRequest("First. Second?", "en", "vi", 8, True)
    )

    assert calls == [("First.", "en", "vi"), ("Second?", "en", "vi")]
    assert update.text == "Đầu. Sau?"
    assert update.source_revision == 8
    assert update.is_final


def test_translation_capabilities_fail_before_model_loading() -> None:
    with pytest.raises(ValueError, match="source and target must differ"):
        OpusMtCTranslate2Backend(
            TranslationConfig(
                backend="opus_ct2",
                model="opus-auto",
                source_language="vi",
                target_language="vi",
            )
        )
    with pytest.raises(ValueError, match="model must be 'opus-auto'"):
        OpusMtCTranslate2Backend(
            TranslationConfig(backend="opus_ct2", model="some-random-model")
        )
