import pytest

from onevoice.text import (
    count_complete_sentences,
    detokenize,
    ends_phrase,
    longest_common_prefix,
    restore_terminal_punctuation,
    split_sentences,
    tokenize_text,
)


def test_tokenize_latin_and_vietnamese() -> None:
    tokens = tokenize_text("Dừng máy ngay, an toàn!", "vi")
    assert tokens == ("Dừng", "máy", "ngay", ",", "an", "toàn", "!")
    assert detokenize(tokens, "vi") == "Dừng máy ngay, an toàn!"


def test_tokenize_chinese_without_spaces() -> None:
    tokens = tokenize_text("请检查传送带。", "zh")
    assert tokens == tuple("请检查传送带。")
    assert detokenize(tokens, "zh") == "请检查传送带。"


def test_tokenize_korean_and_english() -> None:
    assert tokenize_text("통로를 비워 주세요.", "ko")[-1] == "."
    assert tokenize_text("safety helmet", "en") == ("safety", "helmet")


def test_longest_common_prefix() -> None:
    assert longest_common_prefix((("a", "b", "x"), ("a", "b", "y"))) == ("a", "b")


def test_count_complete_sentences_for_supported_punctuation() -> None:
    assert count_complete_sentences(("One", ".", "Two", "?", "draft")) == 2
    assert count_complete_sentences(("第", "一", "句", "。", "第", "二", "句", "！")) == 2
    assert count_complete_sentences(("Really", "?", "!")) == 1


@pytest.mark.parametrize(
    "text",
    [
        "这是一个句子。",
        "这是一个问题？",
        "这是一个句子。”",
        "「これは文です。」",
        "한국어 문장입니다.",
        "This is a sentence.",
        "话还没有说完…",
    ],
)
def test_ends_phrase_supports_ascii_cjk_and_closing_quotes(text: str) -> None:
    assert ends_phrase(text)


def test_cjk_sentence_boundaries_are_shared_by_split_and_count() -> None:
    text = "第一句。”第二句？第三句！"

    assert split_sentences(text, "zh") == ("第一句。”", "第二句？", "第三句！")
    assert count_complete_sentences(text, "zh") == 3


@pytest.mark.parametrize("text", ["这只是分句，", "这还没有结束；"])
def test_cjk_comma_and_semicolon_are_not_sentence_endpoints(text: str) -> None:
    assert not ends_phrase(text, "zh")
    assert count_complete_sentences(text, "zh") == 0


def test_split_sentences_retains_boundaries_and_restores_dropped_mark() -> None:
    assert split_sentences("First sentence. Second one? trailing", "en") == (
        "First sentence.",
        "Second one?",
        "trailing",
    )
    assert restore_terminal_punctuation("Complete!", "Hoàn tất", "en", "vi") == "Hoàn tất!"


@pytest.mark.parametrize(
    "text",
    [
        "The Time Machine by H. G. Wells.",
        "Mr. Smith arrived at 7.30 p.m.",
        "Dr. Wells met Prof. Brown.",
        "The value is 3.14.",
        "The office is in the U.S.A.",
        "TP.HCM is a city.",
        "PGS. TS. Nguyễn Văn A phát biểu.",
    ],
)
def test_sentence_splitter_does_not_break_abbreviations_or_decimals(text: str) -> None:
    assert split_sentences(text) == (text,)
    assert count_complete_sentences(text) == 1


def test_acronym_at_sentence_end_splits_before_next_sentence() -> None:
    text = "The office is in the U.S.A. We left."
    assert split_sentences(text) == (
        "The office is in the U.S.A.",
        "We left.",
    )
    assert count_complete_sentences(text) == 2


@pytest.mark.parametrize(
    "closing", ['"', "'", "”", "’", ")", "]", "}", "）", "】", "》", "」", "』"]
)
def test_restore_terminal_punctuation_before_closing_mark(closing: str) -> None:
    source = f"He said {closing}Go.{closing}" if closing in {'"', "'", "”", "’"} else f"Go.{closing}"
    target = f"Anh ấy nói {closing}Đi{closing}" if closing in {'"', "'", "”", "’"} else f"Đi{closing}"
    expected = f"Anh ấy nói {closing}Đi.{closing}" if closing in {'"', "'", "”", "’"} else f"Đi.{closing}"
    assert restore_terminal_punctuation(source, target, "en", "vi") == expected
