from onevoice.text import count_complete_sentences, detokenize, longest_common_prefix, tokenize_text


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
