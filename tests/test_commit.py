from time import monotonic

from onevoice.backends.commit import LocalAgreementCommitter
from onevoice.config import CommitConfig
from onevoice.models import AsrUpdate
from onevoice.text import tokenize_text


def update(text: str, revision: int, final: bool = False, language: str = "en") -> AsrUpdate:
    return AsrUpdate(
        text=text,
        language=language,
        confidence=1.0,
        revision=revision,
        is_final=final,
        started_at=monotonic(),
        tokens=tokenize_text(text, language),
    )


def test_local_agreement_is_monotonic_and_holds_last_token() -> None:
    committer = LocalAgreementCommitter(CommitConfig(agreement_updates=2, hold_tokens=1))
    assert committer.update(update("hello brave", 1)) is None
    committed = committer.update(update("hello brave world", 2))
    assert committed is not None
    assert committed.text == "hello"

    # A revised hypothesis cannot retract already committed output.
    assert committer.update(update("different ending", 3)) is None
    final = committer.update(update("hello brave world.", 4, final=True))
    assert final is not None
    assert final.text == "hello brave world."
    assert final.is_final


def test_chinese_final_flush() -> None:
    committer = LocalAgreementCommitter(CommitConfig())
    result = committer.update(update("请检查传送带。", 1, final=True, language="zh"))
    assert result is not None
    assert result.text == "请检查传送带。"


def test_final_revision_does_not_discard_visible_draft_suffix() -> None:
    committer = LocalAgreementCommitter(CommitConfig(agreement_updates=2, hold_tokens=1))
    draft = "investor confidence was bolstered by government data, indicating inflation rose."
    assert committer.update(update(draft, 1)) is None
    committed = committer.update(update(draft, 2))
    assert committed is not None
    assert committed.text.endswith("inflation rose")

    # The final pass revises the beginning. The previous implementation fell
    # back to the committed prefix and lost the visible sentence suffix.
    final = committer.update(
        update(
            "confidence was bolstered by government data, indicating inflation rose.",
            3,
            final=True,
        )
    )
    assert final is not None
    assert final.is_final
    assert final.text == draft
