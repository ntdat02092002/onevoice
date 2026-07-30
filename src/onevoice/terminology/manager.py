from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .loader import load_bundle
from .profile import TerminologyProfile, build_profile
from .schema import TerminologyBundle


class TerminologyManager:
    def __init__(
        self,
        bundle: TerminologyBundle,
        *,
        case_sensitive_for_codes: bool = True,
    ) -> None:
        self.bundle = bundle
        self.case_sensitive_for_codes = case_sensitive_for_codes

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        case_sensitive_for_codes: bool = True,
    ) -> "TerminologyManager":
        return cls(
            load_bundle(path),
            case_sensitive_for_codes=case_sensitive_for_codes,
        )

    def activate(
        self,
        *,
        domain: str | None,
        source_language: str,
        target_language: str,
        mt_route: Sequence[str] | None = None,
        asr_model_id: str | None = None,
        tts_model_id: str | None = None,
    ) -> TerminologyProfile:
        return build_profile(
            self.bundle,
            domain=domain,
            source_language=source_language,
            target_language=target_language,
            mt_route=mt_route,
            asr_model_id=asr_model_id,
            tts_model_id=tts_model_id,
            case_sensitive_for_codes=self.case_sensitive_for_codes,
        )
