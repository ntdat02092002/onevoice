from pathlib import Path

from onevoice.terminology import (
    TerminologyManager,
    prepare_terminology_bundle,
)


SAMPLE_BUNDLE = Path(
    "assets/terminology/factory-sample-v1/terminology.yaml"
)


def test_manager_reuses_immutable_compiled_profile() -> None:
    manager = TerminologyManager.from_path(SAMPLE_BUNDLE)

    first = manager.activate(
        domain="factory-safety",
        source_language="en",
        target_language="vi",
    )
    second = manager.activate(
        domain="factory-safety",
        source_language="en",
        target_language="vi",
    )

    assert first is second


def test_preflight_validates_and_compiles_bundle_with_relative_identity() -> None:
    manager, build = prepare_terminology_bundle(
        SAMPLE_BUNDLE,
        domain="factory-safety",
        source_language="en",
        target_language="vi",
    )

    assert manager.bundle.bundle_id == "factory-sample-v1"
    assert build.bundle_id == "factory-sample-v1"
    assert build.schema_version == 1
    assert build.bundle_path == SAMPLE_BUNDLE.as_posix()
    assert len(build.bundle_sha256) == 64
    assert build.profile_count == 1
    assert build.entry_count == 2
    assert build.asr_term_count > 0
    assert build.mt_binding_count == 2
    assert build.profiles[0].mt_route == ("en", "vi")


def test_preflight_compiles_auto_source_profiles() -> None:
    _, build = prepare_terminology_bundle(
        SAMPLE_BUNDLE,
        domain="factory-maintenance",
        source_language="auto",
        target_language="vi",
    )

    assert tuple(
        profile.source_language for profile in build.profiles
    ) == ("en", "zh", "ko")


def test_preflight_compiles_resolved_pivot_route() -> None:
    def pivot(source: str, target: str):
        return ((source, "en"), ("en", target))

    _, build = prepare_terminology_bundle(
        SAMPLE_BUNDLE,
        domain="factory-maintenance",
        source_language="vi",
        target_language="ko",
        route_resolver=pivot,
    )

    profile = build.profiles[0]
    assert profile.mt_route == ("vi", "en", "ko")
    assert profile.mt_binding_count == profile.entry_count * 2
