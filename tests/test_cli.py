from pathlib import Path

from onevoice.cli import build_parser


def test_cli_accepts_terminology_preflight_options() -> None:
    args = build_parser().parse_args(
        [
            "sample.wav",
            "--terminology-bundle",
            "assets/terminology/sample.yaml",
            "--terminology-domain",
            "factory-safety",
        ]
    )

    assert args.terminology_bundle == Path(
        "assets/terminology/sample.yaml"
    )
    assert args.terminology_domain == "factory-safety"
