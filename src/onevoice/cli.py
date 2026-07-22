from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .audio import iter_audio_file
from .config import load_config
from .models import EventType
from .pipeline import RealtimePipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OneVoice realtime file smoke test")
    parser.add_argument("audio", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--source", choices=("auto", "vi", "en", "zh", "ko"), default="vi")
    parser.add_argument("--target", choices=("vi", "en", "zh", "ko"), default="en")
    parser.add_argument("--asr-backend", choices=("moonshine", "dolphin", "faster_whisper", "fake"))
    parser.add_argument("--asr-model", help="Model/architecture name for the selected ASR backend")
    parser.add_argument("--mt-backend", choices=("opus_ct2", "m2m100", "fake"))
    parser.add_argument("--tts", action="store_true", help="Enable phrase-level translated speech")
    parser.add_argument("--tts-backend", choices=("sherpa_onnx", "fake"), default="sherpa_onnx")
    parser.add_argument("--tts-model-dir", type=Path, help="VITS/Piper asset directory")
    parser.add_argument("--realtime", action="store_true", help="Pace input at the original audio rate")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_config(args.config)
    config.asr.language = args.source
    config.translation.source_language = args.source
    config.translation.target_language = args.target
    if args.asr_backend:
        config.asr.backend = args.asr_backend
        if not args.asr_model:
            config.asr.model = {
                "moonshine": "auto",
                "dolphin": "base",
                "faster_whisper": "base",
                "fake": "fake",
            }[args.asr_backend]
    if args.asr_model:
        config.asr.model = args.asr_model
    if args.mt_backend:
        config.translation.backend = args.mt_backend
        config.translation.model = (
            "facebook/m2m100_418M" if args.mt_backend == "m2m100" else "opus-auto"
        )
    if args.tts:
        config.tts.enabled = True
        config.tts.backend = args.tts_backend
        config.tts.model_dir = str(args.tts_model_dir) if args.tts_model_dir else None
    pipeline = RealtimePipeline(config)
    pipeline.start()

    def write_event(event) -> None:
        print(json.dumps(event.to_dict(), ensure_ascii=False, default=str), flush=True)
        if event.type in (EventType.TTS_PARTIAL, EventType.TTS_FINAL):
            pipeline.acknowledge_tts(event.payload.phrase_id)

    try:
        for chunk in iter_audio_file(args.audio, config.audio.sample_rate, config.audio.frame_ms):
            pipeline.push_audio(chunk)
            if args.realtime:
                time.sleep(chunk.duration_seconds)
            for event in pipeline.poll_events():
                write_event(event)
        pipeline.finish()
        pipeline.wait_until_idle(timeout=120)
        for event in pipeline.poll_events(1000):
            write_event(event)
    finally:
        pipeline.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
