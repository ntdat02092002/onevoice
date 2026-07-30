from __future__ import annotations

import io
import logging
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from onevoice.audio import AudioFrameNormalizer, RealtimePacer, encode_wav, iter_audio_file
from onevoice.backends.asr import asr_model_options, validate_asr_selection
from onevoice.config import load_config
from onevoice.models import EventType
from onevoice.pipeline import RealtimePipeline
from onevoice.text import ends_phrase


st.set_page_config(page_title="OneVoice Realtime", page_icon="🎙️", layout="wide")
logging.getLogger("streamlit_webrtc").setLevel(logging.WARNING)
logging.getLogger("aioice").setLevel(logging.WARNING)


LANGUAGES = {
    "Tự nhận diện": "auto",
    "Tiếng Việt": "vi",
    "English": "en",
    "中文": "zh",
    "한국어": "ko",
}

REALTIME_PROFILE = Path(__file__).resolve().parents[1] / "config" / "realtime_conversation.yaml"
SAMPLE_TERMINOLOGY_BUNDLE = (
    Path("assets")
    / "terminology"
    / "factory-sample-v1"
    / "terminology.yaml"
)
PROFILE_DEFAULTS = load_config(REALTIME_PROFILE)


def _empty_view() -> dict[str, Any]:
    return {
        "status": "Chưa khởi tạo",
        "draft": "",
        "stable_current": "",
        "committed_history": [],
        "translation_current": "",
        "translation_history": [],
        "language": "-",
        "asr_ms": 0.0,
        "mt_ms": 0.0,
        "tts_ms": 0.0,
        "tts_rtf": 0.0,
        "tts_audio": None,
        "tts_chunks": [],
        "tts_chunk_texts": [],
        "tts_session_chunks": [],
        "tts_session_audio": None,
        "tts_session_wav": None,
        "tts_sample_rate": None,
        "tts_text": "",
        "tts_pending_audio": [],
        "tts_seen_phrase_ids": set(),
        "tts_playing": None,
        "tts_playing_started_at": 0.0,
        "file_session": False,
        "file_complete": False,
        "file_realtime": True,
        "file_name": "onevoice",
        "input_duration_seconds": 0.0,
        "input_started_at": None,
        "input_started_monotonic": None,
        "input_finished_at": None,
        "input_finished_monotonic": None,
        "output_started_at": None,
        "asr_final_monotonic": None,
        "mt_final_monotonic": None,
        "tts_final_monotonic": None,
        "tts_finished_at": None,
        "overall_latency_seconds": None,
        "warnings": [],
        "errors": [],
    }


def _join_history(history: list[str]) -> str:
    return "\n\n".join(item.strip() for item in history if item.strip())


def _format_duration(seconds: float | None, *, milliseconds: bool = False) -> str:
    if seconds is None:
        return "--:--"
    seconds = max(0.0, seconds)
    minutes, remainder = divmod(seconds, 60)
    if milliseconds:
        return f"{int(minutes):02d}:{remainder:06.3f}"
    return f"{int(minutes):02d}:{int(remainder):02d}"


def _format_clock(timestamp: float | None) -> str:
    return "--:--:--" if timestamp is None else time.strftime("%H:%M:%S", time.localtime(timestamp))


def _format_latency(seconds: float | None) -> str:
    if seconds is None:
        return "--"
    seconds = max(0.0, seconds)
    return f"{seconds * 1000:.0f} ms" if seconds < 1 else f"{seconds:.2f} s"


def _elapsed_between(end: float | None, start: float | None) -> float | None:
    if end is None or start is None:
        return None
    return max(0.0, end - start)


if "runtime" not in st.session_state:
    st.session_state.runtime = {
        "pipeline": None,
        "normalizer": AudioFrameNormalizer(),
        "mode": None,
        "feeder": None,
        "view": _empty_view(),
        "was_playing": False,
    }

runtime = st.session_state.runtime
for key, value in _empty_view().items():
    runtime["view"].setdefault(key, value)

st.title("🎙️ OneVoice Realtime Translation")
st.caption("Offline ASR → Stable Prefix → Machine Translation → Phrase TTS")

with st.sidebar:
    st.header("Pipeline")
    st.caption("Profile mặc định: realtime_conversation")
    controls_locked = runtime["pipeline"] is not None
    if controls_locked:
        st.caption(f"TTS emission: {runtime['pipeline'].config.tts.emission_mode}")
    source_label = st.selectbox("Ngôn ngữ nguồn", LANGUAGES, index=1, disabled=controls_locked)
    source = LANGUAGES[source_label]
    target_options = {name: code for name, code in LANGUAGES.items() if code != "auto" and code != source}
    target_label = st.selectbox("Ngôn ngữ đích", target_options, disabled=controls_locked)
    target = target_options[target_label]
    asr_backend = st.selectbox(
        "ASR backend",
        ("moonshine", "sherpa_onnx", "dolphin", "faster_whisper", "fake"),
        disabled=controls_locked,
    )
    asr_models = asr_model_options(asr_backend, source)
    asr_model = st.selectbox("ASR model", asr_models, disabled=controls_locked)
    asr_capability_error = None
    try:
        validate_asr_selection(asr_backend, asr_model, source)
    except ValueError as exc:
        asr_capability_error = str(exc)
        st.error(asr_capability_error)
    mt_backend = st.selectbox("MT backend", ("opus_ct2", "m2m100", "fake"), disabled=controls_locked)
    terminology_enabled = st.checkbox(
        "Bật terminology dictionary",
        value=False,
        disabled=controls_locked,
        help="Bảo vệ thuật ngữ qua MT và khôi phục canonical form theo ngôn ngữ đích.",
    )
    terminology_domain = st.selectbox(
        "Terminology domain",
        ("factory-safety", "factory-maintenance", "test"),
        disabled=controls_locked or not terminology_enabled,
    )
    terminology_bundle = st.text_input(
        "Terminology bundle",
        value=str(SAMPLE_TERMINOLOGY_BUNDLE),
        disabled=controls_locked or not terminology_enabled,
    )
    tts_enabled = st.checkbox("Bật phát giọng dịch (TTS)", value=False, disabled=controls_locked)
    tts_backend = st.selectbox(
        "TTS backend",
        ("sherpa_onnx", "fake"),
        disabled=controls_locked or not tts_enabled,
        help="sherpa-onnx dùng model VITS/Piper offline; fake chỉ tạo tone để test pipeline.",
    )
    device = st.selectbox("Device", ("cpu", "cuda"), disabled=controls_locked)
    compute_type = "int8" if device == "cpu" else "float16"
    offline = st.checkbox("Chỉ dùng model cache (offline)", value=False, disabled=controls_locked)
    semantic_endpoint = st.checkbox(
        "Tự chốt utterance theo dấu câu",
        value=True,
        disabled=controls_locked,
        help="Đóng utterance khi stable/committed có đủ số câu hoàn chỉnh.",
    )
    semantic_sentences = st.number_input(
        "Số câu trước khi tự chốt",
        min_value=1,
        max_value=10,
        value=PROFILE_DEFAULTS.vad.semantic_endpoint_sentences,
        step=1,
        disabled=controls_locked or not semantic_endpoint,
    )

    if runtime["pipeline"] is None:
        if st.button(
            "Khởi tạo pipeline",
            type="primary",
            use_container_width=True,
            disabled=asr_capability_error is not None,
        ):
            config = load_config(REALTIME_PROFILE)
            config.asr.backend = asr_backend
            config.asr.model = asr_model
            config.asr.device = device
            config.asr.compute_type = compute_type
            if asr_backend == "sherpa_onnx":
                config.asr.sherpa.provider = device
            config.asr.language = source
            config.asr.offline = offline
            config.vad.semantic_endpoint_enabled = semantic_endpoint
            config.vad.semantic_endpoint_sentences = int(semantic_sentences)
            config.translation.backend = mt_backend
            config.translation.model = (
                "facebook/m2m100_418M" if mt_backend == "m2m100" else "opus-auto"
            )
            config.translation.source_language = source
            config.translation.target_language = target
            config.translation.device = device
            config.translation.compute_type = compute_type
            config.translation.offline = offline
            config.terminology.enabled = terminology_enabled
            config.terminology.bundle_path = (
                terminology_bundle if terminology_enabled else None
            )
            config.terminology.domain = (
                terminology_domain if terminology_enabled else None
            )
            config.tts.enabled = tts_enabled
            config.tts.backend = tts_backend
            config.tts.device = device
            config.tts.offline = offline
            with st.spinner("Đang nạp model. Lần đầu có thể cần tải model..."):
                try:
                    pipeline = RealtimePipeline(config)
                    pipeline.start()
                except Exception as exc:
                    st.error(f"Không thể khởi tạo pipeline: {exc}")
                else:
                    runtime["pipeline"] = pipeline
                    runtime["normalizer"] = AudioFrameNormalizer(config.audio.sample_rate)
                    runtime["view"] = _empty_view()
                    st.success("Pipeline đã sẵn sàng")
    else:
        st.success("Pipeline đã sẵn sàng")
        if st.button("Dừng và giải phóng model", use_container_width=True):
            runtime["pipeline"].close()
            runtime["pipeline"] = None
            runtime["mode"] = None

pipeline: RealtimePipeline | None = runtime["pipeline"]


def audio_frame_callback(frame):
    if pipeline is not None and runtime["mode"] in (None, "mic"):
        runtime["mode"] = "mic"
        for chunk in runtime["normalizer"].process(frame):
            pipeline.push_audio(chunk)
    return frame


def audio_ended_callback() -> None:
    if pipeline is not None and runtime["mode"] == "mic":
        pipeline.finish()
        runtime["mode"] = None


st.subheader("Live microphone")
if pipeline is None:
    st.info("Khởi tạo pipeline ở thanh bên trước khi bật microphone.")

webrtc_ctx = webrtc_streamer(
    key="onevoice-microphone",
    mode=WebRtcMode.SENDONLY,
    audio_frame_callback=audio_frame_callback if pipeline is not None else None,
    on_audio_ended=audio_ended_callback if pipeline is not None else None,
    media_stream_constraints={"video": False, "audio": True},
    async_processing=True,
    sendback_audio=False,
    desired_playing_state=False if pipeline is None else None,
)

playing = bool(webrtc_ctx.state.playing)
if runtime["was_playing"] and not playing and pipeline is not None and runtime["mode"] == "mic":
    pipeline.finish()
    runtime["mode"] = None
runtime["was_playing"] = playing

st.subheader("Audio file fallback")
uploaded = st.file_uploader("WAV, MP3, M4A, FLAC hoặc định dạng PyAV hỗ trợ", type=None)
realtime_file = st.checkbox("Phát file theo tốc độ realtime 1×", value=True)


def feed_file(data: bytes, realtime: bool) -> None:
    assert pipeline is not None
    runtime["mode"] = "file"
    view = runtime["view"]
    view["input_started_at"] = time.time()
    view["input_started_monotonic"] = time.monotonic()
    view["input_duration_seconds"] = 0.0
    pacer = RealtimePacer(started_at=view["input_started_monotonic"])
    try:
        for chunk in iter_audio_file(
            io.BytesIO(data), pipeline.config.audio.sample_rate, pipeline.config.audio.frame_ms
        ):
            if not pipeline.is_running:
                break
            view["input_duration_seconds"] += chunk.duration_seconds
            pipeline.push_audio(chunk)
            if realtime:
                delay = pacer.delay_after(chunk.duration_seconds)
                if delay:
                    time.sleep(delay)
        view["input_finished_monotonic"] = time.monotonic()
        view["input_finished_at"] = time.time()
        pipeline.finish()
        if not pipeline.wait_until_idle(timeout=300):
            runtime["view"]["warnings"].append(
                "Pipeline chưa xử lý xong file sau 300 giây"
            )
        runtime["view"]["file_complete"] = True
    finally:
        runtime["mode"] = None


feeder = runtime.get("feeder")
file_busy = feeder is not None and feeder.is_alive()
if st.button("Chạy file", disabled=pipeline is None or uploaded is None or playing or file_busy):
    runtime["view"] = _empty_view()
    runtime["view"]["file_session"] = True
    runtime["view"]["file_realtime"] = realtime_file
    runtime["view"]["file_name"] = Path(uploaded.name).stem or "onevoice"
    thread = threading.Thread(target=feed_file, args=(uploaded.getvalue(), realtime_file), daemon=True)
    runtime["feeder"] = thread
    thread.start()
    feeder = thread
    file_busy = True

if file_busy:
    st.info("Đang xử lý file audio...")

if pipeline is not None and st.button("Flush câu hiện tại", disabled=file_busy):
    pipeline.finish()


def apply_events() -> None:
    if pipeline is None:
        return
    view = runtime["view"]
    for event in pipeline.poll_events(200):
        if event.type == EventType.STATUS:
            view["status"] = event.message
        elif event.type == EventType.SPEECH_START:
            view["status"] = "Đang nghe giọng nói"
            view["draft"] = ""
            # ASR/MT finalization for the preceding utterance can arrive after
            # audio for the next one starts. Keep partial results visible until
            # their corresponding final events move them into session history.
        elif event.type == EventType.SPEECH_END:
            view["status"] = "Đang hoàn tất câu"
        elif event.type in (EventType.ASR_PARTIAL, EventType.ASR_FINAL):
            view["draft"] = event.payload.text
            view["language"] = event.payload.language or "-"
            view["asr_ms"] = event.payload.latency_ms
            if (
                event.type == EventType.ASR_FINAL
                and view["file_session"]
                and view["input_started_monotonic"] is not None
                and event.payload.completed_at >= view["input_started_monotonic"]
            ):
                previous = view["asr_final_monotonic"]
                view["asr_final_monotonic"] = max(
                    event.payload.completed_at,
                    previous if previous is not None else event.payload.completed_at,
                )
        elif event.type == EventType.ASR_COMMITTED:
            if event.payload.is_final:
                if event.payload.text.strip():
                    view["committed_history"].append(event.payload.text)
                view["draft"] = ""
                view["stable_current"] = ""
            else:
                view["stable_current"] = event.payload.text
        elif event.type in (EventType.TRANSLATION_PARTIAL, EventType.TRANSLATION_FINAL):
            if event.payload.is_final:
                if event.payload.text.strip():
                    view["translation_history"].append(event.payload.text)
                view["translation_current"] = ""
            else:
                view["translation_current"] = event.payload.text
            view["mt_ms"] = event.payload.latency_ms
            if (
                event.type == EventType.TRANSLATION_FINAL
                and view["file_session"]
                and view["input_started_monotonic"] is not None
                and event.payload.completed_at >= view["input_started_monotonic"]
            ):
                previous = view["mt_final_monotonic"]
                view["mt_final_monotonic"] = max(
                    event.payload.completed_at,
                    previous if previous is not None else event.payload.completed_at,
                )
            view["status"] = "Hoàn tất" if event.payload.is_final else "Đang dịch"
        elif event.type in (EventType.TTS_PARTIAL, EventType.TTS_FINAL):
            speech = event.payload
            if speech.phrase_id in view["tts_seen_phrase_ids"]:
                continue
            view["tts_seen_phrase_ids"].add(speech.phrase_id)
            if view["tts_sample_rate"] not in (None, speech.sample_rate):
                view["tts_chunks"] = []
                view["tts_chunk_texts"] = []
            view["tts_sample_rate"] = speech.sample_rate
            view["tts_chunks"].append(speech.samples)
            view["tts_chunk_texts"].append(speech.text)
            view["tts_session_chunks"].append(speech.samples)
            view["tts_session_audio"] = None
            view["tts_session_wav"] = None
            started_monotonic = view["input_started_monotonic"]
            started_at = view["input_started_at"]
            if (
                view["file_session"]
                and started_monotonic is not None
                and started_at is not None
                and speech.completed_at >= started_monotonic
            ):
                elapsed = speech.completed_at - started_monotonic
                view["tts_finished_at"] = started_at + elapsed
                view["overall_latency_seconds"] = elapsed
                previous = view["tts_final_monotonic"]
                view["tts_final_monotonic"] = max(
                    speech.completed_at,
                    previous if previous is not None else speech.completed_at,
                )
            pipeline.acknowledge_tts(speech.phrase_id)
            view["tts_text"] = event.payload.text
            view["tts_ms"] = event.payload.latency_ms
            view["tts_rtf"] = event.payload.real_time_factor
            # Join only internal hard-limit chunks of one sentence. A completed
            # sentence enters autoplay immediately; it no longer waits for the
            # final event of the whole utterance.
            if ends_phrase(speech.text, speech.language) or speech.is_final:
                sentence_audio = np.concatenate(view["tts_chunks"]).astype(np.float32)
                view["tts_audio"] = sentence_audio
                view["tts_pending_audio"].append(
                    replace(
                        speech,
                        samples=sentence_audio,
                        text=" ".join(view["tts_chunk_texts"]),
                    )
                )
                view["tts_chunks"] = []
                view["tts_chunk_texts"] = []
                view["status"] = "Hoàn tất"
            else:
                view["status"] = "Đang tổng hợp giọng dịch"
        elif event.type == EventType.OVERLOAD:
            view["warnings"].append(event.message)
            view["warnings"] = view["warnings"][-5:]
        elif event.type == EventType.ERROR:
            if event.message.startswith(("Translation error:", "TTS error:")):
                view["tts_chunks"] = []
                view["tts_chunk_texts"] = []
            view["errors"].append(event.message)
            view["errors"] = view["errors"][-5:]


@st.fragment(run_every=0.25)
def realtime_results() -> None:
    apply_events()
    view = runtime["view"]

    playing_speech = view["tts_playing"]
    if playing_speech is not None:
        elapsed = time.monotonic() - view["tts_playing_started_at"]
        if elapsed >= playing_speech.duration_seconds + 0.15:
            view["tts_playing"] = None
            playing_speech = None
    if playing_speech is None and view["tts_pending_audio"]:
        playing_speech = view["tts_pending_audio"].pop(0)
        view["tts_playing"] = playing_speech
        view["tts_playing_started_at"] = time.monotonic()
        if view["file_session"] and view["output_started_at"] is None:
            view["output_started_at"] = time.time()

    st.subheader("Kết quả realtime")
    st.write(f"Trạng thái: **{view['status']}** · Ngôn ngữ: **{view['language']}**")
    st.caption("Đang xử lý · utterance hiện tại")
    left, middle, right = st.columns(3)
    with left:
        st.caption("ASR draft")
        st.text_area("ASR draft", view["draft"], height=150, label_visibility="collapsed", disabled=True)
    with middle:
        st.caption("ASR stable / committed hiện tại")
        st.text_area(
            "ASR stable hiện tại",
            view["stable_current"],
            height=150,
            label_visibility="collapsed",
            disabled=True,
        )
    with right:
        st.caption("Translation hiện tại")
        st.text_area(
            "Translation hiện tại",
            view["translation_current"],
            height=150,
            label_visibility="collapsed",
            disabled=True,
        )
    metric_a, metric_b, metric_c, metric_d = st.columns(4)
    metric_a.metric("ASR latency", f"{view['asr_ms']:.0f} ms")
    metric_b.metric("MT latency", f"{view['mt_ms']:.0f} ms")
    metric_c.metric("TTS latency", f"{view['tts_ms']:.0f} ms")
    metric_d.metric("TTS RTF", f"{view['tts_rtf']:.2f}")

    if playing_speech is not None:
        st.caption(f"TTS đang phát · {playing_speech.text}")
        st.audio(
            playing_speech.samples,
            sample_rate=playing_speech.sample_rate,
            autoplay=True,
        )
        if view["tts_pending_audio"]:
            st.caption(f"Còn {len(view['tts_pending_audio'])} phrase trong hàng đợi phát")
    elif view["file_complete"] and view["tts_session_chunks"]:
        if view["tts_session_audio"] is None:
            view["tts_session_audio"] = np.concatenate(view["tts_session_chunks"]).astype(
                np.float32
            )
        st.caption("TTS · toàn bộ file")
        st.audio(
            view["tts_session_audio"],
            sample_rate=view["tts_sample_rate"],
            autoplay=False,
        )
        if view["tts_session_wav"] is None:
            view["tts_session_wav"] = encode_wav(
                view["tts_session_audio"], view["tts_sample_rate"]
            )
        st.download_button(
            "Tải toàn bộ TTS (.wav)",
            data=view["tts_session_wav"],
            file_name=f"{view['file_name']}_tts.wav",
            mime="audio/wav",
            use_container_width=True,
        )

        input_duration = view["input_duration_seconds"]
        output_duration = len(view["tts_session_audio"]) / view["tts_sample_rate"]
        overall = view["overall_latency_seconds"]
        overhead = None if overall is None else overall - input_duration
        input_metric, output_metric, overall_metric = st.columns(3)
        input_metric.metric("Input duration", _format_duration(input_duration))
        output_metric.metric("Output duration", _format_duration(output_duration))
        overall_metric.metric(
            "End-to-end elapsed",
            _format_duration(overall, milliseconds=True),
            delta=None if overhead is None else f"{overhead:+.2f}s beyond media duration",
            delta_color="off",
        )
        start_metric, input_end_metric, output_start_metric, end_metric = st.columns(4)
        start_metric.metric("Input started at", _format_clock(view["input_started_at"]))
        input_end_metric.metric("Input finished at", _format_clock(view["input_finished_at"]))
        output_start_metric.metric(
            "Output playback started at", _format_clock(view["output_started_at"])
        )
        end_metric.metric("TTS finished at", _format_clock(view["tts_finished_at"]))
        input_end = view["input_finished_monotonic"]
        asr_end = view["asr_final_monotonic"]
        mt_end = view["mt_final_monotonic"]
        tts_end = view["tts_final_monotonic"]
        asr_tail = _elapsed_between(asr_end, input_end)
        mt_tail = _elapsed_between(mt_end, asr_end)
        tts_tail = _elapsed_between(tts_end, mt_end)
        completed_marks = [
            mark for mark in (asr_end, mt_end, tts_end) if mark is not None
        ]
        pipeline_end = max(completed_marks) if completed_marks else None
        post_input = _elapsed_between(pipeline_end, input_end)
        feed_elapsed = _elapsed_between(
            view["input_finished_monotonic"], view["input_started_monotonic"]
        )
        feed_drift = (
            None
            if feed_elapsed is None or not view["file_realtime"]
            else max(0.0, feed_elapsed - input_duration)
        )

        st.markdown("**Post-input component latency**")
        feed_metric, asr_tail_metric, mt_tail_metric, tts_tail_metric, total_tail_metric = st.columns(5)
        feed_metric.metric("Realtime feed drift", _format_latency(feed_drift))
        asr_tail_metric.metric("ASR · input end → final", _format_latency(asr_tail))
        mt_tail_metric.metric("MT · ASR final → final", _format_latency(mt_tail))
        tts_tail_metric.metric("TTS · MT final → ready", _format_latency(tts_tail))
        total_tail_metric.metric("Total after input", _format_latency(post_input))
        st.caption(
            "Overall latency đo từ lúc bắt đầu feed audio đến khi chunk TTS cuối synthesize xong; "
            "các stage phía dưới dùng timestamp final liên tiếp và clamp phần overlap về 0 ms. "
            "Không tính thời gian người dùng phát lại player toàn file."
        )

    st.divider()
    st.subheader("Lịch sử hoàn tất")
    history_left, history_right = st.columns(2)
    with history_left:
        st.caption("Committed transcript")
        st.text_area(
            "Committed history",
            _join_history(view["committed_history"]),
            height=240,
            label_visibility="collapsed",
            disabled=True,
        )
    with history_right:
        st.caption("Translated transcript")
        st.text_area(
            "Translation history",
            _join_history(view["translation_history"]),
            height=240,
            label_visibility="collapsed",
            disabled=True,
        )
    for warning in view["warnings"]:
        st.warning(warning)
    for error in view["errors"]:
        st.error(error)


realtime_results()
