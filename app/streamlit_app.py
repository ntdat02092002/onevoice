from __future__ import annotations

import io
import logging
import threading
import time
from typing import Any

import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from onevoice.audio import AudioFrameNormalizer, iter_audio_file
from onevoice.backends.asr import asr_model_options, validate_asr_selection
from onevoice.config import load_config
from onevoice.models import EventType
from onevoice.pipeline import RealtimePipeline


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
        "errors": [],
    }


def _join_history(history: list[str]) -> str:
    return "\n\n".join(item.strip() for item in history if item.strip())


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
st.caption("Offline ASR → Stable Prefix → Machine Translation")

with st.sidebar:
    st.header("Pipeline")
    controls_locked = runtime["pipeline"] is not None
    source_label = st.selectbox("Ngôn ngữ nguồn", LANGUAGES, index=1, disabled=controls_locked)
    source = LANGUAGES[source_label]
    target_options = {name: code for name, code in LANGUAGES.items() if code != "auto" and code != source}
    target_label = st.selectbox("Ngôn ngữ đích", target_options, disabled=controls_locked)
    target = target_options[target_label]
    asr_backend = st.selectbox(
        "ASR backend",
        ("moonshine", "dolphin", "faster_whisper", "fake"),
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
        value=2,
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
            config = load_config()
            config.asr.backend = asr_backend
            config.asr.model = asr_model
            config.asr.device = device
            config.asr.compute_type = compute_type
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
    try:
        for chunk in iter_audio_file(
            io.BytesIO(data), pipeline.config.audio.sample_rate, pipeline.config.audio.frame_ms
        ):
            if not pipeline.is_running:
                break
            pipeline.push_audio(chunk)
            if realtime:
                time.sleep(chunk.duration_seconds)
        pipeline.finish()
    finally:
        runtime["mode"] = None


feeder = runtime.get("feeder")
file_busy = feeder is not None and feeder.is_alive()
if st.button("Chạy file", disabled=pipeline is None or uploaded is None or playing or file_busy):
    runtime["view"] = _empty_view()
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
            view["status"] = "Hoàn tất" if event.payload.is_final else "Đang dịch"
        elif event.type in (EventType.ERROR, EventType.OVERLOAD):
            view["errors"].append(event.message)
            view["errors"] = view["errors"][-5:]


@st.fragment(run_every=0.25)
def realtime_results() -> None:
    apply_events()
    view = runtime["view"]
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
    metric_a, metric_b = st.columns(2)
    metric_a.metric("ASR latency", f"{view['asr_ms']:.0f} ms")
    metric_b.metric("MT latency", f"{view['mt_ms']:.0f} ms")

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
    for error in view["errors"]:
        st.error(error)


realtime_results()
