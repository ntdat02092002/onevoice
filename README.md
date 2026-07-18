# OneVoice Realtime Translation

Pipeline dịch giọng nói offline, dạng module, hỗ trợ Việt (`vi`), Anh (`en`), Trung (`zh`) và Hàn (`ko`). V1 thực hiện:

```text
WebRTC microphone / audio file
  -> PCM 16 kHz mono
  -> WebRTC VAD
  -> Moonshine native streaming ASR (Dolphin/Faster-Whisper optional)
  -> Local Agreement stable prefix
  -> Wait-k policy
  -> OPUS-MT + CTranslate2 INT8 translation (M2M100 optional)
  -> CLI / Streamlit events
```

## Cài đặt

Python 3.11 trên Windows:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e ".[moonshine,opus,app,dev]"
```

Project khóa `transformers>=4.46,<5`. Nhánh Transformers 5.x có thể lazy-import các image processor như ZoeDepth và đòi `torchvision` dù pipeline chỉ xử lý text; không cài `torchvision` để chữa lỗi này.

Chạy test không tải model:

```powershell
pytest
```

Chạy Streamlit:

```powershell
streamlit run app/streamlit_app.py
```

Lần khởi tạo đầu tiên sẽ tải model Moonshine và một hoặc hai model OPUS-MT đúng với hướng dịch đã chọn. OPUS-MT được convert INT8 một lần vào `.cache/onevoice/opus_ct2`; các lần sau dùng lại trực tiếp. Không bật **Chỉ dùng model cache (offline)** trước khi asset đã được cache và convert.

Cấu hình mặc định dùng Moonshine CPU/ONNX Runtime vì model cache state khi audio được thêm dần, không decode lại toàn bộ câu. Nếu chọn CUDA, runtime phải có CUDA Execution Provider; nếu không adapter sẽ báo lỗi rõ. Faster-Whisper vẫn được giữ làm backend fallback.

Chỉ cài backend cần dùng:

```powershell
# Mặc định, phù hợp realtime và đủ vi/en/zh/ko khi chọn source cụ thể
python -m pip install -e ".[moonshine,opus,app]"

# Dolphin Base/Small theo report, hỗ trợ vi/zh/ko (không có en)
python -m pip install -e ".[dolphin,app]"
```

Moonshine là model riêng theo từng ngôn ngữ nên không hỗ trợ `source=auto` trong adapter này. Chọn rõ `vi`, `en`, `zh` hoặc `ko`. Dolphin có LID/auto trong phạm vi ngôn ngữ mà model hỗ trợ, nhưng bản chính thức không có English.

## CLI smoke test

```powershell
onevoice sample.wav --source vi --target en --asr-backend moonshine --realtime
```

Chạy toàn pipeline bằng fake model, không tải weights:

```powershell
onevoice sample.wav --source en --target vi --asr-backend fake --mt-backend fake
```

CLI xuất JSON Lines cho ASR partial/final, stable transcript, translation và latency.

## Cấu hình và thay model

Cấu hình mặc định nằm ở `config/default.yaml`. Core chỉ phụ thuộc các protocol trong `onevoice.protocols`; backend thật nằm trong `onevoice.backends` và được tạo qua registry.

Các backend tích hợp sẵn validate capability trước khi import runtime hoặc tải weights. Ví dụ Moonshine `zh` chỉ nhận `auto/base`, Moonshine không nhận source `auto`, và Dolphin không nhận `en`. Streamlit lọc model theo ngôn ngữ và hiện lỗi ngay; CLI/config sai sẽ raise `ValueError` khi tạo `RealtimePipeline`.

Hướng dẫn chi tiết để thêm ASR/MT model, VAD, audio preprocessing, stable-prefix policy hoặc component hoàn toàn mới: [docs/ADDING_MODELS_AND_COMPONENTS.md](docs/ADDING_MODELS_AND_COMPONENTS.md).

Danh sách backend/model hiện có, ma trận ngôn ngữ, streaming, dependency, license và config mẫu: [docs/MODEL_ZOO.md](docs/MODEL_ZOO.md).

Để thêm Gipformer, Zipformer, model MT khác hoặc QNN:

1. Viết adapter triển khai protocol tương ứng và lifecycle `load/reset/close`.
2. Đăng ký adapter bằng `registry.register(kind, name, factory)` trong `onevoice.backends`.
3. Đổi `backend` trong YAML hoặc UI. Không sửa `RealtimePipeline`.

Các contract chính:

- `StreamingAsrBackend.transcribe(SpeechSegment, language) -> AsrUpdate`
- `CommitPolicy.update(AsrUpdate) -> CommittedTranscript | None`
- `TranslationBackend.translate(TranslationRequest) -> TranslationUpdate`
- `VadBackend.process(AudioChunk) -> list[SpeechSegment]`
- `VadBackend.request_endpoint() -> None` (tín hiệu thread-safe; audio worker mới thực sự đóng utterance)

Queue audio/ASR/MT đều có giới hạn. Khi audio overload, pipeline phát event, bỏ utterance bị đứt và reset generation để kết quả cũ không lọt xuống MT. Translation có revision guard để bỏ kết quả đã stale.

Mặc định pipeline tự đóng utterance khi stable/committed có đủ 2 câu hoàn chỉnh. Chỉnh `vad.semantic_endpoint_sentences`, hoặc đặt `vad.semantic_endpoint_enabled: false` để chỉ dùng khoảng lặng VAD và `max_utterance_seconds`. Đây là heuristic theo dấu `. ! ? 。 ！ ？`; điểm cắt audio xảy ra ngay sau khi ASR xác nhận ngưỡng nên có thể dư một đoạn ngắn do độ trễ inference.

## Streamlit microphone

Live microphone dùng `streamlit-webrtc`; inference không chạy trong audio callback. Callback chỉ resample và enqueue frame, còn `st.fragment` poll event khoảng 250 ms và chỉ rerun vùng kết quả. Streamlit không cho callback WebRTC ở thread riêng sửa trực tiếp widget; full-app rerun chỉ còn xảy ra khi người dùng đổi control, bấm nút hoặc trạng thái WebRTC thay đổi.

- `localhost` được browser xem là secure context và dùng mic trực tiếp.
- Deploy remote cần HTTPS.
- Môi trường NAT/firewall có thể cần cấu hình STUN/TURN trong `webrtc_streamer`.
- Không chạy mic và file đồng thời. Dừng WebRTC hoặc bấm **Flush câu hiện tại** để commit câu cuối.

## Tiêu chí kiểm tra thủ công

| Ngôn ngữ | Câu mẫu |
|---|---|
| Việt | Dừng máy ngay, có người đang ở gần băng chuyền. |
| Anh | Please wear your safety helmet before entering the warehouse. |
| Trung | 请检查传送带，叉车马上进入仓库。 |
| Hàn | 지게차가 들어오니까 통로를 비워 주세요. |

Theo dõi ASR latency, MT latency và thời gian từ lúc nói đến output đầu tiên. Mục tiêu dưới 1.5 giây/output đầu và 2–3 giây end-to-end là mục tiêu benchmark, không phải bảo đảm trên mọi CPU.

## Model và license

- [Moonshine Voice](https://github.com/moonshine-ai/moonshine): code và model English dùng MIT; model Việt/Trung/Hàn dùng Moonshine Community License, **chỉ phi thương mại**. Backend dùng native incremental stream và chỉ nạp phần waveform mới.
- [Dolphin](https://github.com/DataoceanAI/Dolphin): Apache-2.0 cho code và weights; adapter hiện hỗ trợ `base`/`small` cho Việt, Trung, Hàn. Repo chính thức không liệt kê English.
- [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper): MIT, CTranslate2 implementation của Whisper.
- [OPUS-MT](https://github.com/Helsinki-NLP/Opus-MT) chạy bằng [CTranslate2](https://opennmt.net/CTranslate2/): backend mặc định. Các model Việt↔Anh và phần lớn model theo cặp dùng Apache-2.0; `zh-en` và model English→Korean dùng CC-BY-4.0. Kiểm tra model card tương ứng trước khi phân phối sản phẩm.
- [M2M100-418M](https://huggingface.co/facebook/m2m100_418M): MIT, hỗ trợ trực tiếp cả 12 hướng giữa bốn ngôn ngữ.
- `streamlit-webrtc`: MIT.

V1 chưa gồm TTS, model khử nhiễu, glossary, mobile packaging hoặc Qualcomm QNN. `AudioPreprocessor` và backend registry là điểm cắm cho các phase sau.
