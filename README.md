# OneVoice Realtime Translation

Pipeline dịch giọng nói offline, dạng module, hỗ trợ Việt (`vi`), Anh (`en`), Trung (`zh`) và Hàn (`ko`). V1 thực hiện:

```text
WebRTC microphone / audio file
  -> PCM 16 kHz mono
  -> WebRTC VAD
  -> Moonshine / Sherpa streaming ASR (Dolphin/Faster-Whisper optional)
  -> ASR terminology guidance + exact-alias canonicalization (optional)
  -> Term-aware Local Agreement: locked sentences + mutable fragment
  -> Wait-k policy
  -> Terminology-safe OPUS-MT or M2M100 + CTranslate2 INT8
  -> Term-aware sentence phrase chunker (8–24 token)
  -> Display text / TTS spoken-form normalization
  -> sherpa-onnx VITS/Piper TTS (optional)
  -> CLI / Streamlit audio events
```

## Cài đặt

Python 3.11 trên Windows:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e ".[moonshine,opus,app,dev]"

# Thêm TTS offline
python -m pip install -e ".[moonshine,opus,tts,app,dev]"

# Stack Sherpa streaming Zipformer + OPUS + UI
python -m pip install -e ".[sherpa-asr,opus,app,dev]"
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

Lần khởi tạo đầu tiên sẽ tải model Moonshine và model dịch đã chọn. OPUS-MT được convert một lần vào `.cache/onevoice/opus_ct2`; M2M100 được convert vào `.cache/onevoice/m2m100_ct2`. Các lần sau CTranslate2 dùng lại cache trực tiếp. Không bật **Chỉ dùng model cache (offline)** trước khi asset đã được tải và convert.

Cấu hình mặc định dùng Moonshine CPU/ONNX Runtime vì model cache state khi audio được thêm dần, không decode lại toàn bộ câu. Nếu chọn CUDA, runtime phải có CUDA Execution Provider; nếu không adapter sẽ báo lỗi rõ. Faster-Whisper vẫn được giữ làm backend fallback.

Chỉ cài backend cần dùng:

```powershell
# Mặc định, phù hợp realtime và đủ vi/en/zh/ko khi chọn source cụ thể
python -m pip install -e ".[moonshine,opus,app]"

# Dolphin Base/Small theo report, hỗ trợ vi/zh/ko (không có en)
python -m pip install -e ".[dolphin,app]"
```

Để thử streaming Zipformer qua sherpa-onnx:

```powershell
python -m pip install -e ".[sherpa-asr,opus,app]"
onevoice sample.wav --source vi --target en --asr-backend sherpa_onnx

# Bật sample terminology profile
onevoice sample.wav --source en --target vi --asr-backend sherpa_onnx `
  --terminology-bundle assets/terminology/factory-sample-v1/terminology.yaml `
  --terminology-domain test
```

Model được chọn tự động theo source `vi/en/zh/ko` và tải lần đầu vào
`.cache/onevoice/asr`. Backend dùng `OnlineRecognizer`, giữ native stream state và
chỉ feed phần waveform mới. Semantic endpoint có thể cắt final tại một sentence
boundary cũ hơn snapshot mới nhất; backend sẽ decode lại final prefix trên stream
mới rồi reset theo utterance, không giữ cursor lỗi sang câu sau. Model English
dùng thêm punctuator online INT8 chính thức của sherpa-onnx để phục hồi dấu câu
và casing; đặt `asr.sherpa.punctuation_enabled: false` nếu không muốn tải model
phụ này. Model Vietnamese có punctuation nhưng vocabulary ALL CAPS, nên adapter
chuẩn hóa về sentence case. Model Vietnamese dùng license CC-BY-NC-ND-4.0; cần
review riêng trước khi dùng thương mại hoặc phân phối lại.

Moonshine là model riêng theo từng ngôn ngữ nên không hỗ trợ `source=auto` trong adapter này. Chọn rõ `vi`, `en`, `zh` hoặc `ko`. Dolphin có LID/auto trong phạm vi ngôn ngữ mà model hỗ trợ, nhưng bản chính thức không có English.

## CLI smoke test

```powershell
onevoice sample.wav --source vi --target en --asr-backend moonshine --realtime
```

Chạy toàn pipeline bằng fake model, không tải weights:

```powershell
onevoice sample.wav --source en --target vi --asr-backend fake --mt-backend fake

# Smoke test cả TTS queue không tải model (phát tone fake)
onevoice sample.wav --source en --target vi --asr-backend fake --mt-backend fake --tts --tts-backend fake
```

CLI xuất JSON Lines cho ASR partial/final, stable transcript, translation và latency.

## Terminology dictionary

Sample bundle nằm tại
[`assets/terminology/factory-sample-v1/terminology.yaml`](assets/terminology/factory-sample-v1/terminology.yaml).
Trong Streamlit:

1. Chọn source/target và backend trước.
2. Bật **Terminology dictionary**.
3. Chọn domain, ví dụ `test`, `factory-safety` hoặc `factory-maintenance`.
4. Chờ `Terminology preflight` báo hợp lệ rồi khởi tạo pipeline.

Preflight validate schema/language coverage và compile trước ASR terms, MT
bindings, TTS spoken forms. UI hiển thị bundle ID, schema, relative path,
checksum, route và artifact counts. Sau khi pipeline start, toàn bộ lựa chọn bị
khóa; muốn đổi bundle/domain/backend phải **Stop pipeline**, chỉnh lại rồi Start.
Không có hot-swap giữa một pipeline session.

Ba cơ chế ASR terminology đang dùng:

| Backend | Guidance trước/trong decode | Lớp an toàn sau ASR |
|---|---|---|
| Sherpa streaming Zipformer | Native hotwords trên `modified_beam_search` | Exact alias → canonical |
| Faster-Whisper | Bounded `initial_prompt` | Exact alias → canonical |
| Moonshine, Dolphin | Không truyền option native chưa được xác nhận | Exact alias → canonical |

Native hotword và prompt là bias, không phải hard guarantee. Post-correction chỉ
sửa canonical/alias được khai báo trong active profile; không bật fuzzy
edit-distance mặc định.

Cấu hình YAML tối thiểu:

```yaml
terminology:
  enabled: true
  bundle_path: assets/terminology/factory-sample-v1/terminology.yaml
  domain: test
```

Với Sherpa + terminology, pipeline tự đổi `greedy_search` sang
`modified_beam_search`. Profile được giới hạn bởi `max_hotword_terms` và
`max_hotword_tokens`; term ngoài tokenizer vocabulary bị loại khỏi native
hotwords nhưng exact-alias correction vẫn còn hiệu lực. Chi tiết schema và cách
thêm term: [sample bundle README](assets/terminology/factory-sample-v1/README.md).
Thiết kế và trạng thái từng phase:
[terminology implementation roadmap](docs/plans/terminology/README.md).

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
- `TtsBackend.synthesize(TtsRequest) -> TtsUpdate`
- `VadBackend.process(AudioChunk) -> list[SpeechSegment]`
- `VadBackend.request_endpoint(started_at=..., cut_sample=...) -> None` (tín hiệu thread-safe; audio worker cắt đúng snapshot ASR đã xác nhận)

Queue audio/ASR/MT/TTS đều có giới hạn. Khi audio overload, pipeline phát event, bỏ utterance bị đứt và reset generation để kết quả cũ không lọt xuống downstream. Partial ASR và MT được coalesce latest-only theo utterance; ASR final, MT final và TTS sinh từ final đi qua lane lossless.

Mặc định pipeline tự đóng utterance khi stable/committed có đủ 2 câu hoàn chỉnh.
Các backend có word timestamps, gồm streaming Zipformer, ánh xạ đúng boundary
câu thứ N sang sample cursor. Khi endpoint đã pending, partial đến muộn bị freeze;
câu sau được giữ trong suffix làm đầu utterance kế tiếp thay vì lọt vào final.
Backend không có timestamp vẫn dùng fallback an toàn và chỉ endpoint khi stable
text cùng hypothesis mới nhất đều kết thúc tại boundary. Chỉnh
`vad.semantic_endpoint_sentences`, hoặc đặt
`vad.semantic_endpoint_enabled: false` để chỉ dùng khoảng lặng VAD và
`max_utterance_seconds`. Với Zipformer, `semantic_endpoint_context_ms: null`
tự resolve thành 200 ms acoustic pre-roll cho suffix; phần pre-roll chỉ cấp
context cho recognizer và bị loại khỏi transcript để không lặp cuối câu trước.
Backend khác auto resolve thành 0.

TTS phrase chỉ được commit sau khi synthesis thành công và consumer chấp nhận audio vào playback queue. Validity dựa trên generation và exact translated-prefix reservation: revision mới vẫn giữ phrase nếu content prefix không đổi, nhưng cancel reservation chưa synthesize khi content phân kỳ. Request bị queue drop, model lỗi hoặc event không giao được không bị coi nhầm là đã phát. Completed stream có idempotency guard để duplicate final không phát lại audio.

## Streamlit microphone

Live microphone dùng `streamlit-webrtc`; inference không chạy trong audio callback. Callback chỉ resample và enqueue frame, còn `st.fragment` poll event khoảng 250 ms và chỉ rerun vùng kết quả. Streamlit không cho callback WebRTC ở thread riêng sửa trực tiếp widget; full-app rerun chỉ còn xảy ra khi người dùng đổi control, bấm nút hoặc trạng thái WebRTC thay đổi.

Streamlit khởi tạo từ `config/realtime_conversation.yaml`: semantic endpoint 1 câu, MT hybrid wait-k `6/4/1200 ms` với minimum interval `500 ms`, và TTS `stable_sentence` qua 2 translation revisions. Default an toàn trong `config/default.yaml` vẫn là TTS `final_utterance`; CLI chỉ dùng realtime profile khi được truyền `--config`.

TTS partial được autoplay theo từng target sentence hoàn chỉnh. Nếu một câu vượt hard maximum 24 token, UI chỉ ghép các internal chunk của chính câu đó rồi đưa vào playback queue; không chờ toàn utterance. Khi chạy file, feeder pace theo absolute media timeline để không tích lũy sai số `sleep` trên Windows. Sau khi hoàn tất, UI tạo player và WAV toàn file, đồng thời hiển thị input/output duration, time-to-output, synthesis timeline, thời điểm playback dự kiến kết thúc, feed drift và post-input ASR/MT/TTS tail latency.

`Playback finished at (estimated)` được tính từ thời điểm server đưa từng phrase
vào autoplay cộng đúng duration audio và chỉ được ghi nhận khi scheduler đã chạy
hết phrase. Đây không phải browser/loa `ended` telemetry vì `st.audio` không gửi
event đó về Python.

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

Theo dõi ASR/MT/TTS inference latency, thời gian đến output đầu tiên, realtime feed drift và playback tail. `End-to-end elapsed` bao gồm toàn bộ media duration; `Total after input` mới là phần pipeline còn lại sau khi input kết thúc. Mục tiêu dưới 1.5 giây/output đầu và 2–3 giây tail là mục tiêu benchmark, không phải bảo đảm trên mọi CPU hoặc mọi tốc độ voice.

## Model và license

- [Moonshine Voice](https://github.com/moonshine-ai/moonshine): code và model English dùng MIT; model Việt/Trung/Hàn dùng Moonshine Community License, **chỉ phi thương mại**. Backend dùng native incremental stream và chỉ nạp phần waveform mới.
- [Dolphin](https://github.com/DataoceanAI/Dolphin): Apache-2.0 cho code và weights; adapter hiện hỗ trợ `base`/`small` cho Việt, Trung, Hàn. Repo chính thức không liệt kê English.
- [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper): MIT, CTranslate2 implementation của Whisper.
- [Sherpa-ONNX](https://github.com/k2-fsa/sherpa-onnx): runtime Apache-2.0; license weights phụ thuộc model. Vietnamese streaming Zipformer `hynt` đang dùng CC-BY-NC-ND-4.0 và cần review trước sử dụng thương mại/phân phối lại.
- [OPUS-MT](https://github.com/Helsinki-NLP/Opus-MT) chạy bằng [CTranslate2](https://opennmt.net/CTranslate2/): backend mặc định. Các model Việt↔Anh và phần lớn model theo cặp dùng Apache-2.0; `zh-en` và model English→Korean dùng CC-BY-4.0. Kiểm tra model card tương ứng trước khi phân phối sản phẩm.
- [M2M100-418M](https://huggingface.co/facebook/m2m100_418M): MIT, hỗ trợ trực tiếp cả 12 hướng giữa bốn ngôn ngữ; backend convert Hugging Face weights một lần rồi inference bằng CTranslate2 với target-language prefix.
- `streamlit-webrtc`: MIT.

MT dùng hybrid wait-k: sentence boundary là trigger ưu tiên, nhưng partial vẫn chạy
sau mỗi cụm token ổn định hoặc timeout để subtitle không phải chờ hết câu.
Pending partial cùng utterance được coalesce latest-only; final đi qua lane
lossless. Khi bật terminology, OPUS/M2M100 split cả partial theo câu: completed
sentence được cache bằng exact source text + stream ID, còn mutable tail luôn
dịch lại. Final tái sử dụng cache rồi xóa state của utterance. Khi terminology
tắt, partial vẫn dùng một growing-prefix request như trước. TTS global default
là `final_utterance`; riêng Streamlit/realtime profile dùng `stable_sentence`,
chỉ phát target sentence hoàn chỉnh đã đồng thuận hoặc tail từ final. Hard
maximum vẫn là 24 token và UI ghép internal chunk theo câu trước khi autoplay.
VAD semantic endpoint và TTS emission là hai policy độc lập. Backend
`sherpa_onnx` tự chọn voice theo target language, tải lần đầu vào
`.cache/onevoice/tts` và tái sử dụng cache; UI không yêu cầu đường dẫn model.
`offline: true` sẽ fail sớm nếu voice chưa được cache.

Các profile mẫu nằm trong `config/realtime_conversation.yaml` (endpoint 1 câu), `config/continuous_speech.yaml` (2 câu) và `config/stable_demo.yaml` (agreement bảo thủ hơn). Truyền profile bằng `onevoice sample.wav --config config/<profile>.yaml`; profile được deep-merge lên `config/default.yaml`, nên field không khai báo vẫn giữ nguyên model/language/device/cache mặc định.
