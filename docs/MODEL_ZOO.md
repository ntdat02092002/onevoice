# OneVoice Model Zoo

Tài liệu này liệt kê các backend và option **đang được codebase đăng ký**. Đây không phải danh sách model dự kiến trong roadmap. Capability runtime được validate trước khi tải weights; tổ hợp không hỗ trợ sẽ raise `ValueError`.

Ngôn ngữ sản phẩm:

| Mã | Ngôn ngữ |
|---|---|
| `vi` | Tiếng Việt |
| `en` | English |
| `zh` | 中文 / Mandarin |
| `ko` | 한국어 |
| `auto` | Backend tự nhận diện, chỉ dùng khi backend có hỗ trợ |

## ASR

### Bảng chọn nhanh

| Backend | Language | Model hợp lệ | Auto language | Native streaming | Dependency |
|---|---|---|---|---|---|
| `moonshine` | `en` | `auto`, `tiny`, `base`, `tiny_streaming`, `small_streaming`, `medium_streaming` | Không | Có với các model `_streaming` | `moonshine` |
| `moonshine` | `vi` | `auto`, `base` | Không | Không có weights streaming riêng | `moonshine` |
| `moonshine` | `zh` | `auto`, `base` | Không | Không có weights streaming riêng | `moonshine` |
| `moonshine` | `ko` | `auto`, `tiny` | Không | Không có weights streaming riêng | `moonshine` |
| `dolphin` | `auto`, `vi`, `zh`, `ko` | `base`, `small` | Có, trong tập ngôn ngữ Dolphin hỗ trợ | Không; adapter decode growing utterance | `dolphin` |
| `faster_whisper` | `auto`, `vi`, `en`, `zh`, `ko` | `tiny`, `base`, `small` trong UI; config có thể dùng Whisper model ID hợp lệ khác | Có | Không; adapter decode lại growing utterance | `models` |
| `fake` | mọi language sản phẩm | `fake` | Giả lập | Giả lập | Không |

`Moonshine model=auto` là tự chọn **architecture cho language đã biết**, không phải tự nhận diện language. Với catalog hiện tại, nó resolve `en -> medium_streaming`, `vi -> base`, `zh -> base`, `ko -> tiny`.

### Moonshine

| Architecture | Tham số tham khảo | Language trong app | Gợi ý |
|---|---:|---|---|
| `tiny` | 26M | `en`, `ko` | Nhẹ nhất; partial inference thường thấp |
| `base` | 58M | `en`, `vi`, `zh` | Model thường, cân bằng accuracy/compute |
| `tiny_streaming` | 34M | `en` | Realtime CPU; cache encoder/decoder |
| `small_streaming` | 123M | `en` | Accuracy tốt hơn, compute cao hơn |
| `medium_streaming` | 245M | `en` | Chất lượng cao nhất trong catalog hiện tại |

- Adapter chỉ thêm phần waveform mới vào native `Transcriber`; không gửi lại phần audio đã xử lý.
- `device=cpu` dùng ONNX Runtime CPU. `device=cuda` yêu cầu CUDA Execution Provider khả dụng.
- Model English dùng MIT. Model `vi/zh/ko` dùng Moonshine Community License, chỉ phi thương mại.
- Không dùng `language=auto`; phải chọn rõ language trước khi load model.

### Dolphin

- Model: `base`, `small`.
- Language: `auto`, `vi`, `zh`, `ko`; official weights hiện không có English.
- `language=auto` dùng language identification của Dolphin.
- Runtime dựa trên PyTorch; chọn `device=cpu` hoặc `cuda`.
- Code và weights: Apache-2.0 theo upstream Dolphin.

### Faster-Whisper

- UI cung cấp `tiny`, `base`, `small`; backend truyền `config.asr.model` trực tiếp cho `WhisperModel`, nên YAML/CLI có thể dùng model ID Faster-Whisper hợp lệ khác.
- `language=auto` truyền `language=None` để Whisper detect language và trả `language_probability`.
- CPU thường dùng `compute_type=int8`; CUDA thường dùng `float16`.
- Đây là fallback đa ngôn ngữ. Adapter hiện decode lại toàn growing utterance nên thường kém hiệu quả hơn Moonshine native streaming cho mic realtime.
- License implementation Faster-Whisper: MIT; license weights phụ thuộc model được chọn.

## Machine Translation

| Backend | Model config | Hướng dịch | Route | Dependency |
|---|---|---:|---|---|
| `opus_ct2` | bắt buộc `opus-auto` | 12 hướng giữa `vi/en/zh/ko` | Trực tiếp với English; cặp còn lại pivot qua English | `opus` |
| `m2m100` | mặc định `facebook/m2m100_418M` | 12 hướng trực tiếp | Một model multilingual | `models` |
| `fake` | tên model không rỗng | Giả lập | Thêm prefix target language | Không |

Source và target phải khác nhau. `source_language=auto` chỉ có nghĩa MT nhận language do ASR phát hiện; MT không tự nghe audio để detect language.

### OPUS-MT + CTranslate2

Các pair model vật lý:

| Pair | Hugging Face model ID |
|---|---|
| `vi -> en` | `Helsinki-NLP/opus-mt-vi-en` |
| `en -> vi` | `Helsinki-NLP/opus-mt-en-vi` |
| `zh -> en` | `Helsinki-NLP/opus-mt-zh-en` |
| `en -> zh` | `Helsinki-NLP/opus-mt-en-zh` |
| `ko -> en` | `Helsinki-NLP/opus-mt-ko-en` |
| `en -> ko` | `Helsinki-NLP/opus-mt-tc-big-en-ko` |

Ví dụ `vi -> ko` chạy `vi -> en -> ko`, nên load hai model và có thể kém nhanh/chính xác hơn hướng trực tiếp. Snapshot được tải và convert một lần vào `.cache/onevoice/opus_ct2`; các lần sau tái sử dụng CT2 model.

License thay đổi theo pair/model card. Các pair Việt–Anh và phần lớn pair dùng Apache-2.0; `zh-en` và English→Korean có model dùng CC-BY-4.0. Kiểm tra model card trước khi phân phối thương mại.

### M2M100-418M

- Model mặc định: `facebook/m2m100_418M`.
- Dịch trực tiếp đủ 12 hướng, không pivot qua English.
- Nặng và thường chậm hơn OPUS INT8 trên CPU, nhưng giữ context qua một model multilingual duy nhất.
- Hỗ trợ CPU/CUDA qua PyTorch; license model: MIT.

## Text-to-Speech

| Backend | Model | Streaming trong OneVoice | Dependency |
|---|---|---|---|
| `sherpa_onnx` | Piper/VITS hoặc Supertonic 3 INT8 | Sentence-aware phrase 8–24 token, worker/queue riêng | `tts` |
| `fake` | Tone kiểm thử | Giả lập | Không |

`sherpa_onnx` là lựa chọn mặc định khi bật TTS vì phù hợp khuyến nghị offline/edge của report. Với `model: auto`, backend chọn voice theo ngôn ngữ đích, tải một lần từ release chính thức vào `.cache/onevoice/tts` và tái sử dụng cache. UI không yêu cầu đường dẫn model.

| Target | Voice auto | Loại |
|---|---|---|
| `vi` | `vits-piper-vi_VN-25hours_single-low` | Piper/VITS 16 kHz |
| `en` | `vits-piper-en_US-amy-low` | Piper/VITS 16 kHz |
| `zh` | `vits-piper-zh_CN-chaowen-medium` | Piper/VITS 22.05 kHz |
| `ko` | `sherpa-onnx-supertonic-3-tts-int8-2026-05-11` | Supertonic 3 INT8 24 kHz |

```yaml
tts:
  enabled: true
  backend: sherpa_onnx
  model: auto
  language: auto
  cache_dir: .cache/onevoice/tts
  offline: false
  device: cpu
  num_threads: 2
  speaker_id: 0
  speed: 0.9
  min_chunk_tokens: 8
  max_chunk_tokens: 24
  agreement_updates: 2
  sentence_boundary_only: true
  final_only: true
  emission_mode: final_utterance
  timeout_ms: 1200
```

`language: auto` được pipeline resolve thành target language. Bật `offline: true` chỉ sau khi voice đã có trong cache. `model_dir` và các tên asset vẫn được backend hỗ trợ cho custom voice nhưng là advanced YAML override, không xuất hiện trên UI.

Đây là global default an toàn: `final_utterance`. Streamlit load `config/realtime_conversation.yaml`, override thành `stable_sentence`, `final_only: false`, `sentence_boundary_only: true`, `agreement_updates: 2`. Target sentence phải hoàn chỉnh và đồng thuận qua hai translation revisions; final flush tail còn lại. UI autoplay từng câu, chỉ ghép các internal chunk nếu câu vượt hard maximum 24 token.

## VAD và endpoint

| Backend | Option | Mục đích |
|---|---|---|
| `webrtc` | mặc định | VAD realtime, gom frame thành utterance |
| `passthrough` | test/file | Xem mọi audio là speech; không dùng cho mic production |

`webrtc` yêu cầu PCM mono 16 kHz và frame `10`, `20` hoặc `30` ms. Các option chính:

```yaml
vad:
  backend: webrtc
  aggressiveness: 2
  min_speech_ms: 250
  end_silence_ms: 600
  speech_padding_ms: 200
  max_utterance_seconds: 15
  semantic_endpoint_enabled: true
  semantic_endpoint_sentences: 2
```

Semantic endpoint đóng utterance khi stable/committed có đủ số câu hoàn chỉnh và cả stable text lẫn ASR hypothesis mới nhất đều kết thúc tại sentence boundary. Vì vậy fragment câu tiếp theo chưa hoàn chỉnh không bị cắt. Terminal mark đã vượt Local Agreement được publish dù `hold_tokens=1`, tạo cửa sổ endpoint thực sự. Đặt `semantic_endpoint_enabled: false` để chỉ dùng silence/max-duration endpoint.

## Audio preprocessing, commit và translation policy

| Kind | Backend | Trạng thái hiện tại |
|---|---|---|
| `preprocessor` | `passthrough` | Không đổi samples; điểm cắm RNNoise/GTCRN sau này |
| `commit` | `local_agreement` | LA-2 + Hold-1; lock completed sentence, cho phép current fragment revision |
| translation policy | `WaitKTranslationPolicy` | Wait 6 token; update mỗi 4 token, boundary hoặc timeout 1200 ms; minimum interval 500 ms |

Các option mặc định:

```yaml
commit:
  backend: local_agreement
  agreement_updates: 2
  hold_tokens: 1

translation:
  wait_tokens: 6
  update_tokens: 4
  timeout_ms: 1200
  min_request_interval_ms: 500
  sentence_boundary_only: false
```

Local Agreement chỉ làm immutable các completed sentence. Fragment của current sentence vẫn mutable: nếu ASR sửa token đầu, policy chờ đủ agreement rồi emit revision mới thay vì freeze đến final. MT queue/revision guard nhận các revision này và chỉ giữ pending partial mới nhất.

## Streamlit realtime và metric file

Streamlit dùng realtime profile (semantic endpoint 1 câu và TTS `stable_sentence`). File feeder pace theo absolute media timeline để sai số scheduler của từng frame không cộng dồn. Khi hoàn tất, UI tạo player cùng file WAV TTS toàn bộ và hiển thị:

- input/output duration và realtime feed drift;
- input start/end, output playback start, TTS synthesis finish;
- `End-to-end elapsed`, bao gồm thời lượng media;
- ASR `input end -> final`, MT `ASR final -> MT final`, TTS `MT final -> ready`, và `Total after input`.

Các stage chạy overlap nên giá trị âm được clamp về `0 ms`. `TTS finished at` không phải playback end; nếu output duration dài hơn input thì người nghe vẫn có playback tail.

## Recipe đề xuất

### English realtime trên CPU yếu

```yaml
asr:
  backend: moonshine
  language: en
  model: tiny_streaming
  device: cpu
translation:
  backend: opus_ct2
  model: opus-auto
  target_language: vi
tts:
  enabled: true
  backend: fake
  device: cpu
  compute_type: int8
```

### Vietnamese/Chinese/Korean với Moonshine

```yaml
asr:
  backend: moonshine
  language: vi  # hoặc zh / ko
  model: auto
```

`auto` chọn model hợp lệ theo language. Không đổi thành `tiny_streaming` cho `vi/zh/ko` vì catalog hiện không có weights tương ứng.

### Auto language detection

```yaml
asr:
  backend: faster_whisper
  language: auto
  model: base
```

Hoặc dùng Dolphin `language=auto` nếu input chỉ thuộc `vi/zh/ko`.

### Test pipeline không tải weights

```yaml
vad:
  backend: passthrough
asr:
  backend: fake
  language: en
  model: fake
translation:
  backend: fake
  source_language: en
  target_language: vi
```

## Cài dependency

```powershell
# Stack mặc định
python -m pip install -e ".[moonshine,opus,app]"

# Dolphin
python -m pip install -e ".[dolphin,app]"

# Faster-Whisper hoặc M2M100
python -m pip install -e ".[models,app]"

# TTS VITS/Piper qua sherpa-onnx
python -m pip install -e ".[tts,app]"

# Tất cả backend
python -m pip install -e ".[all,app]"
```

## Khi thêm model mới

1. Thêm adapter và registry entry.
2. Khai báo model/language capability cạnh adapter và validate trước download.
3. Thêm dependency group nếu runtime mới là optional.
4. Cập nhật bảng trong file này: language, model ID, streaming, device, license và cache behavior.
5. Thêm contract/unit test không tự tải weights.

Xem quy trình chi tiết tại [ADDING_MODELS_AND_COMPONENTS.md](ADDING_MODELS_AND_COMPONENTS.md).
