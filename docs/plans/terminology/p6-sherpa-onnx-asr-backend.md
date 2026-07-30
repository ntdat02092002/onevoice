# P7 — Zipformer terminology và native hotwords

Trạng thái: **Planned — chưa implement**

> Backend foundation, model download, registry/config và decode không terminology
> đã được tách lên [P3](p3-zipformer-asr-backend.md). Tài liệu này giữ phần
> compatibility research và kế hoạch hotwords/terminology sau khi P3 ổn định;
> các mục B0–B4 bên dưới là bối cảnh lịch sử.

Phụ thuộc:

- P3 Zipformer backend foundation;
- P1 terminology profile/hotword compiler;
- quyết định license/model distribution.

## Mục tiêu

Mở rộng backend `sherpa_onnx` đã hoàn thành ở P3 bằng terminology profile và
native hotwords. Hai runtime mode dưới đây là ma trận compatibility dài hạn:

```text
online_transducer
offline_transducer
```

Phần mở rộng P7 phải:

- hỗ trợ hotwords khi dùng Transducer + `modified_beam_search`;
- không làm thay đổi backend Moonshine/Dolphin/Faster-Whisper;
- không tự download model chưa được pin/checksum.

## Candidate model matrix

| Model | Mode dự kiến | Điểm mạnh | Giới hạn/quyết định |
|---|---|---|---|
| `hynt/Zipformer-30M-RNNT-Streaming-6000h` | `online_transducer` | Streaming thật; graph chunk 16/32/64; khoảng 30M params | License CC-BY-NC-ND-4.0; phải smoke-test ONNX bằng exact sherpa version |
| `sherpa-onnx-zipformer-vi-30M-int8-2026-02-09` | `offline_transducer` | Official sherpa package; INT8; tổng model khoảng 32 MB; Transducer/hotwords | Không phải OnlineRecognizer; VAD mode là simulated streaming |
| `g-group-ai-lab/gipformer-65M-rnnt` | `offline_transducer` trước | Standard encoder/decoder/joiner, INT8, MIT, domain coverage rộng | Chỉ bật online nếu graph metadata thực sự hỗ trợ OnlineRecognizer |
| `gipformer-callbot-vi-v4` | Không thuộc generic backend ban đầu | Có ChainAttention/domain callbot | Custom graph/runtime; cần adapter riêng |

Không suy mode từ chữ `streaming` trong tên repo. Mode được xác định bằng model
manifest và compatibility smoke test.

## License gate

Trước khi model được đưa vào model zoo:

- lưu `source_url`, revision/commit và license;
- review quyền dùng thương mại và quyền redistribute;
- pin checksum cho encoder, decoder, joiner, tokens và BPE vocab;
- không repackage model `CC-BY-NC-ND` vào distribution thương mại nếu chưa được
  legal/product chấp thuận.

Backend code có thể dùng license của project; quyền sử dụng model được quản lý
riêng trong manifest.

## Kiến trúc code dự kiến

```text
src/onevoice/backends/
└── asr_sherpa.py

src/onevoice/
└── sherpa_models.py

tests/
├── test_sherpa_asr_backend.py
└── model/
    └── test_sherpa_asr_models.py
```

Không đặt model binary trong Git.

### Thành phần nội bộ

```text
SherpaOnnxAsrBackend
├── SherpaModelManifest
├── _OnlineTransducerSession
├── _OfflineTransducerSession
├── SherpaResultMapper
└── HotwordStreamFactory
```

`SherpaOnnxAsrBackend` là adapter duy nhất được registry biết. Hai session strategy
ẩn khác biệt OnlineRecognizer/OfflineRecognizer khỏi pipeline.

## Model manifest

Manifest tối thiểu:

```yaml
id: vi-zipformer-30m-streaming-6000h
language: vi
family: zipformer_transducer
recognizer_mode: online_transducer
sample_rate: 16000
encoder: encoder-epoch-31-avg-11-chunk-32-left-128.fp16.onnx
decoder: decoder-epoch-31-avg-11-chunk-32-left-128.fp16.onnx
joiner: joiner-epoch-31-avg-11-chunk-32-left-128.fp16.onnx
tokens: tokens.txt
bpe_model: bpe.model
bpe_vocab: bpe.vocab
modeling_unit: bpe
chunk_size: 32
left_context: 128
quantization: fp16
license: cc-by-nc-nd-4.0
source:
  url: https://huggingface.co/hynt/Zipformer-30M-RNNT-Streaming-6000h
  revision: null
sha256:
  encoder: null
  decoder: null
  joiner: null
  tokens: null
  bpe_vocab: null
```

Validation:

- đúng một recognizer mode;
- đủ encoder/decoder/joiner/tokens;
- file tồn tại và checksum khớp;
- language/model family hợp lệ;
- online graph có chunk/left-context đồng nhất;
- BPE hotwords có `bpe.vocab`;
- INT8/FP16 artifact phù hợp provider.

Không hard-code `cjkchar` cho tiếng Việt. Modeling unit phải lấy từ manifest và
được kiểm chứng với tokenizer/model package.

## Config dự kiến

Mở rộng `AsrConfig`:

```yaml
asr:
  backend: sherpa_onnx
  model: vi-zipformer-30m-streaming-6000h
  model_dir: assets/asr/vi-zipformer-30m-streaming-6000h
  device: cpu
  sherpa:
    recognizer_mode: auto
    provider: cpu
    num_threads: 2
    decoding_method: modified_beam_search
    max_active_paths: 4
    chunk_size: 32
    left_context: 128
    partial_strategy: incremental
    endpoint_mode: pipeline
```

Nếu config loader chưa hỗ trợ nested backend settings, lựa chọn implementation:

1. thêm `SherpaAsrConfig` vào `PipelineConfig`; hoặc
2. thêm các field có prefix `sherpa_*` vào `AsrConfig`.

Khuyến nghị lựa chọn 1 để không làm `AsrConfig` thành một tập option của mọi
runtime.

## Phase B0 — Compatibility spike

Chưa sửa pipeline.

Với từng candidate:

1. Load exact ONNX files bằng phiên bản `sherpa-onnx` đang pin.
2. Xác định recognizer API thực tế:
   - `OnlineRecognizer`; hoặc
   - `OfflineRecognizer`.
3. Decode official/test audio ở greedy mode.
4. Decode bằng `modified_beam_search`.
5. Decode với hotword inline/file.
6. Kiểm tra result:
   - text;
   - tokens;
   - timestamps;
   - partial updates;
   - final flush.
7. Ghi model load time, RTF, peak RAM và failure diagnostics.

Deliverable: compatibility report và manifest đã điền checksum.

### Candidate-specific spike

#### Streaming 30M

- Test cả chunk 16, 32 và 64.
- Xác nhận graph dùng được với sherpa `OnlineRecognizer`.
- So sánh WER/latency; model card cho thấy streaming chunk-32 có WER cao hơn bản
  non-streaming trên các benchmark được công bố, nên không chọn chỉ dựa trên tốc
  độ.
- Chọn một chunk configuration làm default; không load ba encoder cùng lúc.

#### Official INT8 30M

- Dùng `OfflineRecognizer`.
- Xác nhận `modified_beam_search` + hotwords.
- Đo hai partial strategy:
  - `final_only`: decode khi VAD final;
  - `growing_snapshot`: decode lại toàn audio snapshot.
- Không gọi simulated streaming là true incremental streaming.

#### Gipformer 65M

- Test standard encoder/decoder/joiner bằng `OfflineRecognizer` trước.
- Chỉ thử `OnlineRecognizer` nếu ONNX metadata/state input chứng minh đây là
  streaming graph.
- So sánh general Vietnamese và domain factory/call-center riêng.

## Phase B1 — Config, catalog và registry

- Thêm optional dependency `sherpa-onnx` cho ASR; tái sử dụng package đã dùng bởi
  TTS khi version compatible.
- Thêm model manifest loader/validator.
- Đăng ký:

```text
registry.register("asr", "sherpa_onnx", ...)
```

- Validate backend/model/language trước import runtime hoặc model load.
- Cập nhật CLI/Streamlit model filtering.
- Cập nhật `MODEL_ZOO.md` với mode, license, size và hotword capability.

## Phase B2 — Online Transducer adapter

Lifecycle:

### `load`

- import `sherpa_onnx`;
- validate provider;
- tạo recognizer một lần;
- chưa tạo utterance stream.

### `reset`

- destroy/drop active stream;
- reset processed sample cursor, revision và last result;
- tạo stream mới khi audio utterance tiếp theo đến.

### `transcribe`

- nếu chưa có stream, tạo stream với hotwords của immutable active profile;
- chỉ feed `SpeechSegment.samples[processed_samples:]`;
- gọi decode trong khi recognizer báo ready;
- lấy incremental result;
- map text/tokens/timestamps sang `AsrUpdate`;
- khi final, feed trailing silence nếu model yêu cầu, decode đến khi hết ready rồi
  flush/drop stream.

### `close`

- drop stream trước recognizer;
- release references;
- idempotent.

Generation reset của pipeline vẫn là source of truth. Sherpa endpoint detection
không được tự cắt utterance trong bản đầu.

## Phase B3 — Offline Transducer adapter

Hai policy:

### `final_only`

- partial không gọi model hoặc chỉ giữ draft backend khác nếu pipeline hỗ trợ;
- decode một lần khi VAD final;
- latency thấp hơn về tổng compute nhưng không có realtime subtitle/MT partial.

### `growing_snapshot`

- decode toàn `SpeechSegment.samples` ở mỗi update;
- có partial nhưng compute tăng theo chiều dài utterance;
- phải benchmark queue pressure và stale result rate.

Không dùng offline backend làm default realtime nếu p95 first-output/commit latency
không đạt budget.

## Phase B4 — Result mapping và endpoint compatibility

`AsrUpdate` cần:

- normalized display text;
- language `vi`;
- monotonic revision;
- token tuple;
- token/word timing khi model cung cấp;
- final và endpoint-cut flag.

Token timestamps không tự động tương đương word timings. Mapper phải:

- merge SentencePiece pieces bắt đầu bằng `▁`;
- tính word start từ token đầu;
- tính word end từ token kế tiếp hoặc duration estimate;
- giữ fallback rỗng nếu alignment không đáng tin cậy.

Zipformer output có thể không có punctuation. Benchmark phải xác định:

- giữ semantic endpoint dựa VAD;
- thêm punctuation restoration stage; hoặc
- chỉ endpoint khi model output có boundary đáng tin.

Không synthesize word timestamps giả chỉ để vượt guard hiện tại.

## Phase B5 — Hotwords

### Compile

- lấy canonical và aliases từ active terminology profile;
- normalize theo model tokenizer;
- loại term không tokenize được;
- giới hạn count/tokens theo profile;
- tạo score theo term hoặc global fallback.

### Activate

- stream mới nhận hotwords của profile tại utterance boundary;
- không đổi hotwords giữa utterance;
- `decoding_method` bắt buộc là `modified_beam_search`;
- validate `modeling_unit` và `bpe_vocab`.

### Tune

So sánh:

```text
greedy
modified beam, no hotwords
modified beam, low/default/high score
```

Negative corpus là acceptance gate bắt buộc để tránh false insertion.

## Phase B6 — Test plan

### Unit tests, không tải model

- config/manifest validation;
- registry creation;
- dependency missing error;
- model file/checksum mismatch;
- online feed chỉ unseen tail;
- offline growing snapshot feed toàn buffer;
- reset/final/endpoint lifecycle;
- stale generation safety;
- result/token/timestamp mapping;
- hotword compile và stream creation;
- capability mismatch;
- error không làm rò stream/reservation.

Fake `sherpa_onnx` module phải mô phỏng API, không require binary/model trong test
mặc định.

### Model-marked tests

- mỗi candidate load/decode smoke;
- hotword positive;
- near-acoustic negative;
- long utterance;
- final flush;
- repeated reset;
- CPU INT8;
- Windows runtime cùng sherpa TTS.

### End-to-end tests

- ASR partial -> Local Agreement -> MT.
- Cross-chunk terminology.
- Semantic endpoint.
- Queue coalescing dưới load.
- TTS enabled để phát hiện ONNX Runtime ABI/provider conflict.

## Benchmark và go/no-go

So sánh tối thiểu:

```text
Moonshine vi
Zipformer streaming 30M
Zipformer official offline INT8 30M
Gipformer 65M
```

Metrics:

- WER/CER;
- terminology recall/precision;
- false insertions/minute;
- first partial;
- stable commit latency;
- final latency;
- partial revision rate;
- RTF/CPU/RAM;
- model load;
- endpoint accuracy;
- punctuation/timestamp quality.

Quyết định độc lập:

- backend technical compatibility;
- model quality;
- hotword benefit;
- production license.

Model không đạt license gate vẫn có thể dùng cho research benchmark nhưng không
được chọn production default.

## Rollout

1. Backend tồn tại nhưng không xuất hiện mặc định.
2. CLI-only opt-in với local `model_dir`.
3. Streamlit experimental selector.
4. Canary nội bộ theo model ID.
5. Chọn default tiếng Việt nếu vượt toàn bộ go/no-go gate.
6. Giữ Moonshine fallback ít nhất một release.

## Definition of Done

- Một adapter hỗ trợ rõ ràng online và offline Transducer.
- Streaming model chỉ feed unseen audio tail.
- Official offline model không bị quảng bá nhầm thành true streaming.
- Hotwords hoạt động ở utterance boundary và có negative tests.
- Model manifest pin source, license và checksum.
- Existing ASR backends/test suite không regression.
- Có benchmark report và quyết định model riêng, không gắn cứng backend với một
  model duy nhất.

## Tài liệu tham khảo

- [hynt Vietnamese streaming Zipformer 30M](https://huggingface.co/hynt/Zipformer-30M-RNNT-Streaming-6000h)
- [Official sherpa Vietnamese INT8 30M documentation](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-transducer/zipformer-transducer-models.html)
- [Gipformer 65M RNNT model](https://huggingface.co/g-group-ai-lab/gipformer-65M-rnnt)
- [sherpa-onnx hotwords](https://k2-fsa.github.io/sherpa/onnx/hotwords/index.html)
- [sherpa-onnx Online Transducer API](https://k2-fsa.github.io/sherpa/onnx/c-api/html/online_asr.html)
- [sherpa-onnx Offline ASR API](https://k2-fsa.github.io/sherpa/onnx/c-api/html/offline_asr.html)
