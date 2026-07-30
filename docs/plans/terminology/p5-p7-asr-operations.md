# P5–P7 — ASR Capability, Lifecycle và Operations

Trạng thái: **Planned — chưa implement**

Phụ thuộc: P1. Nên thực hiện sau khi P2–P4 đã có baseline.

## P5 — ASR terminology cho backend hiện tại

### Hiện trạng

Các ASR backend hiện có:

- Moonshine native streaming.
- Dolphin.
- Faster-Whisper.
- Fake backend cho test.

Repo chưa có Sherpa Transducer ASR. Vì vậy không được giả định mọi backend hỗ trợ hotword/contextual beam bias.

### Capability model

Mỗi backend cần công bố capability:

```text
none
initial_prompt
native_hotwords
post_correction
```

Profile compiler chỉ sinh artifact mà backend đã công bố hỗ trợ.

### Faster-Whisper

- Inject danh sách nhỏ vào `initial_prompt`.
- Filter theo domain và language.
- Giới hạn số term/token prompt.
- Không xem prompt là hard constraint.
- Test hallucination/false insertion với negative audio.

### Moonshine và Dolphin

- Không truyền option chưa được backend API xác nhận.
- Có thể dùng exact/confusion-map post-correction opt-in.
- Fuzzy edit-distance correction không bật mặc định.

### Post-correction guardrails

- Chỉ sửa term từ active profile.
- Ưu tiên exact alias/confusion map.
- Không sửa từ phổ biến chỉ vì gần edit distance.
- Log original/canonical term ID, không log raw audio.
- Hard safety term correction cần confidence/rule rõ ràng.

### Metrics

- Term recall.
- Term precision.
- False term insertions per minute.
- Prompt term count.
- Post-correction count và rejection count.
- ASR latency delta.

### P5 exit criteria

- Backend không hỗ trợ vẫn chạy bình thường.
- Prompt/correction không bật ngoài active domain.
- Có positive và near-acoustic negative benchmark.
- Không tuyên bố hard guarantee cho prompt-based ASR.

## P6 — Đánh giá chuyển sang sherpa-onnx Zipformer

Plan implementation chi tiết: [Backend sherpa-onnx ASR cho Zipformer/Gipformer tiếng Việt](p6-sherpa-onnx-asr-backend.md).

### Kết luận kiến trúc ban đầu

Không thực hiện big-bang switch toàn bộ ASR.

Hướng đề xuất là thêm `sherpa_zipformer` như một backend độc lập, chạy benchmark
song song với backend hiện tại, sau đó enable theo từng language/model khi đạt gate.

Lý do:

- sherpa-onnx chỉ hỗ trợ hotwords cho Transducer với
  `modified_beam_search`; Zipformer CTC, Paraformer và các model không phải
  Transducer không có capability này;
- official catalog có online/streaming Zipformer Transducer cho English, Chinese
  và Korean;
- Vietnamese package trong official sherpa model catalog hiện thuộc nhóm offline
  Transducer. Ngoài catalog có `hynt/Zipformer-30M-RNNT-Streaming-6000h` với
  online chunk graph, nhưng cần compatibility và license gate trước khi chọn;
- một model multilingual streaming duy nhất không cover đủ Việt–Anh–Trung–Hàn;
- OneVoice đang dùng growing ASR snapshots và semantic endpoint dựa một phần vào
  timestamp/punctuation, nên model switch phải benchmark cả partial stability và
  endpoint behavior, không chỉ WER.

### Model coverage cần xác nhận trong spike

| Ngôn ngữ | Candidate ban đầu | Streaming state | Hotwords | Quyết định ban đầu |
|---|---|---|---|---|
| `en` | Streaming Zipformer Transducer | Online | Có | Canary |
| `zh` | Streaming Zipformer Transducer | Online | Có | Canary |
| `ko` | Streaming Zipformer Transducer | Online | Có | Canary |
| `vi` | hynt Streaming Zipformer 30M | Online candidate | Có nếu sherpa-compatible | Canary sau compatibility/license gate |
| `vi` | Official sherpa Zipformer INT8 30M | Offline/VAD segment | Có | Benchmark, không gọi là true streaming |

Model ID, license, model size, tokenizer và checksum phải được pin trong
`MODEL_ZOO.md`; không dùng tên chung `zipformer-auto`.

### P6.0 — Benchmark spike, chưa tích hợp pipeline

So sánh trên cùng corpus:

1. Backend hiện tại.
2. Zipformer `greedy_search`.
3. Zipformer `modified_beam_search` không hotwords.
4. Zipformer `modified_beam_search` có hotwords ở nhiều score.

Corpus phải có:

- speech tổng quát;
- exact terminology;
- multi-token terminology;
- alias;
- near-acoustic negative;
- code/product name;
- noisy/far-field audio;
- code-switch nếu domain thực tế có sử dụng.

Metrics:

- WER/CER tổng quát;
- Term Recall và Term Precision;
- False Term Insertions per Minute;
- first partial latency;
- stable commit latency;
- RTF và CPU;
- peak RAM và model load time;
- partial revision rate;
- token/timestamp availability;
- punctuation và semantic endpoint behavior.

Không dùng RTF từ model documentation làm acceptance result cho thiết bị đích.

### P6.1 — Backend canary

Thêm adapter qua registry, không hard-code vào pipeline:

```text
backend: sherpa_zipformer
mode: online_transducer
model_dir: ...
encoder: ...
decoder: ...
joiner: ...
tokens: ...
modeling_unit: bpe | cjkchar | cjkchar+bpe
bpe_vocab: ...
decoding_method: modified_beam_search
max_active_paths: 4
```

Adapter online phải:

- tạo recognizer một lần cho model/profile;
- tạo stream mới ở utterance boundary;
- chỉ feed unseen audio tail;
- decode khi stream ready;
- map incremental result sang `AsrUpdate`;
- map token timestamps sang word timing khi khả thi;
- reset/destroy stream đúng generation;
- không để sherpa endpoint tự ý cạnh tranh với VAD/semantic endpoint của pipeline
  trong phase đầu.

### P6.2 — Hotword artifact và per-stream activation

- Compile canonical/alias term theo tokenizer/modeling unit.
- Validate `tokens.txt` và `bpe.vocab` checksum.
- Bắt buộc `modified_beam_search`.
- Tạo hotword stream tại utterance start từ immutable terminology profile.
- Tune score theo model, language và domain.
- Giới hạn số term/profile để kiểm soát false insertion và memory.
- Fail sớm nếu config yêu cầu hotwords nhưng model không phải Transducer.

Không dùng một global hotword score cho mọi model/ngôn ngữ.

### P6.3 — Vietnamese decision

Không đưa Vietnamese offline Zipformer vào realtime primary path chỉ để có
hotwords.

Các lựa chọn phải được benchmark riêng:

1. Giữ Moonshine streaming cho `vi`, dùng post-correction/terminology ở tầng text.
2. Dùng `hynt/Zipformer-30M-RNNT-Streaming-6000h` sau khi vượt compatibility,
   quality và license gate.
3. Dùng official offline Zipformer theo final VAD segment, chấp nhận mất partial
   realtime.
4. Chuẩn bị/train một online Vietnamese Zipformer Transducer có license phù hợp.

Không dùng mô hình two-pass “Moonshine partial + Zipformer final” nếu TTS đã phát
partial, trừ khi thiết kế lại rollback/correction policy. Final correction sau khi
audio đã phát có thể tạo nội dung mâu thuẫn không thể thu hồi.

### Go/no-go gate

Cho phép Zipformer trở thành default của một ngôn ngữ khi:

- terminology recall tăng có ý nghĩa so với baseline;
- general WER/CER không vượt regression budget đã chốt;
- false term insertion đạt ngưỡng negative corpus;
- p95 first-partial và commit latency đạt realtime budget;
- peak RAM/model load phù hợp thiết bị đích;
- semantic endpoint, timestamp và punctuation không làm giảm trải nghiệm;
- model/license/distribution đã được duyệt.

Nếu chỉ hotword metric tốt nhưng general ASR hoặc latency kém, giữ Zipformer dưới
dạng optional domain profile thay vì default.

### Không thuộc P6

- Keyword spotting thay full ASR.
- Mid-utterance hotword swap.
- Switch cả bốn ngôn ngữ trong một release.
- Tự động download một model không được pin/checksum.

### P6 exit criteria

- Có benchmark report theo từng language/device.
- `sherpa_zipformer` là adapter tùy chọn và không ảnh hưởng backend hiện tại.
- Native hotword artifact tương thích model.
- Term recall/precision và false insertion được đo với/không hotwords.
- Chỉ những language vượt go/no-go gate mới được enable ngoài canary.

## P7 — Bundle lifecycle và compatibility

### Activation

- Load/validate bundle trước khi inference.
- Tạo immutable profile cho session/domain/language route.
- Một utterance chỉ dùng một profile version.
- Swap request giữa utterance được áp dụng ở utterance kế tiếp.

### Compatibility manifest

```yaml
bundle_id: factory-2026-07-v1
schema_version: 1
asr:
  model_id: null
  tokenizer_sha256: null
mt:
  route:
    - vi
    - en
    - ko
  models:
    - opus-mt-vi-en
    - opus-mt-en-ko
tts:
  model_id: null
  tokens_sha256: null
```

Checksum chỉ bắt buộc với artifact phụ thuộc model. Spoken-form text không cần phoneme checksum.

### Rollback

- Giữ profile/bundle trước gần nhất.
- Activation mới chỉ publish sau khi toàn bộ validation pass.
- Failure không làm mất active profile.
- Rollback cũng áp dụng tại utterance boundary.

## P7 — Config, CLI và Streamlit

### Config

```yaml
terminology:
  enabled: false
  bundle_path: null
  domain: null
  apply_at: utterance_boundary
  mt:
    strategy: placeholder_with_validation
    validate_coverage: true
    pivot_canonicalization: true
  stable_prefix:
    term_prefix_timeout_ms: 400
  chunker:
    protect_term_spans: true
  tts:
    strategy: spoken_form
```

### CLI

- `--terminology-bundle`
- `--terminology-domain`
- Option chỉ override config; không tạo schema riêng.

### Streamlit

- Bundle/domain selector bị khóa khi utterance đang chạy.
- Hiển thị active bundle ID/profile.
- Báo compatibility error trước khi bật microphone/file processing.

## P7 — Observability

Nhóm metric:

- bundle load/validation/activation;
- active bundle/profile version;
- match/conflict;
- placeholder validation/fallback;
- term commit latency;
- protected chunk adjustments;
- spoken-form substitutions;
- ASR term recall/precision benchmark.

Event lỗi cần phân biệt:

- bundle/schema error;
- profile/route coverage error;
- model compatibility error;
- MT terminology validation error;
- ASR capability mismatch.

## P7 test plan

- Invalid bundle không replace active profile.
- Swap chỉ có hiệu lực ở utterance kế tiếp.
- Rollback.
- Concurrent MT job giữ profile version cũ.
- Config deep merge.
- CLI override.
- Streamlit control locking.
- Manifest/checksum mismatch.
- Disabled mode không yêu cầu bundle.

## P7 exit criteria

- Activation atomic.
- Không có mixed profile trong cùng utterance.
- Có rollback được kiểm thử.
- UI/CLI dùng chung config contract.
- Metrics đủ để xác định lỗi thuộc ASR, commit, MT, chunker hay TTS.

## P8 — R&D sau prototype

Chỉ mở khi metric cho thấy baseline chưa đạt:

- constrained beam search;
- dictionary-aware fine-tuning/copy mechanism;
- LLM prompt và KV cache;
- vector glossary retrieval;
- custom phoneme lexicon/direct phoneme IDs.

Mỗi hướng P8 cần ADR và benchmark riêng; không thêm dependency vào prototype mặc định.

## Tài liệu kỹ thuật dùng cho quyết định P6

- [sherpa-onnx Hotwords / Contextual biasing](https://k2-fsa.github.io/sherpa/onnx/hotwords/index.html)
- [Online Transducer model catalog](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/index.html)
- [Offline Transducer model catalog](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-transducer/index.html)
- [Vietnamese Zipformer model documentation](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-transducer/zipformer-transducer-models.html)
- [Streaming ASR C API](https://k2-fsa.github.io/sherpa/onnx/c-api/html/online_asr.html)
- [Online recognizer result and token timestamps](https://k2-fsa.github.io/sherpa/onnx/c-api/html/structSherpaOnnxOnlineRecognizerResult.html)
