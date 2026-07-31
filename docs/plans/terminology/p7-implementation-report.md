# P7 — Native Sherpa Zipformer hotwords: Implementation Report

Trạng thái: **Done**

## Kết quả

Terminology profile hiện được biên dịch thành hotword list riêng cho language
đang chạy và truyền trực tiếp vào mỗi `OnlineRecognizer` stream của backend
`sherpa_onnx`.

Luồng thực tế:

```text
active terminology profile
  -> canonical + aliases + per-term asr_boost
  -> giới hạn term/token
  -> model-specific casing và tokenizer validation
  -> Sherpa create_stream(inline_hotwords)
  -> exact-alias post-correction trước commit/MT
```

Native hotword là contextual bias trong lúc Transducer beam search, không phải
thay thế text sau khi nhận dạng. Lớp exact-alias correction của P6 vẫn được giữ
để canonicalize alias và xử lý các term không đi qua native tokenizer.

## Backend strategy

| Backend | Cơ chế terminology ASR |
|---|---|
| Sherpa streaming Zipformer Transducer | Native hotwords trên `modified_beam_search`, sau đó exact-alias correction |
| Faster-Whisper | `initial_prompt` có giới hạn, sau đó exact-alias correction |
| Moonshine/Dolphin | Exact-alias correction; không truyền option native chưa được backend xác nhận |

`initial_prompt` chỉ là ngữ cảnh mềm của decoder. Exact-alias correction chỉ sửa
các alias khai báo rõ trong active profile. Native hotwords tác động ngay lúc
Sherpa chọn token/path nên phù hợp hơn cho tên riêng và thuật ngữ dễ nghe nhầm,
nhưng vẫn không phải hard guarantee.

## Model compatibility

- Vietnamese, English và Korean dùng BPE. Backend tạo `bpe.vocab` từ
  `bpe.model` một lần trong model cache.
- Chinese dùng `cjkchar`.
- Term chứa `<unk>`, BPE piece không có trong `tokens.txt`, hoặc CJK character
  ngoài vocabulary sẽ không được đưa vào native stream.
- Rejected term không làm hỏng stream và vẫn có thể được xử lý bởi P6
  post-correction nếu ASR tạo ra một alias đã khai báo.
- Native hotwords chỉ được bật cho Transducer với
  `modified_beam_search`. Khi app đang dùng Sherpa + terminology và config còn
  là `greedy_search`, pipeline tự chuyển decoding mode trước validation/load.

## Config

```yaml
terminology:
  asr:
    native_hotwords_enabled: true
    max_hotword_terms: 64
    max_hotword_tokens: 256
    hotword_score: 1.5
```

`asr_boost` của từng term được ghi trên inline hotword. `hotword_score` là
recognizer-level default/baseline; cần tune bằng negative corpus trước khi tăng
rộng trong production.

## Metrics

- `asr_hotword_auto_beam_switch`
- `asr_hotword_term_count`
- `asr_hotword_token_count`
- `asr_hotword_rejection_count`
- `asr_hotword_stream_count`
- `asr_post_correction_count`
- `asr_post_correction_timing_drops`

## Verification

### Real-model stream creation

Profile `factory-maintenance` đã được smoke test offline từ model cache:

| Language | Compiled | Native accepted | Rejected | Stream |
|---|---:|---:|---:|---:|
| vi | 15 | 15 | 0 | tạo thành công |
| en | 13 | 6 | 7 | tạo thành công |
| zh | 11 | 3 | 8 | tạo thành công |
| ko | 12 | 12 | 0 | tạo thành công |

Số accepted phụ thuộc chính xác tokenizer/vocabulary của model và profile, không
phải danh sách capability cố định theo language.

### English A/B canary

Trên 130 giây đầu của `data/b2-test-64-2.mp3`, cùng English streaming Zipformer
và `modified_beam_search`:

| Mode | `WIND SURFING` | `OUTDOOR LIFE` |
|---|---:|---:|
| Không hotword | 0; model ra `WIND SERVING` | 1 |
| Profile `test` hotwords | 5 | 1 |

Sau native decode, alias `wind surfing` được P6 canonicalize thành
`windsurfing`.

Đây là canary chứng minh integration có tác dụng trên sample người dùng, chưa
phải benchmark production. False insertion, WER/CER, latency và RTF vẫn cần đo
trên positive/negative corpus đủ lớn trước khi đổi score hoặc đưa Sherpa thành
default.

## Tests

Coverage gồm:

- compile canonical/alias với score, casing và budgets;
- tự chuyển sang `modified_beam_search`;
- reject native hotwords ở decoding mode không tương thích;
- tạo inline hotword stream;
- export BPE vocabulary;
- lọc BPE/CJK term không biểu diễn được và metrics;
- regression của prompt/post-correction hiện có.

## Tài liệu tham khảo

- [sherpa-onnx Hotwords / Contextual biasing](https://k2-fsa.github.io/sherpa/onnx/hotwords/index.html)
- [hynt Vietnamese streaming Zipformer 30M](https://huggingface.co/hynt/Zipformer-30M-RNNT-Streaming-6000h)
