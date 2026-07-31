# P3 — Multi-language Streaming Zipformer Backend

Trạng thái: **Done — 2026-07-30**

## Thứ tự và phạm vi

P3 đứng ngay sau P2 để thử true-streaming Zipformer trước các phase ASR
terminology:

```text
P2 MT terminology
  -> P3 sherpa-onnx OnlineRecognizer for vi/en/zh/ko, no terminology
  -> P4/P5 streaming + TTS terminology
  -> P6 ASR terminology cho backend hiện tại
  -> P7 Zipformer native hotwords
```

P3 không load `TerminologyManager`, không consume bundle/profile, không truyền
hotwords và không post-correct transcript.

## Model matrix

| Language | Default model | Runtime | Artifact |
|---|---|---|---|
| `vi` | `hynt-zipformer-vi-30m-streaming-6000h-chunk-32` | `OnlineRecognizer` | FP16, chunk 32, left context 128 |
| `en` | `sherpa-onnx-streaming-zipformer-en-2023-06-26` | `OnlineRecognizer` | INT8, chunk 16, left context 128 |
| `zh` | `sherpa-onnx-streaming-zipformer-zh-int8-2025-06-30` | `OnlineRecognizer` | INT8 |
| `ko` | `sherpa-onnx-streaming-zipformer-korean-2024-06-16` | `OnlineRecognizer` | INT8 |

Vietnamese được tải trực tiếp từ Hugging Face vì chưa có trong official sherpa-onnx
online model index. Ba model còn lại dùng official sherpa-onnx release archives.

Package offline `sherpa-onnx-zipformer-vi-30M-int8-2026-02-09` không còn là model
của P3. Nó có thể tồn tại trong cache từ spike trước nhưng registry/config không
chọn nó.

## Streaming contract

- Một native stream sống trong một utterance.
- `SpeechSegment` là growing snapshot; backend chỉ feed
  `samples[processed_samples:]`.
- Decode trong khi `recognizer.is_ready(stream)`.
- Final feed 500 ms zero padding, gọi `input_finished()`, decode đến khi hết ready.
- Internal sherpa endpoint detection tắt; pipeline VAD là source of truth.
- Final/reset/drop stream không giữ state sang utterance tiếp theo.

## Deliverables

- [x] Model catalog theo `vi/en/zh/ko`.
- [x] `OnlineRecognizer.from_transducer`.
- [x] Unseen-tail feed và final flush.
- [x] Registry, CLI và Streamlit model filtering.
- [x] Download/cache, manual `model_dir`, offline cache mode.
- [x] Unit tests không tải model và assert không truyền hotwords.
- [x] Load/decode smoke test cho cả bốn model thật.
- [x] End-to-end pipeline smoke test.
- [x] Full regression suite.
- [x] Semantic-endpoint hardening: re-decode backward-cut final và reset theo
  utterance/error.
- [x] English online punctuation/casing restoration; Vietnamese ALL-CAPS
  sentence-case normalization.
- [x] Strict semantic endpoint: export Sherpa token timestamps, cắt đúng câu
  thứ N và freeze partial đến muộn trong khi chờ VAD final.
- [x] Carry suffix với acoustic pre-roll 200 ms và loại context token khỏi
  transcript, tránh mất đầu câu hoặc lặp đuôi câu trước.

## Exit criteria

- [x] Mỗi language có model streaming load được trên runtime hiện tại.
- [x] Official/reference WAV decode qua chunk updates.
- [x] Pipeline nhận partial/final đúng revision và metadata.
- [x] Existing backends không regression.
- [x] Source P3 không có terminology/hotword integration.

Chi tiết: [P3 implementation report](p3-zipformer-implementation-report.md).
