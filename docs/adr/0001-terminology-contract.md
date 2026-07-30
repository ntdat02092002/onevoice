# ADR-0001 — Terminology Bundle và Runtime Contract

- Status: **Accepted**
- Date: 2026-07-30
- Scope: OneVoice terminology dictionary P0

## Context

OneVoice cần bảo toàn thuật ngữ xuyên suốt ASR, Stable Prefix, MT, phrase
chunking và TTS. Một bước `str.replace()` sau MT không xử lý được lỗi ASR, có thể
thay sai substring, không bảo vệ pivot translation và không kiểm soát spoken form.

Hệ thống hiện chưa có terminology runtime. Quyết định này chốt contract dữ liệu và
failure policy trước khi P1 bắt đầu implement loader/matcher.

## Decision

### Một master bundle

- Master format là YAML UTF-8.
- `schema_version` đầu tiên là `1`.
- Mỗi entry biểu diễn một concept bằng `id` ổn định.
- Canonical form, aliases và ASR boost nằm theo language.
- Spoken form tách khỏi canonical display form.
- Bundle có thể chứa nhiều domain; runtime profile lọc theo domain/session.

### Language và normalization

- Language code hợp lệ: `vi`, `en`, `zh`, `ko`.
- Chuẩn hóa chung: Unicode NFC và whitespace.
- Không bỏ dấu tiếng Việt toàn cục.
- Match có thể case-fold nhưng restore phải giữ canonical case.
- Trung Quốc dùng character-aware matching, không phụ thuộc whitespace.
- Hàn Quốc hỗ trợ alias có/không spacing.
- Mọi normalized match phải giữ alignment về original span.

### Translation policies

| Policy | Contract |
|---|---|
| `preserve` | Giữ canonical source/display qua MT |
| `preferred_term` | Restore canonical form của target language |
| `transliterate` | Dùng canonical transliteration đã khai báo |
| `expand` | Mở rộng acronym/short form theo target canonical |
| `spell_out` | Giữ display form, ưu tiên spelled-out spoken form |
| `display_preserve_speech_override` | Display giữ canonical, TTS dùng spoken form |

P1 validator phải reject policy ngoài danh sách này.

### Conflict resolution

Thứ tự duy nhất:

```text
longest match -> priority cao hơn -> declaration order
```

Alias trùng chỉ hợp lệ khi domain hoặc priority làm kết quả deterministic.

### MT protection và failure policy

Hard terms được protect trước inference. Sau inference phải kiểm tra:

- missing placeholder;
- duplicate placeholder;
- unexpected placeholder;
- case/whitespace corruption;
- order violation nếu policy yêu cầu.

Fallback order:

1. Retry bằng placeholder format dự phòng đã benchmark.
2. Dịch không placeholder và target-side canonicalize với rule an toàn.
3. Với `preserve`, giữ source canonical.
4. Raise terminology error với hard safety term không thể bảo toàn.

Không emit output còn raw/corrupted placeholder.

### Streaming và lifecycle

- Một utterance chỉ dùng một immutable terminology profile.
- Profile mới chỉ activate tại session start hoặc utterance boundary.
- Stable Prefix giữ suffix đang là open prefix của term dài hơn.
- TTS chunker không tạo boundary giữa protected target term.
- Nội dung đã phát TTS không được sửa hồi tố.

### Compatibility

- Terminology mặc định disabled.
- Disabled mode phải giữ nguyên output/lifecycle hiện tại.
- Model-dependent artifact phải pin model/tokenizer/tokens checksum.
- Bundle data và model license được quản lý độc lập.

## Initial artifacts

- Sample bundle:
  `assets/terminology/factory-sample-v1/terminology.yaml`
- Benchmark corpus:
  `tests/fixtures/terminology/p0-benchmark-corpus.yaml`
- Baseline:
  `docs/plans/terminology/p0-baseline-report.md`

Các artifact P0 là contract/sample, chưa được runtime load.

## Consequences

### Positive

- Một nguồn dữ liệu cho mọi pipeline stage.
- Direct và pivot MT dùng cùng canonical mapping.
- Có thể benchmark/tune từng backend mà không đổi master data.
- Display text và spoken text không bị trộn.

### Cost

- P1 cần strict loader, normalization alignment và matcher.
- Model-specific compiler phải xử lý tokenizer compatibility.
- Hotword score/profile cần negative testing.
- Bundle activation cần lifecycle rõ để tránh mixed profile.

## Deferred

- Vector retrieval.
- LLM glossary/KV cache.
- Constrained beam search cho MT.
- Direct phoneme IDs.
- Mid-utterance bundle swap.
