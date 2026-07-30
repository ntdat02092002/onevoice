# P2 — MT Terminology Integration

Trạng thái: **Done — 2026-07-30**

Phụ thuộc: P0 và P1.

## Mục tiêu

Đảm bảo hard terms được bảo vệ trước MT, canonicalize đúng ngôn ngữ đích và được validate trước khi output đi xuống TTS.

Đây là mốc MVP terminology đầu tiên.

## Kiến trúc

```text
source text
  -> normalize + match
  -> select non-overlapping terms
  -> replace with placeholders
  -> MT inference
  -> validate placeholder coverage
  -> restore target canonical forms
  -> terminology-safe output
```

## Placeholder contract

Binding tối thiểu:

```text
placeholder
term_id
source_language
target_language
source_span
source_form
target_canonical
translation_policy
```

Validator phải kiểm tra:

- missing placeholder;
- duplicate placeholder;
- unexpected placeholder;
- placeholder bị đổi case;
- placeholder bị chèn whitespace;
- order change khi policy yêu cầu giữ thứ tự.

## Lựa chọn placeholder

Benchmark tối thiểu:

- `__TERM_0001__`
- `<TERM_0001>`
- `ZXTERM0001ZX`

Format được chọn theo từng tokenizer/model pair, không chọn chỉ dựa trên khả năng đọc của con người.

## OPUS direct route

Với một hop:

```text
protect source -> translate once -> validate -> restore target canonical
```

Final translation hiện đang split theo sentence. Protection và validation phải chạy độc lập cho từng sentence để failure không làm mất binding giữa các câu.

## OPUS pivot route

Canonicalization phải xảy ra ở từng hop:

```text
vi source
  -> protect vi terms
  -> OPUS vi-en
  -> validate + restore English canonical
  -> protect English canonical terms
  -> OPUS en-ko
  -> validate + restore Korean canonical
```

Không chỉ giữ một placeholder xuyên qua hai model vì:

- survival rate giảm qua nhiều tokenizer/model;
- intermediate synonym có thể làm hop tiếp theo không match;
- khó xác định hop nào làm hỏng binding.

## M2M100

M2M100 là direct multilingual route nên dùng cùng protector/validator abstraction nhưng chỉ chạy một hop.

## Fallback policy

Theo thứ tự cấu hình:

1. Retry với placeholder format đã benchmark thứ hai.
2. Dịch không placeholder và canonicalize target-side cho term an toàn.
3. Với `preserve`, giữ source canonical.
4. Raise terminology error nếu hard safety term không thể bảo toàn.

Không emit output chứa placeholder chưa restore.

Fallback cần có metric và event riêng; không được bị ghi chung như lỗi model MT.

## Integration points dự kiến

- `src/onevoice/backends/translation.py`
  - OPUS: tích hợp trong vòng lặp route/hop.
  - M2M100: tích hợp quanh direct inference.
- `src/onevoice/models.py`
  - chỉ thêm terminology metadata nếu observability thật sự cần đi qua queue;
  - tránh đưa mutable profile vào request.
- `src/onevoice/pipeline.py`
  - inject immutable active profile vào translator;
  - emit terminology metrics/errors.
- `src/onevoice/terminology/compiler/mt.py`
- `src/onevoice/terminology/runtime/mt_protector.py`, nếu tách runtime subpackage.

## Concurrency và revision

- Binding chỉ sống trong một translation request/hop.
- Không cache binding theo revision toàn cục. Chỉ cache translated completed
  sentence bằng exact source text trong một `stream_id`; binding vẫn request-local.
- Partial mới coalesce không được làm binding của inference đang chạy thay đổi.
- Final jobs vẫn dùng lossless lane hiện tại.
- Terminology profile phải immutable trong suốt utterance.

## Test plan

### Matcher/protector

- Một term.
- Nhiều term.
- Alias.
- Overlap.
- Code và punctuation.
- Chinese không có whitespace.

### Validator

- Missing.
- Duplicate.
- Extra.
- Case mutation.
- Inserted whitespace.
- Reordered placeholders.

### Translation backend

- OPUS direct.
- OPUS pivot.
- Final sentence-by-sentence.
- Partial sentence-aware khi bật terminology: completed sentence exact-cache,
  mutable tail luôn dịch lại.
- M2M100 direct.
- Source và target giống nhau.
- Fake backend/no-op khi terminology tắt.

### Regression

- Existing translation tests không đổi kết quả khi terminology tắt.
- Metadata `source_revision`, language và final flag được giữ nguyên.
- Terminal punctuation restoration vẫn chạy sau canonical restore.

## Metrics

- `terminology_matches`
- `terminology_hard_matches`
- `mt_placeholder_expected`
- `mt_placeholder_missing`
- `mt_placeholder_corrupted`
- `mt_placeholder_retry`
- `mt_terminology_fallback`
- `mt_pivot_terms_preserved`
- protection/validation latency

## Exit criteria

- [x] Hard-term fixtures qua direct route đạt canonical target.
- [x] Pivot fixtures canonicalize đúng ở từng hop.
- [x] Mọi placeholder corruption được phát hiện.
- [x] Không output placeholder thô.
- [x] Disabled mode giữ nguyên output và latency path hiện tại ngoài chi phí kiểm tra cờ.

## Kết quả implementation

P2 đã hoàn tất. Chi tiết implementation, benchmark model thật và hướng dẫn test app:
[P2 implementation report](p2-implementation-report.md).
