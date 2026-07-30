# P2 — MT Terminology Implementation Report

Trạng thái: **Done — 2026-07-30**

## Kết quả

P2 đã nối terminology core vào toàn bộ MT path hiện có:

- OPUS CTranslate2 direct route và canonicalization sau từng pivot hop;
- M2M100 direct multilingual route;
- fake MT backend để smoke-test không cần model;
- pipeline tự load bundle và inject immutable terminology runtime khi
  `terminology.enabled: true`;
- sidebar Streamlit có control bật bundle sample và chọn domain.

Khi terminology tắt, backend tiếp tục chạy đường inference cũ và không thực hiện
match/protect/validate.

## Runtime contract

Mỗi MT hop thực hiện:

```text
match source terms
  -> replace non-overlapping spans with request-local placeholders
  -> infer MT
  -> validate exact placeholder coverage
  -> restore target canonical forms
```

Validator phát hiện placeholder bị mất, lặp, đổi case, chèn whitespace, xuất hiện
ngoài binding và đổi thứ tự khi `validate_order` được bật. Binding không được cache
giữa request. Nếu toàn bộ format dự phòng đều hỏng, runtime raise
`TerminologyCoverageError`; output chứa placeholder thô không được emit xuống TTS.

Sau kiểm thử runtime với câu dài, OPUS có thể làm hỏng cả ba sentinel dù
đầu vào đã được tách theo câu. Cấu hình mặc định vì vậy dùng
`on_validation_error: segment_fallback` với hai tầng an toàn:

1. Nếu output vẫn có đúng một placeholder-like span cho mỗi binding, sửa
   sentinel biến dạng theo thứ tự rồi chạy lại validator nghiêm ngặt.
2. Nếu placeholder đã bị mất hẳn, dịch riêng các source span nằm quanh term
   và chèn canonical term trực tiếp. Nhánh này ưu tiên coverage hơn độ trôi
   chảy và chỉ chạy sau khi mọi placeholder format đều thất bại.

Metric `mt_terminology_fallback` đếm số hop phải đi qua nhánh này. Chế độ
`raise` vẫn được giữ để benchmark hoặc fail-fast khi cần.

Với OPUS pivot, output canonical của hop trước trở thành input để match/protect lại
ở hop sau. Vì vậy có thể xác định chính xác hop nào làm hỏng coverage.

## Cấu hình

Default vẫn tắt:

```yaml
terminology:
  enabled: false
  bundle_path: null
  domain: null
  mt:
    strategy: placeholder_with_validation
    placeholder_formats:
      - "__TERM_{id:04d}__"
      - "OVT{id:04d}OVT"
      - "ZXTERM{id:04d}ZX"
    validate_coverage: true
    pivot_canonicalization: true
    validate_order: false
    on_validation_error: segment_fallback
```

`validate_order` mặc định tắt vì MT hợp lệ có thể đổi trật tự thành phần câu. Các
hard-safety guarantee còn lại không cho phép tắt khi terminology đang enabled.

## Benchmark placeholder với model cache thật

Smoke corpus: `Press the emergency stop button now.` với bundle
`factory-sample-v1`, domain `factory-safety`, OPUS INT8 offline cache.

| Route | Baseline P0 | P2 output | Retry |
|---|---|---|---:|
| `en→vi` | term accuracy `5/6` trên corpus P0 | `Nhấn nút dừng khẩn cấp ngay.` | 1 |
| `en→zh` | term accuracy `3/6` trên corpus P0 | `现在就按紧急停止按钮。` | 1 |

Candidate `<TERM_0001>` trong plan không sống sót qua model `en→zh`, nên được thay
bằng `OVT0001OVT` sau benchmark. Format human-readable đầu tiên vẫn được giữ để
quan sát/debug; runtime tự retry format đã benchmark khi cần.

## Observability

`TranslationUpdate` và pipeline expose:

- `terminology_matches`;
- `terminology_hard_matches`;
- `mt_placeholder_expected`;
- `mt_placeholder_retry`;
- `mt_terminology_validation_error`;
- số terminology hop trên update.

## Verification

- Full automated suite: **170 passed**.
- Unit/integration coverage gồm alias, overlap matcher từ P1, missing/duplicate/
  corrupted/extra placeholder, retry, OPUS direct/pivot, M2M100, fake backend,
  pipeline injection và disabled regression.
- `git diff --check`: pass.
- Streamlit headless smoke test: HTTP 200 trên local test port; process test đã
  được dừng sau khi xác minh.

## Test app

Chạy:

```powershell
.\venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```

Trong sidebar:

1. Chọn source/target và MT backend.
2. Bật **Terminology dictionary**.
3. Chọn `factory-safety`, `factory-maintenance` hoặc profile thử nghiệm `test`.
4. Giữ bundle sample mặc định rồi khởi tạo pipeline.

Có thể chọn MT backend `fake` để kiểm tra wiring/canonical restore nhanh mà không
nạp model. Chọn `opus_ct2` để test inference thật; nếu bật offline thì route tương
ứng phải có sẵn trong cache.

Profile `test` bảo vệ `windsurfing` (aliases `wind surfing`, `winssurfing`) và
tên riêng `Outdoor Life` bằng policy `preserve`. Smoke test OPUS `en -> vi` phải
giữ đúng hai canonical string này trong translated transcript.

OPUS và M2M100 partial có terminology được split theo sentence. Completed
sentence cache dùng `(generation, utterance_id) + language pair + exact source`;
mutable tail luôn inference lại, source correction tạo cache miss, final reuse
rồi xóa cache. Metric `mt_sentence_cache_hits` cho biết số chunk được tái sử dụng.
