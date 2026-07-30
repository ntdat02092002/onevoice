# P0–P1 — Baseline, Data Model và Terminology Core

Trạng thái:

- **P0: Done — 2026-07-30**
- **P1: Done — 2026-07-30**

Phụ thuộc: không có.

## P0 — Baseline và quyết định kiến trúc

Kết quả:

- [ADR terminology contract](../../adr/0001-terminology-contract.md)
- [P0 baseline report](p0-baseline-report.md)
- [Sample terminology bundle](../../../assets/terminology/factory-sample-v1/terminology.yaml)
- [Benchmark corpus](../../../tests/fixtures/terminology/p0-benchmark-corpus.yaml)

### Mục tiêu

Chốt contract trước khi thay đổi pipeline và tạo số đo baseline để đánh giá lợi ích cũng như latency regression.

### Công việc

1. Chốt master format là YAML, sử dụng `PyYAML` hiện có.
2. Chốt schema version đầu tiên và compatibility policy.
3. Xác định nhóm translation policy:
   - `preserve`
   - `preferred_term`
   - `transliterate`
   - `expand`
   - `spell_out`
   - `display_preserve_speech_override`
4. Chốt conflict policy:
   - longest match;
   - priority;
   - declaration order.
5. Chốt MT failure policy:
   - validate placeholder coverage;
   - thử fallback đã cấu hình;
   - không trả output còn placeholder lỗi.
6. Xây corpus nhỏ có term dương tính, overlap, alias, pivot và near-match âm học.
7. Ghi baseline:
   - MT terminology accuracy;
   - ASR term recall/precision nếu có audio;
   - first-output latency;
   - MT/TTS latency hiện tại.

### Deliverable

- Architecture decision record cho terminology contract.
- Test corpus fixture dùng xuyên suốt các phase.
- Baseline report.

### Exit criteria

- [x] Các policy không còn mơ hồ.
- [x] Corpus có đủ Việt, Anh, Trung và Hàn.
- [x] Existing tests pass trước khi bắt đầu P1 (`135 passed`).

## P1 — Data model và terminology core

Kết quả:

- [P1 implementation report](p1-implementation-report.md)
- Core package: `src/onevoice/terminology/`
- Config mặc định disabled: `config/default.yaml`
- Full regression: `156 passed`

### Cấu trúc module dự kiến

```text
src/onevoice/terminology/
├── __init__.py
├── schema.py
├── loader.py
├── normalizer.py
├── matcher.py
├── profile.py
├── manager.py
├── errors.py
└── compiler/
    ├── __init__.py
    ├── asr.py
    ├── mt.py
    └── tts.py
```

Compiler có thể bắt đầu bằng artifact in-memory. Không cần sinh toàn bộ file vật lý trong P1.

### Master bundle schema tối thiểu

```yaml
bundle_id: factory-2026-07-v1
schema_version: 1
entries:
  - id: emergency_stop_button
    domain:
      - factory-safety
    priority: 100
    translation_policy: preferred_term
    forms:
      vi:
        canonical: nút dừng khẩn cấp
        aliases:
          - nút E-stop
        asr_boost: 1.8
      en:
        canonical: emergency stop button
        aliases:
          - E-stop button
      ko:
        canonical: 비상 정지 버튼
        aliases:
          - 비상정지 버튼
    tts:
      en:
        spoken_form: emergency stop button
```

### Validation

- `bundle_id`, `schema_version`, entry `id` và policy hợp lệ.
- Không có duplicate entry ID.
- Canonical form không rỗng.
- Language code thuộc `vi`, `en`, `zh`, `ko`.
- Alias conflict phải được giải quyết bằng domain hoặc priority.
- Route được chọn phải có canonical form cho mọi hop cần thiết.
- Spoken form phải thuộc ngôn ngữ đã khai báo.

### Normalization

- Chung: Unicode NFC và whitespace normalization.
- Việt: giữ dấu; hỗ trợ alias/hyphen variants có chủ đích.
- Anh: case-fold để match nhưng bảo toàn case khi restore.
- Trung: character-level matching, không phụ thuộc whitespace.
- Hàn: Hangul NFC và alias có/không có khoảng trắng.
- Mọi normalization phải trả alignment về original span.

### Matcher

- Trie cho incremental prefix/suffix query.
- Multi-pattern matcher cho phrase scanning.
- Longest-match-first.
- Không match substring bên trong code/token khác nếu schema không cho phép.
- Kết quả match tối thiểu gồm:
  - `term_id`;
  - normalized span;
  - original span;
  - source form;
  - target canonical;
  - policy và priority.

### Profile activation

Input:

```text
domain
source_language
target_language
ASR backend/model
MT backend/route
TTS backend/model
```

Output:

- source matcher/trie;
- target matcher/trie;
- translation mappings theo từng route hop;
- ASR candidate terms theo capability;
- spoken-form mappings;
- compatibility metadata.

### Config dự kiến

Thêm `TerminologyConfig` vào `PipelineConfig`:

```yaml
terminology:
  enabled: false
  bundle_path: null
  domain: null
  matching:
    normalization: unicode_nfc
    longest_match_first: true
    case_sensitive_for_codes: true
```

### File dự kiến bị ảnh hưởng

- `src/onevoice/config.py`
- `config/default.yaml`
- `src/onevoice/terminology/*`
- `tests/test_config_registry.py`
- `tests/terminology/*`

### Test plan

- Valid/invalid schema.
- Duplicate IDs.
- Alias conflict theo domain.
- Longest-match overlap.
- Unicode composed/decomposed.
- English case preservation.
- Chinese character matching.
- Korean spacing aliases.
- Alignment từ normalized span về original span.
- Profile không nạp term ngoài domain.
- Disabled config là no-op.

### Exit criteria

- [x] Bundle hợp lệ load thành immutable profile.
- [x] Bundle sai fail trước khi model inference.
- [x] Matching deterministic trên cả bốn ngôn ngữ.
- [x] Không làm thay đổi pipeline output khi terminology tắt.
