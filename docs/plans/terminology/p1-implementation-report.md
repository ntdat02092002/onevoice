# P1 Implementation Report — Terminology Core

- Status: **Complete**
- Date: 2026-07-30
- Scope: data model, validation, normalization, matching, profile compilation và config

## Outcome

P1 đã cung cấp terminology core độc lập với model backend. Pipeline
ASR/MT/TTS chưa consume profile trong phase này; `terminology.enabled: false` là
mặc định và giữ nguyên behavior hiện tại.

## Implemented modules

| Module | Kết quả |
|---|---|
| `schema.py` | Immutable bundle/entry/form models và translation policy enum |
| `loader.py` | Strict YAML schema v1 loader và actionable validation errors |
| `normalizer.py` | Unicode NFC, whitespace/hyphen normalization và original-span alignment |
| `matcher.py` | Character trie matcher, overlap resolution và token prefix trie |
| `profile.py` | Immutable domain/language/route profile |
| `manager.py` | Bundle loading và profile activation API |
| `compiler/asr.py` | In-memory ASR term/score artifacts |
| `compiler/mt.py` | Canonical mapping theo từng MT hop |
| `compiler/tts.py` | Display/spoken-form artifacts |
| `errors.py` | Bundle và profile error taxonomy |

Public package: `onevoice.terminology`.

## Validation contract

Loader hiện reject:

- schema version không hỗ trợ;
- unknown field;
- bundle không có entries;
- invalid/duplicate snake-case ID;
- invalid domain, priority hoặc translation policy;
- canonical/alias rỗng hoặc trùng sau normalization;
- language ngoài `vi`, `en`, `zh`, `ko`;
- invalid/non-positive ASR boost;
- TTS language không có language form tương ứng;
- alias conflict không được domain hoặc priority giải quyết.

Profile activation reject:

- source/target/route language không hỗ trợ;
- route không bắt đầu/kết thúc đúng language;
- active term thiếu canonical form ở một route hop;
- không có domain và bundle cũng không có default domain.

## Matching behavior

- Unicode NFC giữ dấu tiếng Việt.
- Decomposed Hangul được compose.
- Whitespace được collapse.
- Unicode hyphen variants được chuẩn hóa thành `-`.
- Code-like form có thể case-sensitive.
- Chinese matching không phụ thuộc word boundary/spacing.
- Korean canonical term match được trước attached particle.
- Overlap resolution:

```text
longest match -> priority -> declaration order
```

- Normalized span được map lại original character span.
- Prefix trie phát hiện term vừa là full term vừa là prefix term dài hơn, ví dụ
  `nút dừng` và `nút dừng khẩn cấp`.

## Profile artifacts

Ví dụ API:

```python
from onevoice.terminology import TerminologyManager

manager = TerminologyManager.from_path(
    "assets/terminology/factory-sample-v1/terminology.yaml"
)
profile = manager.activate(
    domain="factory-maintenance",
    source_language="vi",
    target_language="ko",
    mt_route=("vi", "en", "ko"),
    asr_model_id="moonshine-vi",
    tts_model_id="vits-ko",
)
```

Profile chứa:

- filtered immutable entries;
- source/target matcher;
- source/target prefix trie;
- ASR canonical/alias terms và boost;
- MT canonical mapping `vi->en` và `en->ko`;
- target spoken forms.

## Configuration

`PipelineConfig` có thêm:

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

Config loader deep-merge nested matching settings và reject:

- enabled nhưng thiếu bundle path;
- normalization ngoài `unicode_nfc`;
- `longest_match_first: false`, vì schema v1 chỉ hỗ trợ deterministic policy đã
  chốt.

## Tests

Targeted coverage:

- valid/invalid schema;
- duplicate IDs và alias conflict;
- immutable data;
- Unicode composed/decomposed alignment;
- Vietnamese overlap;
- English/code case behavior;
- Chinese continuous characters;
- Korean particle/Hangul normalization;
- source prefix trie;
- domain filtering;
- pivot compiler;
- missing route coverage;
- same-language route;
- nested config deep merge/no-op default.

Full result:

```text
156 passed in 1.00s
```

P0 baseline là `135 passed`; P1 thêm coverage mà không làm regression test cũ.

## Exit checklist

- [x] Bundle hợp lệ load thành immutable data/profile.
- [x] Bundle sai fail trước model inference.
- [x] Matching deterministic trên Việt, Anh, Trung và Hàn.
- [x] Normalized match map được về original span.
- [x] Domain profile loại term không liên quan.
- [x] Direct, pivot và same-language route compile đúng.
- [x] Disabled config giữ pipeline behavior hiện tại.
- [x] Sample bundle load thành công.
- [x] Full regression suite pass.

## Deferred to later phases

- P2: MT placeholder/protection/validator.
- P3: inject source trie vào committer và target matcher vào chunker.
- P4: apply TTS spoken form.
- P5/P6: compile/inject backend-specific ASR prompt/hotwords.
- P7: atomic profile activation/hot-swap, UI/CLI và operational metrics.
