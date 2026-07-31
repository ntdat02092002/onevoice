# P8 — Simplified terminology lifecycle: Implementation Report

Trạng thái: **Done**

## Quyết định lifecycle

P8 không triển khai hot-swap giữa utterance và không giữ rollback stack trong
process. Lifecycle cố ý đơn giản:

```text
stop pipeline
  -> chọn bundle/domain/language route
  -> preflight validate + compile
  -> start pipeline với immutable build
  -> controls bị khóa đến khi stop
```

Nếu bundle mới lỗi, nút start bị khóa trên Streamlit và pipeline mới không được
tạo. Muốn quay lại bundle trước, chọn lại file/domain cũ sau khi stop.

## Preflight build

`prepare_terminology_bundle()` thực hiện:

- đọc và validate schema/bundle conflicts;
- resolve source language và MT route thực tế;
- compile profile thành ASR terms, MT hop bindings và TTS spoken forms;
- cache immutable profile trong `TerminologyManager`;
- tạo SHA-256 build identity;
- chỉ lưu/display bundle path tương đối với workspace.

Với source `auto`, compiler build trước profile cho mọi source language sản phẩm
khác target. Thiếu language coverage làm preflight fail thay vì lỗi muộn khi
audio đang chạy.

## Build information

Mỗi pipeline giữ một `TerminologyBuildInfo` bất biến:

- bundle ID và schema version;
- description, selected domain và relative bundle path;
- SHA-256;
- compiled language routes;
- profile/entry count;
- ASR term count;
- MT binding count;
- TTS spoken-form count.

Streamlit hiển thị `Terminology preflight` trước start và
`Active terminology` sau start.

## UI và CLI

- Source/target/backend/model/bundle/domain và các pipeline controls đều bị khóa
  khi pipeline tồn tại.
- Start/Stop gọi rerun ngay để trạng thái khóa được phản ánh lập tức.
- Start bị disable nếu ASR selection hoặc terminology preflight invalid.
- CLI hỗ trợ `--terminology-bundle` và `--terminology-domain`; pipeline dùng cùng
  compiler/validation path với Streamlit.

## Metrics

- `terminology_build_count`
- `terminology_profile_count`
- `terminology_compiled_entry_count`

Các metric runtime P2–P7 vẫn giữ nguyên cho matching, placeholder validation,
hotwords, commit và spoken-form normalization.

## Verification

Tests bao phủ:

- compiled profile cache trả lại cùng immutable object;
- relative build identity và SHA-256;
- concrete/auto source compilation;
- pivot MT route compilation;
- pipeline exposes build info/metrics;
- CLI terminology options.

Hot-swap, atomic publish giữa utterance, rollback stack và persistent bundle
registry được bỏ khỏi phạm vi theo quyết định simplified lifecycle.
