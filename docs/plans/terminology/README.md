# OneVoice Terminology Dictionary — Implementation Roadmap

Trạng thái: **In progress — P0 đến P8 đã hoàn tất**

Tài liệu nguồn: [OneVoice_Terminology_Dictionary_Final_Report.docx](../../OneVoice_Terminology_Dictionary_Final_Report.docx)

## Mục tiêu

Tích hợp terminology dictionary vào pipeline OneVoice theo hướng:

```text
Master Terminology Bundle
  -> profile theo domain/language/model
  -> ASR guidance khi backend hỗ trợ
  -> term-aware Stable Prefix
  -> MT protection + validation + canonicalization
  -> term-aware TTS chunking
  -> spoken-form normalization
```

Terminology không được triển khai như một bước `str.replace()` sau MT. Một bundle gốc phải được validate và biên dịch thành biểu diễn phù hợp cho từng tầng.

## Nguyên tắc triển khai

- `terminology.enabled: false` phải giữ nguyên hành vi hiện tại.
- Ưu tiên deterministic, không retrain model và có thể test độc lập.
- Không emit output chứa placeholder hỏng hoặc chưa được restore.
- Nội dung đã gửi TTS là rollback-free; chỉ dùng text đã committed.
- Chỉ bật capability theo backend/model thực tế.
- Bundle chỉ được đổi tại session start hoặc utterance boundary.
- Conflict resolution thống nhất: longest match, priority, declaration order.

## Roadmap

| Phase | Status | Nội dung | Kết quả |
|---|---|---|---|
| P0 | **Done** | Baseline và kiến trúc contract | ADR, sample bundle, corpus và baseline report |
| P1 | **Done** | Terminology core | Loader, validator, normalizer, matcher và domain profile |
| P2 | **Done** | MT-first | Placeholder, validator, canonical restore và OPUS pivot per-hop |
| P3 | **Done** | Streaming Zipformer độc lập | `OnlineRecognizer` cho vi/en/zh/ko; không nối terminology |
| P4 | **Done** | Streaming terminology safety | Stable Prefix và phrase chunking không cắt giữa term |
| P5 | **Done** | TTS spoken form | Tách display text khỏi synthesis text |
| P6 | **Done** | ASR terminology cho backend hiện tại | Prompt và exact-alias post-correction có kiểm soát |
| P7 | **Done** | Zipformer terminology/hotwords | Native per-stream hotwords, tokenizer validation, bounded profile và metrics |
| P8 | **Done** | Lifecycle và vận hành đơn giản | Stop/change/start, preflight compile, build identity, UI/CLI và metrics |
| P9 | Deferred | R&D tùy chọn | Constrained decoding, retrieval, LLM glossary, phoneme control |

## Tài liệu chi tiết

- [P0–P1: Baseline, data model và terminology core](p0-p1-core.md)
- [P0 baseline report](p0-baseline-report.md)
- [P1 implementation report](p1-implementation-report.md)
- [P2: MT integration](p2-mt-integration.md)
- [P2 implementation report](p2-implementation-report.md)
- [P3: Zipformer backend độc lập với terminology](p3-zipformer-asr-backend.md)
- [P3 implementation report](p3-zipformer-implementation-report.md)
- [P4–P5: Streaming safety và TTS spoken form](p3-p4-streaming-tts.md)
- [P4 implementation report](p4-implementation-report.md)
- [P5 implementation report](p5-implementation-report.md)
- [P6 implementation report](p6-implementation-report.md)
- [P6–P8: ASR terminology, lifecycle và observability](p5-p7-asr-operations.md)
- [P7 chi tiết: Zipformer terminology/hotwords](p6-sherpa-onnx-asr-backend.md)
- [P7 implementation report](p7-implementation-report.md)
- [P8 implementation report](p8-implementation-report.md)

## Thứ tự thực hiện khuyến nghị

```text
P0 -> P1 -> P2 -> P3 (Zipformer backend, no terminology)
                    -> P4 -> P5 -> P6 -> P7 -> P8
```

P2 là mốc MVP đầu tiên có giá trị sử dụng. P3 thêm backend/canary độc lập cho cả
vi/en/zh/ko, không switch toàn bộ ASR. P7 đã nối terminology profile vào native
hotwords của Sherpa mà không thay đổi backend mặc định. P9 không thuộc phạm vi
prototype.

## Definition of Done toàn chương trình

- Existing test suite vẫn pass khi terminology tắt.
- Bundle/profile sai fail sớm với lỗi có thể hành động.
- Hard terms không bị mất, lặp hoặc rò placeholder qua MT.
- Pivot route canonicalize term ở từng hop.
- Stable Prefix và TTS chunker không emit nửa term.
- UI giữ canonical display text trong khi TTS có thể dùng spoken form.
- Metrics phân biệt lỗi matching, MT validation, ASR bias và TTS normalization.
- Có rollback về bundle trước mà không restart giữa utterance.

## Ngoài phạm vi mặc định

- Fine-tune ASR, MT hoặc TTS vì terminology.
- LLM glossary/KV cache.
- Vector glossary retrieval.
- Direct phoneme IDs.
- Lexically constrained beam search.
- Dynamic bundle swap giữa một utterance đang xử lý.
