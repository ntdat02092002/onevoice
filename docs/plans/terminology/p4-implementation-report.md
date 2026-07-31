# P4 — Streaming Terminology Safety Implementation Report

Trạng thái: **Done — 2026-07-31**

## Outcome

P4 nối immutable terminology profile vào hai điểm streaming có thể phát ra
một phần term:

1. Local Agreement giữ suffix đang là prefix mở của source term.
2. Phrase TTS dời chunk boundary ra khỏi target term span.

Khi `terminology.enabled: false`, cả hai component giữ nguyên hành vi trước P4.

## Term-aware Stable Prefix

`LocalAgreementCommitter` lazy-activate source trie theo ngôn ngữ ASR hiện tại.
Sau Local Agreement và `hold_tokens`, committer kiểm tra suffix:

- prefix mở được giữ lại;
- term được giải phóng khi hypothesis hoàn thành hoặc đổi hướng;
- final/endpoint vẫn flush theo final policy;
- `commit.term_prefix_timeout_ms` giới hạn thời gian giữ, mặc định 1500 ms.

Metrics:

- `term_prefix_hold_events`;
- `term_prefix_hold_ms`;
- `term_prefix_timeout_flushes`.

## Term-aware phrase chunking

`PhraseTtsPolicy` rematch target terms trên stable translated tokens bằng target
trie. Nếu boundary nằm trong protected span:

- ưu tiên dời về trước term khi chunk trước vẫn đạt minimum;
- nếu không, dời ra sau term;
- term dài hơn `max_chunk_tokens` được emit nguyên vẹn như exceptional chunk.

Reservation, synthesis acknowledgement và completed-stream tombstone tiếp tục
dùng display-token prefix như trước.

Metrics:

- `tts_term_boundary_shifts`;
- `tts_oversized_protected_spans`.

## Verification

Test bao phủ:

- source term mở qua nhiều ASR revision;
- hypothesis hủy term prefix;
- final flush và timeout flush;
- TTS hard boundary nằm giữa multi-token term;
- protected term dài hơn hard maximum;
- toàn bộ regression khi terminology tắt.

P5 tiếp theo sẽ thêm spoken-form normalization, tách display text khỏi synthesis
text; P4 chưa thay đổi contract `TtsRequest`.
