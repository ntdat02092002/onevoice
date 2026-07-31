# P6 — ASR Terminology Implementation Report

Trạng thái: **Done — 2026-07-31**

## Outcome

P6 thêm terminology guidance vào ASR mà không giả định mọi backend có native
hotword support:

| Backend | Capability P6 |
|---|---|
| Faster-Whisper | bounded `initial_prompt` + exact post-correction |
| Moonshine | exact post-correction |
| Dolphin | exact post-correction |
| Sherpa streaming Zipformer | exact post-correction; native hotwords để P7 |
| Fake | exact post-correction để smoke test |

## Faster-Whisper prompt

Active profile compile canonical và aliases theo source language. Prompt bị giới
hạn bởi:

```yaml
terminology:
  asr:
    initial_prompt_enabled: true
    max_prompt_terms: 32
    max_prompt_tokens: 128
```

Source language `auto` không inject prompt trước khi model detect language.
Prompt là soft guidance, không phải hard constraint.

Metrics:

- `asr_prompt_term_count`;
- `asr_prompt_token_count`.

## Guarded post-correction

`TerminologyAsrCorrector` chỉ canonicalize exact alias thuộc active
domain/language:

- `wind surfing` và declared typo `winssurfing` → `windsurfing`;
- near-acoustic `wind serving` không bị sửa;
- canonical term không bị rewrite;
- term ngoài domain không bị đụng;
- không dùng fuzzy/edit-distance.

Correction chạy trước Local Agreement, semantic endpoint và MT. Khi multi-word
alias gộp thành canonical term, word timing được gộp từ first start đến last end,
confidence dùng minimum của source words. Nếu không align an toàn, timing được
bỏ và metric ghi nhận thay vì tạo timestamp giả.

Metrics:

- `asr_post_correction_count`;
- `asr_post_correction_timing_drops`.

## Verification

Test bao phủ positive alias, declared typo, near-acoustic negative, domain
isolation, timing remap, prompt budget, Faster-Whisper prompt injection và
pipeline worker ASR → commit → MT.

Full regression: **221 passed**.

P7 tiếp theo mới đánh giá native sherpa hotwords với
`modified_beam_search`; P6 không thay decoding method hiện tại.
