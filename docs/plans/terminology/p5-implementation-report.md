# P5 — TTS Spoken-form Implementation Report

Trạng thái: **Done — 2026-07-31**

## Outcome

P5 tách display text khỏi synthesis text mà không thay đổi subtitle/UI:

```text
TranslationUpdate.text
  -> P4 term-safe phrase chunk
  -> TtsRequest.text          = canonical display phrase
  -> TtsRequest.spoken_text   = locale-specific synthesis phrase, optional
  -> TTS backend.generate(spoken_text or text)
  -> TtsUpdate.text           = canonical display phrase
  -> TtsUpdate.spoken_text    = synthesis phrase, optional/debug
```

Streamlit tiếp tục dùng `TtsUpdate.text`, vì vậy spoken-form không rò vào
translated transcript hoặc lịch sử hiển thị.

## Normalization

`TerminologyTtsNormalizer` dùng immutable target matcher của active profile:

- match canonical và alias;
- longest match/priority/declaration order dùng chung terminology core;
- original-span alignment giữ nguyên punctuation và spacing xung quanh;
- chỉ thay term có `tts.<language>.spoken_form`;
- term không có spoken form giữ nguyên display text.

P4 đảm bảo phrase boundary không nằm giữa protected term trước khi P5 normalize.

## Backend contract

Cả `SherpaOnnxTtsBackend` và `FakeTtsBackend` synthesize
`request.synthesis_text`, tương đương:

```text
request.spoken_text if present else request.text
```

Output vẫn báo `text=request.text`. Field `spoken_text` riêng phục vụ debug và
verification.

## Configuration and metrics

```yaml
terminology:
  tts:
    strategy: spoken_form
```

Metrics:

- `tts_spoken_form_requests`;
- `tts_spoken_form_substitutions`.

## Verification

Test bao phủ:

- display/spoken split;
- Vietnamese acronym/product/model code;
- English, Chinese và Korean spacing;
- multiple terms;
- missing spoken-form fallback;
- Fake TTS và sherpa native generator nhận synthesis text;
- reservation/acknowledgement tiếp tục so sánh display prefix.

Full regression: **212 passed**. Pipeline smoke xác nhận display
`[vi] M5Stack` trong khi synthesis dùng `[vi] em năm stack`.

Sau P5 có thể test app end-to-end với terminology profile, MT và TTS thật.
