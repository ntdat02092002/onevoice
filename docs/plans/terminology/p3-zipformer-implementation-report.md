# P3 — Multi-language Streaming Zipformer Implementation Report

Trạng thái: **Done — 2026-07-30**

## Outcome

Backend `asr/sherpa_onnx` hiện dùng sherpa-onnx `OnlineRecognizer` thật cho cả bốn
ngôn ngữ OneVoice: Vietnamese, English, Chinese và Korean.

Implementation trước dùng Vietnamese offline package đã được thay thế. Offline
package không còn xuất hiện trong model selector hoặc auto routing.

## Architecture

```text
growing SpeechSegment snapshot
  -> keep one OnlineStream per utterance
  -> feed unseen waveform tail only
  -> decode while ready
  -> expose partial AsrUpdate
  -> final padding + input_finished
  -> final AsrUpdate + drop stream
```

Recognizer không bật sherpa endpoint detection và không nhận `hotwords_file`.
Pipeline VAD/utterance generation tiếp tục là source of truth.

## Runtime configuration

```yaml
asr:
  backend: sherpa_onnx
  model: auto
  language: vi
  sherpa:
    recognizer_mode: online_transducer
    provider: cpu
    num_threads: 2
    decoding_method: greedy_search
    max_active_paths: 4
    final_padding_ms: 500
    cache_dir: .cache/onevoice/asr
```

`model: auto` map theo language. App model selector hiển thị đúng một default
streaming model tương thích với language đang chọn.

## Real model verification

Tất cả model đã được tải vào project-relative cache và load với sherpa-onnx
`1.13.4`. Audio được feed theo growing snapshot 0,5 giây.

| Language | Updates | Audio duration | Total decode compute | Result |
|---|---:|---:|---:|---|
| `vi` | 8 | 3.74 s | ~139 ms | Full Vietnamese reference transcript |
| `en` | 14 | 6.62 s | ~361 ms | Full English reference transcript |
| `zh` | 12 | 5.61 s | ~515 ms | Valid Chinese reference transcript |
| `ko` | 8 | 3.53 s | ~242 ms | Valid Korean reference transcript |

Vietnamese cần 500 ms final zero padding để emit hai token cuối; padding này được
áp dụng chung và có config bound `0..2000 ms`.

End-to-end CLI smoke test:

- WebRTC VAD;
- Vietnamese streaming Zipformer;
- Local Agreement;
- fake MT;
- có `asr_final` và `translation_final`;
- không có error;
- final-stage ASR latency trong pipeline test khoảng 94 ms.

## Verification

- Full suite: **181 passed**.
- Catalog tests cover exactly `vi/en/zh/ko`.
- Lifecycle tests cover unseen-tail feed, shrinking snapshot guard, final flush,
  reset/close và offline cache miss.
- Recognizer construction test assert không truyền `hotwords_file`.
- `git diff --check`: pass.
- Streamlit headless smoke test: HTTP 200.

## License

Vietnamese `hynt/Zipformer-30M-RNNT-Streaming-6000h` dùng
`CC-BY-NC-ND-4.0`; đây là research canary và chưa được duyệt production/commercial.
Các official release model còn lại giữ license/provenance riêng của upstream và
cần compatibility manifest trước production rollout.

## Thử trên app

```powershell
.\venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```

Chọn source `vi`, `en`, `zh` hoặc `ko`, sau đó chọn ASR backend `sherpa_onnx`.
Model selector tự đổi theo language. Các model đã có trong cache máy này, nên có
thể bật offline. Tắt terminology dictionary để benchmark ASR thuần.
