# P3–P4 — Streaming Safety và TTS Spoken Form

Trạng thái: **Planned — chưa implement**

Phụ thuộc: P1; P3 nên triển khai sau P2.

## P3 — Term-aware Stable Prefix

### Mục tiêu

Không commit suffix đang là prefix của một thuật ngữ dài hơn.

### Luồng đề xuất

```text
ASR partial history
  -> Local Agreement / LCP
  -> Hold-n hiện tại
  -> open-term-prefix check
  -> safe committed tokens + pending term suffix
```

### Quy tắc

- Chỉ kiểm tra term thuộc source language và active domain.
- Nếu suffix là full term và không còn continuation ưu tiên hơn, được commit.
- Nếu suffix là prefix của term dài hơn, giữ lại.
- Overlap dùng longest match, priority, declaration order.
- Hold được giải phóng khi:
  - term hoàn thành;
  - suffix không còn match prefix;
  - term-prefix timeout;
  - final/endpoint policy yêu cầu flush.
- Timestamped semantic endpoint hiện tại không được làm mutable tail lọt vào final.

### Integration point

- `src/onevoice/backends/commit.py`
- `src/onevoice/config.py`
- active source trie từ terminology profile.

Không thay thế Local Agreement; chỉ bổ sung một safety check sau khi đã tính stable candidate.

## P3 — Term-aware phrase chunking

### Mục tiêu

Không tạo `TtsRequest` có boundary nằm giữa canonical target term.

`PhraseTtsPolicy` hiện là nơi thực hiện sentence/phrase chunking, reservation và acknowledgement. Không cần tạo một queue stage mới trong phase đầu.

### Luồng đề xuất

```text
TranslationUpdate
  -> rematch canonical target terms
  -> protected target spans
  -> sentence/length/timeout boundary search
  -> reject boundary inside protected span
  -> reserve phrase
```

### Boundary policy

- Ưu tiên sentence boundary.
- Sau đó punctuation/comma gần giới hạn.
- Hard maximum vẫn là giới hạn an toàn, nhưng phải dịch boundary ra trước hoặc sau term.
- Nếu một term riêng lẻ dài hơn hard maximum:
  - emit nguyên term như một exceptional chunk;
  - ghi metric oversized protected span;
  - không cắt term để thỏa cấu hình.
- Chỉ chunk text đã committed theo policy hiện tại.

### Integration point

- `src/onevoice/policy.py`
- target matcher/trie từ immutable profile.

## P3 test plan

- Cross-ASR-chunk term.
- `nút dừng` và `nút dừng khẩn cấp`.
- Open prefix bị hủy bởi hypothesis tiếp theo.
- Timeout.
- Final flush.
- Endpoint cut.
- Target term nằm đúng tại max-token boundary.
- Term đi qua sentence/comma boundary.
- Term dài hơn maximum.
- Revision mới không replay acknowledged phrase.

## P3 exit criteria

- Không có committed partial term trong fixtures.
- Không có TTS request chứa một phần của protected term.
- Không phá semantic endpoint và rollback-free reservation hiện tại.
- Đo được latency tăng thêm do term hold.

## P4 — TTS spoken-form normalization

### Mục tiêu

UI giữ canonical display text trong khi TTS nhận chuỗi tối ưu cho phát âm.

### Contract tương thích ngược

Mở rộng `TtsRequest`:

```text
text: display text hiện tại
spoken_text: optional synthesis text
```

TTS backend dùng:

```text
spoken_text if present else text
```

`TtsUpdate.text` tiếp tục là display text. Nếu cần debug, bổ sung `spoken_text` riêng thay vì thay nghĩa field `text`.

### Normalization

- Exact canonical/alias match trước.
- Longest term trước để tránh thay term con.
- Spoken form theo target locale.
- Hỗ trợ:
  - acronym;
  - model/device code;
  - error code;
  - product name;
  - policy `display_preserve_speech_override`.
- Nếu không có spoken form, giữ nguyên display text.

### Integration point

- `src/onevoice/models.py`
- `src/onevoice/policy.py`
- `src/onevoice/backends/tts.py`
- `src/onevoice/terminology/compiler/tts.py`
- CLI/Streamlit event display nếu cần.

### Model lexicon

Custom phoneme/model lexicon không thuộc P4. Nó chỉ được triển khai sau khi:

- TTS model/voice đã cố định;
- tokens/phoneme inventory có checksum;
- có pronunciation acceptance test;
- bundle compatibility validation đã có.

### P4 test plan

- Display/spoken split.
- Acronym theo locale.
- Product/model code.
- Multiple/overlapping terms.
- Chinese và Korean spacing.
- Missing spoken form fallback.
- Fake TTS nhận synthesis text nhưng event giữ display text.
- Reservation prefix vẫn so sánh theo display text.

### P4 exit criteria

- Spoken form không làm thay đổi subtitle/UI translation.
- Backend synthesize đúng spoken text.
- Không làm sai logic revision, reservation và acknowledgement.
