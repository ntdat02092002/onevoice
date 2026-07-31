# P0 Baseline Report

- Status: **Complete**
- Date: 2026-07-30
- Repository revision: `c8da316`
- Python: `3.11.0`
- Platform: Windows, local CPU profile

## Scope

P0 chốt contract, tạo sample bundle/corpus và ghi baseline trước khi terminology
runtime được implement.

Không có loader, matcher, placeholder, ASR hotword hay spoken-form runtime tại
baseline này.

## Deliverables

- [ADR-0001](../../adr/0001-terminology-contract.md)
- [Sample terminology bundle](../../../assets/terminology/factory-sample-v1/terminology.yaml)
- [P0 benchmark corpus](../../../tests/fixtures/terminology/p0-benchmark-corpus.yaml)

## Unit-test baseline

Command:

```powershell
.\venv\Scripts\python.exe -m pytest
```

Result:

```text
135 passed in 0.98s
```

Đây là regression baseline bắt buộc cho P1 trở đi.

## Cached real-model baseline

Các OPUS CTranslate2 pair có sẵn cục bộ:

- `en -> vi`
- `en -> zh`

Model load time không được tính vào inference latency bên dưới. Mỗi số chỉ là một
sample CPU local; corpus quá nhỏ để báo p95.

### English to Vietnamese

| Source | Baseline output | Latency |
|---|---|---:|
| Press the emergency stop button on the M5Stack controller. | Nhấn nút dừng khẩn cấp trên bộ điều khiển M5Stack. | 62 ms |
| Check the EtherCAT encoder and PLC S7-1200. | Kiểm tra bộ mã hóa EtherCAT và PLC S7-1200. | 47 ms |
| The conveyor must stop immediately. | Băng chuyền phải dừng lại ngay lập tức. | 32 ms |

Canonical terminology hits: `5/6` (`83.3%`).

Mismatch:

- expected canonical `băng tải`;
- baseline model produced alias `băng chuyền`.

Latency snapshot: mean `47 ms`, range `32–62 ms`.

### English to Chinese

| Source | Baseline output | Latency |
|---|---|---:|
| Press the emergency stop button on the M5Stack controller. | 按下M5Stack控制器上的紧急停机按钮。 | 62 ms |
| Check the EtherCAT encoder and PLC S7-1200. | 检查以太CAT编码器和PLC S7-1200。 | 63 ms |
| The conveyor must stop immediately. | 传送器必须立即停止。 | 31 ms |

Canonical terminology hits: `3/6` (`50.0%`).

Mismatches:

- expected `紧急停止按钮`, produced `紧急停机按钮`;
- expected preserved `EtherCAT`, produced `以太CAT`;
- expected `传送带`, produced `传送器`.

Latency snapshot: mean `52 ms`, range `31–63 ms`.

## Metric availability

| Metric | P0 state |
|---|---|
| MT terminology accuracy | Recorded for cached `en->vi` and `en->zh` smoke corpus |
| Placeholder survival | Not applicable; placeholder runtime does not exist |
| Pivot preservation | Corpus defined; required model routes are not fully cached |
| ASR term recall/precision | Not measured; no versioned terminology audio corpus yet |
| False term insertions/minute | Not measured; requires positive/negative audio |
| First-output/commit latency | Existing pipeline tests pass; real ASR benchmark deferred to model-specific phase |
| TTS pronunciation acceptance | Not measured; requires review audio and spoken-form runtime |

Các giá trị chưa đo không được thay bằng số giả. P1/P2 dùng text corpus hiện tại;
P5/P6 bổ sung audio và thiết bị benchmark khi model/backend được pin.

## P0 exit checklist

- [x] Master format, schema version và policies được chốt.
- [x] Conflict và MT fallback policy được chốt.
- [x] Sample bundle có canonical/alias/domain/spoken form.
- [x] Corpus có Việt, Anh, Trung, Hàn.
- [x] Corpus có exact, overlap, cross-chunk, alias, negative, pivot và profile conflict.
- [x] Existing test suite pass.
- [x] Baseline MT snapshot được ghi với model cache sẵn có.
- [x] Các metric chưa thể đo có owner phase rõ ràng.

## P1 entry conditions

P1 có thể bắt đầu khi:

- sample bundle vẫn được xem là development data, không phải production glossary;
- mọi schema thay đổi so với ADR phải tạo ADR superseding hoặc migration note;
- disabled terminology path tiếp tục lấy `135 passed` làm regression baseline.
