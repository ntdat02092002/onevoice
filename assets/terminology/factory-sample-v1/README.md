# Factory Sample Terminology Bundle

File dữ liệu: [terminology.yaml](terminology.yaml)

Đây là development sample để bắt đầu bổ sung thuật ngữ dần dần. Bật
**Terminology dictionary** trong Streamlit, giữ bundle path mặc định và chọn
domain tương ứng để runtime load file này.

Domain `test` có hai term kiểm thử MT preserve:

- `windsurfing`, kèm alias `wind surfing` và lỗi ASR/typing `winssurfing`;
- tên riêng `Outdoor Life`.

Khi target là tiếng Việt, subtitle vẫn giữ hai canonical form trên nhưng TTS
đọc lần lượt là `lướt ván buồm` và `ao đờ lai-ph`. Với target tiếng Anh,
`windsurfing` được synthesize thành `wind surfing`.

## Cách thêm term

Mỗi concept cần:

- `id` ổn định, dùng `snake_case`;
- ít nhất một `domain`;
- `priority`;
- `translation_policy`;
- canonical form theo các ngôn ngữ cần dùng;
- aliases nếu người nói/ASR thường tạo biến thể;
- spoken form chỉ khi cách đọc khác display text.

Ví dụ:

```yaml
- id: safety_helmet
  domain:
    - factory-safety
  priority: 70
  translation_policy: preferred_term
  forms:
    vi:
      canonical: mũ bảo hộ
      aliases:
        - nón bảo hộ
    en:
      canonical: safety helmet
      aliases:
        - hard hat
    zh:
      canonical: 安全帽
      aliases: []
    ko:
      canonical: 안전모
      aliases: []
```

## Quy tắc chỉnh sửa

- Không đổi `id` của term đã được sử dụng; thêm alias/canonical migration thay vì
  tạo lại concept.
- Không thêm cùng alias vào hai concept trong cùng domain nếu priority không giải
  quyết được nghĩa.
- Không bỏ dấu tiếng Việt để tạo alias chung.
- Product/device code phải giữ đúng canonical case.
- Thêm test case tương ứng vào
  `tests/fixtures/terminology/p0-benchmark-corpus.yaml` khi term là safety-critical,
  có overlap hoặc có spoken form đặc biệt.

Contract đầy đủ nằm tại
`docs/adr/0001-terminology-contract.md`.
