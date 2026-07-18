# Guide: thêm model và component mới vào OneVoice

Tài liệu này mô tả cách mở rộng OneVoice mà không sửa luồng điều phối trong `RealtimePipeline`. Nguyên tắc chính là: **pipeline chỉ biết contract, adapter chịu trách nhiệm chuyển API riêng của model sang contract đó**.

## 1. Kiến trúc extension hiện tại

```text
AudioChunk
  -> AudioPreprocessor
  -> VadBackend
  -> SpeechSegment
  -> StreamingAsrBackend
  -> AsrUpdate
  -> CommitPolicy
  -> CommittedTranscript
  -> WaitKTranslationPolicy
  -> TranslationBackend
  -> TranslationUpdate
```

Các contract nằm trong `src/onevoice/protocols.py`, kiểu dữ liệu chung nằm trong `src/onevoice/models.py`, và backend tích hợp sẵn được đăng ký tại `src/onevoice/backends/__init__.py`.

| Loại backend | Method xử lý chính | Constructor factory hiện tại |
|---|---|---|
| `preprocessor` | `process(AudioChunk) -> AudioChunk` | Không truyền config |
| `vad` | `process(AudioChunk) -> list[SpeechSegment]` | `config`, `audio_config` |
| `asr` | `transcribe(SpeechSegment, language) -> AsrUpdate` | `config` |
| `commit` | `update(AsrUpdate) -> CommittedTranscript | None` | `config` |
| `translation` | `translate(TranslationRequest) -> TranslationUpdate` | `config` |

Mọi component đều phải có lifecycle:

```python
def load(self) -> None: ...
def reset(self) -> None: ...
def close(self) -> None: ...
```

- `load`: nạp model/runtime và asset nặng. Có thể được gọi trước lúc worker chạy hoặc lazy-load ở lần inference đầu.
- `reset`: xóa state của **utterance/session hiện tại**, không nên unload weights.
- `close`: giải phóng model, runtime, file handle và cache thiết bị.

## 2. Quy tắc dữ liệu bắt buộc

### Audio

`AudioChunk.samples` luôn phải là:

- `numpy.ndarray` một chiều, mono;
- `float32`, giá trị nên nằm trong `[-1.0, 1.0]`;
- 16 kHz với cấu hình mặc định;
- giữ nguyên `sequence`, `captured_at` và `end_of_stream` sau preprocess.

WebRTC callback chỉ được resample/enqueue audio. Không chạy model, VAD nặng hoặc cập nhật Streamlit UI trong callback.

### ASR partial

`AsrUpdate.text` là **toàn bộ hypothesis hiện tại của utterance**, không phải phần text delta mới xuất hiện. `LocalAgreementCommitter` cần các hypothesis đầy đủ để tìm Longest Common Prefix.

- `revision` tăng theo mỗi update của adapter.
- `is_final` phải phản ánh `SpeechSegment.is_final`.
- `started_at` dùng `time.monotonic()` ngay trước inference.
- `language` dùng mã `vi`, `en`, `zh`, `ko` hoặc kết quả language detection.
- Nên tạo `tokens` bằng `onevoice.text.tokenize_text(text, language)`.
- Validate sớm language/model/device hoặc feature không hỗ trợ ngay trong constructor, trước import dependency, download weights hay cấp phát model.

Backend ASR native-streaming có thể giữ encoder cache/RNN state bên trong, nhưng output ra contract vẫn phải là hypothesis đầy đủ. `reset()` phải xóa cache này khi sang generation/utterance không liên tục.

Nếu backend chỉ có một số tổ hợp model/ngôn ngữ, hãy khai báo capability cạnh adapter và dùng chung cho constructor lẫn UI. Không để UI là lớp validation duy nhất: CLI, YAML và code inject trực tiếp vẫn phải nhận `ValueError` rõ ràng. Ví dụ built-in Moonshine dùng `MOONSHINE_MODELS_BY_LANGUAGE`, `asr_model_options()` và `validate_asr_selection()` trong `onevoice.backends.asr`.

### Translation

MT nhận toàn bộ stable prefix tại thời điểm trigger, không nhận audio và không tự quyết định Wait-k. Adapter phải copy chính xác:

- `source_revision` từ request sang result;
- `is_final` từ request sang result;
- source/target language;
- source text dùng để tạo bản dịch.

Pipeline dùng `source_revision` để loại kết quả model chậm đã stale. Không tự thay revision trong adapter.

## 3. Thêm ASR model mới

Backend tích hợp sẵn để tham khảo:

- `MoonshineAsrBackend`: model có state streaming. `SpeechSegment` từ pipeline là snapshot tăng dần nên adapter lưu `processed_samples` và chỉ gọi native runtime với phần tail chưa thấy.
- `DolphinAsrBackend`: model nhận waveform hoàn chỉnh, adapter chuyển NumPy sang tensor và ánh xạ kết quả về `AsrUpdate`.
- `FasterWhisperBackend`: fallback stateless, decode lại snapshot hiện tại.

Nếu model mới có encoder/KV cache, không được đẩy lại toàn bộ snapshot ở mỗi update. Hãy giữ offset/state trong adapter, xử lý trường hợp snapshot final ngắn hơn do VAD trim, và xóa state trong `reset`. Nếu model stateless thì có thể decode cả snapshot như Dolphin/Faster-Whisper, nhưng phải benchmark để tránh làm đầy inference queue.

Ví dụ adapter tối giản cho một runtime giả định `MyAsrRuntime`:

```python
# src/onevoice/backends/my_asr.py
from time import monotonic

from onevoice.config import AsrConfig
from onevoice.models import AsrUpdate, SpeechSegment
from onevoice.text import tokenize_text


class MyAsrBackend:
    def __init__(self, config: AsrConfig) -> None:
        self.config = config
        self._runtime = None
        self._revision = 0

    def load(self) -> None:
        try:
            from my_asr_package import MyAsrRuntime
        except ImportError as exc:
            raise RuntimeError(
                "Install optional dependency 'my-asr-package' to use my_asr"
            ) from exc

        self._runtime = MyAsrRuntime(
            model=self.config.model,
            device=self.config.device,
            offline=self.config.offline,
        )

    def reset(self) -> None:
        self._revision = 0
        if self._runtime is not None:
            self._runtime.reset_stream()

    def close(self) -> None:
        if self._runtime is not None:
            self._runtime.close()
        self._runtime = None
        self._revision = 0

    def transcribe(
        self, segment: SpeechSegment, language: str | None
    ) -> AsrUpdate:
        if self._runtime is None:
            self.load()

        started = monotonic()
        selected_language = None if language in (None, "auto") else language
        result = self._runtime.transcribe(
            segment.samples,
            sample_rate=segment.sample_rate,
            language=selected_language,
            final=segment.is_final,
        )
        self._revision += 1
        detected_language = selected_language or result.language

        return AsrUpdate(
            text=result.text.strip(),
            language=detected_language,
            confidence=result.confidence,
            revision=self._revision,
            is_final=segment.is_final,
            started_at=started,
            tokens=tokenize_text(result.text, detected_language),
        )
```

Đăng ký backend:

```python
# src/onevoice/backends/__init__.py
from .my_asr import MyAsrBackend

registrations = (
    # ...
    ("asr", "my_asr", MyAsrBackend),
)
```

Chọn bằng YAML:

```yaml
asr:
  backend: my_asr
  model: path/to/model-or-model-id
  device: cpu
  compute_type: int8
  language: auto
  offline: true
```

Nếu model cần option chưa có, ví dụ `encoder_chunk_size`, thêm field vào `AsrConfig`. Loader config đang strict: key YAML lạ sẽ báo lỗi thay vì bị bỏ qua âm thầm.

## 4. Thêm translation model mới

Backend mặc định `OpusMtCTranslate2Backend` là ví dụ cho một adapter có router nội bộ nhưng vẫn giữ nguyên contract:

- cặp có English dùng một model OPUS-MT chuyên biệt;
- cặp Việt/Trung/Hàn với nhau đi qua English và dùng hai model;
- chỉ preload model cần cho `source_language`/`target_language` hiện tại;
- Hugging Face weights được convert INT8 một lần, sau đó CTranslate2 load cache đã convert;
- cache source dùng file thật trong project để không yêu cầu quyền tạo symlink trên Windows;
- `beam_size=1` giảm latency và mỗi request vẫn dịch toàn stable prefix do Wait-k policy quản lý ở ngoài adapter.

Danh sách route/model nằm trong `OPUS_PAIR_MODELS` tại `src/onevoice/backends/translation.py`. Khi thêm model direct mới, chỉ cần thêm cặp vào mapping; không sửa `RealtimePipeline`.

```python
# src/onevoice/backends/my_translation.py
from time import monotonic

from onevoice.config import TranslationConfig
from onevoice.models import TranslationRequest, TranslationUpdate


class MyTranslationBackend:
    def __init__(self, config: TranslationConfig) -> None:
        self.config = config
        self._model = None

    def load(self) -> None:
        from my_mt_package import Translator

        self._model = Translator(
            self.config.model,
            device=self.config.device,
            local_files_only=self.config.offline,
        )

    def reset(self) -> None:
        # Xóa KV cache/history nếu model có state theo hội thoại.
        if self._model is not None:
            self._model.reset_context()

    def close(self) -> None:
        self._model = None

    def translate(self, request: TranslationRequest) -> TranslationUpdate:
        if self._model is None:
            self.load()

        started = monotonic()
        text = self._model.translate(
            request.text,
            source=request.source_language,
            target=request.target_language,
        )
        return TranslationUpdate(
            text=text.strip(),
            source_text=request.text,
            source_language=request.source_language,
            target_language=request.target_language,
            source_revision=request.source_revision,
            is_final=request.is_final,
            started_at=started,
        )
```

Sau đó đăng ký `("translation", "my_mt", MyTranslationBackend)` và đổi `translation.backend` trong YAML/UI.

Nếu dùng LLM SimulMT có KV cache:

- state chỉ thuộc adapter MT;
- append token/context mới, không rebuild prompt nếu runtime hỗ trợ;
- `reset()` phải xóa history khi bắt đầu conversation mới hoặc pipeline overload;
- không chuyển logic Wait-k vào adapter; policy và model vẫn phải thay độc lập được.

## 5. Thêm audio preprocessor

Preprocessor phù hợp cho noise suppression, automatic gain control, high-pass filter hoặc echo cancellation. Nó không nên tự cắt utterance; việc đó thuộc VAD.

```python
# src/onevoice/backends/rnnoise.py
import numpy as np

from onevoice.models import AudioChunk


class RnNoisePreprocessor:
    def __init__(self) -> None:
        self._runtime = None

    def load(self) -> None:
        from my_rnnoise_binding import RnNoise

        self._runtime = RnNoise(sample_rate=16_000)

    def reset(self) -> None:
        if self._runtime is not None:
            self._runtime.reset()

    def close(self) -> None:
        self._runtime = None

    def process(self, chunk: AudioChunk) -> AudioChunk:
        if self._runtime is None:
            self.load()

        cleaned = self._runtime.process(chunk.samples).astype(np.float32)
        return AudioChunk(
            samples=np.clip(cleaned, -1.0, 1.0),
            sample_rate=chunk.sample_rate,
            sequence=chunk.sequence,
            captured_at=chunk.captured_at,
            end_of_stream=chunk.end_of_stream,
        )
```

Hiện `RealtimePipeline` tạo `passthrough` preprocessor mặc định. Có hai cách dùng preprocessor mới:

```python
# Cách 1: inject trực tiếp, không sửa config
pipeline = RealtimePipeline(config, preprocessor=RnNoisePreprocessor())
```

```python
# Cách 2: nếu muốn chọn bằng YAML/UI
# 1. Thêm PreprocessConfig vào PipelineConfig.
# 2. Đăng ký ("preprocessor", "rnnoise", RnNoisePreprocessor).
# 3. Đổi factory trong RealtimePipeline từ "passthrough" sang config backend.
```

Khi thêm noise suppression, test bắt buộc phải kiểm tra output vẫn mono, cùng sample rate, không có `NaN/Inf`, không clipping bất thường và không làm mất EOS.

## 6. Thêm VAD backend

VAD adapter chịu trách nhiệm gom frame thành utterance và phát snapshot:

- partial `SpeechSegment(is_final=False)` chứa toàn bộ audio utterance hiện tại;
- final `SpeechSegment(is_final=True)` khi endpoint, max duration hoặc `flush()`;
- `flush()` phải trả final segment còn lại khi người dùng dừng mic/file;
- `request_endpoint()` chỉ đặt một signal thread-safe; lần `process()` kế tiếp trên audio worker mới đóng utterance và trả final segment;
- sau final phải sẵn sàng nhận utterance tiếp theo;
- giữ speech padding để không cắt mất âm đầu/cuối.

Đăng ký VAD mới với factory nhận đúng hai keyword argument:

```python
class MyVadBackend:
    def __init__(self, config: VadConfig, audio_config: AudioConfig) -> None:
        ...

    def request_endpoint(self) -> None:
        # Không sửa buffer trực tiếp từ ASR worker. Dùng threading.Event
        # và consume event bên trong process().
        self._endpoint_requested.set()
```

Không gọi ASR trực tiếp từ VAD. Pipeline sẽ đưa `SpeechSegment` sang ASR queue và tự xử lý backpressure. `request_endpoint()` được dùng bởi semantic endpoint cấu hình qua `vad.semantic_endpoint_enabled` và `vad.semantic_endpoint_sentences`.

## 7. Thêm Stable Prefix / commit policy

Commit policy chỉ xử lý text hypothesis, không biết model ASR cụ thể. Một policy mới phải:

- không emit lại cùng stable text nếu không có thay đổi;
- giữ committed partial đơn điệu; final được phép thay thế utterance hiện tại nếu ASR sửa prefix, nhưng không được làm mất suffix draft đã hiển thị;
- flush khi `AsrUpdate.is_final=True`;
- tokenize đúng ngôn ngữ, đặc biệt tiếng Trung không dựa vào khoảng trắng;
- reset toàn bộ history khi `reset()`.

Constructor hiện nhận `CommitConfig`. Đăng ký bằng kind `commit`, sau đó đổi `commit.backend`.

## 8. Dependency, model asset và offline mode

Không import dependency nặng ở đầu module nếu dependency đó là optional. Import trong `load()` và trả lỗi có hướng dẫn cài đặt rõ ràng.

Thêm dependency vào group riêng trong `pyproject.toml`, ví dụ:

```toml
[project.optional-dependencies]
gipformer = ["sherpa-onnx>=...", "onnxruntime>=..."]
rnnoise = ["my-rnnoise-binding>=..."]
```

Quy ước offline:

- `offline: false`: lần đầu được phép tải asset, inference sau đó vẫn local;
- `offline: true`: chỉ dùng local path/cache và fail sớm nếu thiếu model;
- không tải model trong unit test hoặc CI mặc định;
- ghi model ID/version, license, kích thước và ngôn ngữ hỗ trợ vào README/model manifest.

## 9. Test bắt buộc cho backend mới

### Unit/contract test

- `load -> process/inference -> reset -> process/inference -> close` không lỗi;
- output đúng dataclass và invariant;
- final/EOS luôn được flush;
- revision tăng đúng và không bị mất qua adapter;
- missing dependency/offline missing asset có lỗi dễ hiểu;
- mock runtime để test không tải weights.

### Pipeline integration test

Inject backend vào `RealtimePipeline` hoặc đăng ký bằng registry, dùng fake component cho các tầng còn lại:

```python
pipeline = RealtimePipeline(
    config,
    asr=MyAsrBackend(test_config),
    translator=FakeTranslationBackend(config.translation),
)
```

Xác nhận event tối thiểu:

```text
speech_start
asr_partial
asr_committed
asr_final
translation_final
```

### Real-model smoke test

Đánh dấu bằng `@pytest.mark.model` để không chạy mặc định. Benchmark ít nhất:

- TTFT/latency và real-time factor;
- RAM peak;
- reset giữa hai utterance;
- bốn ngôn ngữ hoặc đúng phạm vi model công bố;
- silence, audio quá ngắn, audio dài tối đa, noise và EOS giữa chừng.

Chạy kiểm tra:

```powershell
.\venv\Scripts\python.exe -m pytest
.\venv\Scripts\python.exe -m pytest -m model
.\venv\Scripts\python.exe -m pip check
```

## 10. Checklist trước khi merge

- [ ] Adapter triển khai đủ `load/reset/close` và protocol tương ứng.
- [ ] Không sửa `RealtimePipeline` nếu chỉ thay API model.
- [ ] Constructor khớp keyword arguments mà registry factory đang truyền.
- [ ] Không inference trong WebRTC callback hoặc Streamlit UI thread.
- [ ] Audio output vẫn mono float32 16 kHz và giữ metadata/EOS.
- [ ] ASR partial trả full hypothesis; final/translation giữ đúng revision.
- [ ] Backend được đăng ký bằng một tên ổn định và có config mẫu.
- [ ] Dependency optional không làm backend khác import fail.
- [ ] License/model version/offline behavior được ghi lại.
- [ ] Unit test không tải model; real model test có marker riêng.
- [ ] `pytest`, CLI fake pipeline và Streamlit health check đều pass.

## 11. Khi thêm component hoàn toàn mới

Ví dụ TTS không nên được nhét vào `TranslationBackend`. Hãy mở rộng theo đúng tầng:

1. Thêm request/result dataclass mới vào `models.py`.
2. Thêm protocol mới vào `protocols.py`.
3. Thêm registry kind và adapter.
4. Thêm bounded queue/worker mới sau MT.
5. Thêm `EventType` và UI consumer.
6. Viết fake backend trước, sau đó mới nối model thật.

Cách này giữ ranh giới module rõ ràng và cho phép tắt/thay TTS mà không làm thay đổi ASR hoặc MT.
