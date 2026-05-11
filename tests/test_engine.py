from core.engine import AudioEngine


class FakeSegment:
    def __init__(self, start, end, text):
        self.start = start
        self.end = end
        self.text = text


def make_fake_whisper(init_fail=None, runtime_fail=None):
    init_fail = init_fail or set()
    runtime_fail = runtime_fail or set()

    class FakeWhisperModel:
        init_calls = []

        def __init__(self, model, device="cpu", compute_type=None):
            self.device = device
            self.compute_type = compute_type
            self.__class__.init_calls.append((model, device, compute_type))
            if (model, device, compute_type) in init_fail:
                raise RuntimeError("libcublas.so.12 is not found or cannot be loaded")

        def transcribe(self, file_path, beam_size=5, language="ja"):
            if (self.device, self.compute_type) in runtime_fail:
                raise RuntimeError("cuBLAS failed with status CUBLAS_STATUS_NOT_SUPPORTED")
            return [FakeSegment(0.0, 1.0, "テスト")], None

    return FakeWhisperModel


def test_transcribe_rejects_cpu_device(monkeypatch):
    fake_whisper = make_fake_whisper()
    monkeypatch.setattr("core.engine.WhisperModel", fake_whisper)
    monkeypatch.setenv("WHISPER_GPU_COMPUTE_FALLBACKS", "float16")

    eng = AudioEngine()
    eng.device = "cpu"
    eng.compute_type = "int8"
    eng.speaker_tracker.assign = lambda _path, segs: [{**seg, "speaker": "SPEAKER_1"} for seg in segs]

    try:
        eng.transcribe("dummy.mp4")
        assert False, "Expected RuntimeError"
    except RuntimeError as e:
        assert "WHISPER_DEVICE must be 'cuda'" in str(e)
    assert fake_whisper.init_calls == []


def test_transcribe_runs_on_cuda_when_available(monkeypatch):
    fake_whisper = make_fake_whisper()
    monkeypatch.setattr("core.engine.WhisperModel", fake_whisper)
    monkeypatch.setenv("WHISPER_GPU_COMPUTE_FALLBACKS", "float16")

    eng = AudioEngine()
    eng.model_name = "medium"
    eng.device = "cuda"
    eng.compute_type = "int8_float16"
    eng.speaker_tracker.assign = lambda _path, segs: [{**seg, "speaker": "SPEAKER_1"} for seg in segs]

    result = eng.transcribe("dummy.mp4")

    assert result == [{"start": 0.0, "end": 1.0, "text": "テスト", "speaker": "SPEAKER_1"}]
    assert fake_whisper.init_calls == [("medium", "cuda", "int8_float16")]


def test_transcribe_retries_with_other_cuda_compute(monkeypatch):
    fake_whisper = make_fake_whisper(runtime_fail={("cuda", "int8_float16")})
    monkeypatch.setattr("core.engine.WhisperModel", fake_whisper)
    monkeypatch.setenv("WHISPER_GPU_COMPUTE_FALLBACKS", "float16")

    eng = AudioEngine()
    eng.model_name = "medium"
    eng.device = "cuda"
    eng.compute_type = "int8_float16"
    eng.speaker_tracker.assign = lambda _path, segs: [{**seg, "speaker": "SPEAKER_1"} for seg in segs]

    result = eng.transcribe("dummy.mp4")

    assert result == [{"start": 0.0, "end": 1.0, "text": "テスト", "speaker": "SPEAKER_1"}]
    assert fake_whisper.init_calls == [
        ("medium", "cuda", "int8_float16"),
        ("medium", "cuda", "float16"),
    ]


def test_transcribe_raises_when_all_cuda_runtime_modes_fail(monkeypatch):
    fake_whisper = make_fake_whisper(
        runtime_fail={("cuda", "int8_float16"), ("cuda", "float16")}
    )
    monkeypatch.setattr("core.engine.WhisperModel", fake_whisper)
    monkeypatch.setenv("WHISPER_GPU_COMPUTE_FALLBACKS", "float16")

    eng = AudioEngine()
    eng.model_name = "medium"
    eng.device = "cuda"
    eng.compute_type = "int8_float16"
    eng.speaker_tracker.assign = lambda _path, segs: [{**seg, "speaker": "SPEAKER_1"} for seg in segs]

    try:
        eng.transcribe("dummy.mp4")
        assert False, "Expected RuntimeError"
    except RuntimeError as e:
        assert "Whisper GPU runtime failed for all compute modes" in str(e)
        assert "CUBLAS_STATUS_NOT_SUPPORTED" in str(e)
    assert fake_whisper.init_calls == [
        ("medium", "cuda", "int8_float16"),
        ("medium", "cuda", "float16"),
    ]


def test_transcribe_raises_when_all_cuda_init_modes_fail(monkeypatch):
    fake_whisper = make_fake_whisper(
        init_fail={
            ("medium", "cuda", "int8_float16"),
            ("medium", "cuda", "float16"),
        }
    )
    monkeypatch.setattr("core.engine.WhisperModel", fake_whisper)
    monkeypatch.setenv("WHISPER_GPU_COMPUTE_FALLBACKS", "float16")

    eng = AudioEngine()
    eng.model_name = "medium"
    eng.device = "cuda"
    eng.compute_type = "int8_float16"
    eng.speaker_tracker.assign = lambda _path, segs: [{**seg, "speaker": "SPEAKER_1"} for seg in segs]

    try:
        eng.transcribe("dummy.mp4")
        assert False, "Expected RuntimeError"
    except RuntimeError as e:
        assert "Whisper GPU initialization failed for all compute modes" in str(e)
    assert fake_whisper.init_calls == [
        ("medium", "cuda", "int8_float16"),
        ("medium", "cuda", "float16"),
    ]
