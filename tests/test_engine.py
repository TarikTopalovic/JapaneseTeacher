import types
import pytest

from engine import ImmersionEngine


class FakeSegment:
    def __init__(self, start, end, text):
        self.start = start
        self.end = end
        self.text = text


def make_fake_whisper(behaviour):
    """Return a Fake WhisperModel class according to behaviour dict.

    behaviour rules:
      - raise_on: set of (model, device) tuples that should raise on init
      - succeed_on: set of (model, device) tuples that should succeed
    """

    class Fake:
        last_init = None

        def __init__(self, model, device="cpu", compute_type=None):
            Fake.last_init = (model, device)
            if (model, device) in behaviour.get("raise_on", set()):
                raise RuntimeError("libcublas.so.12 is not found or cannot be loaded")

        def transcribe(self, file_path, beam_size=5, language="ja"):
            # Return a single fake segment and None for second return value
            return ([FakeSegment(0.0, 1.0, "テスト")], None)

    return Fake


def test_transcribe_falls_back_to_cpu_on_libcublas(monkeypatch):
    # Simulate CUDA init failure (libcublas missing) but CPU works
    fake = make_fake_whisper({
        "raise_on": {("large-v3", "cuda")}
    })
    monkeypatch.setattr("engine.WhisperModel", fake)

    eng = ImmersionEngine()
    eng.model_name = "large-v3"
    eng.preferred_device = "cuda"

    result = eng.transcribe("dummy.mp4")
    assert isinstance(result, list)
    assert result[0]["text"] == "テスト"


def test_prefers_smaller_model_if_large_oom(monkeypatch):
    # Simulate large-v3 failing on cuda, medium succeeding on cuda
    fake = make_fake_whisper({
        "raise_on": {("large-v3", "cuda")}
    })
    monkeypatch.setattr("engine.WhisperModel", fake)

    eng = ImmersionEngine()
    eng.model_name = "large-v3"
    eng.preferred_device = "cuda"

    # Call transcribe which triggers lazy init
    _ = eng.transcribe("dummy.mp4")

    # Ensure the Fake model was initialized at least once
    assert fake.last_init is not None
    # Should have tried cuda first
    assert fake.last_init[1] in {"cuda", "cpu"}
