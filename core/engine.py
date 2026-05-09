import os
import logging
from faster_whisper import WhisperModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioEngine:
    def __init__(self):
        self.model_name = os.getenv("WHISPER_MODEL", "medium")
        self.device = os.getenv("WHISPER_DEVICE", "cuda")
        self.compute_type = os.getenv("WHISPER_COMPUTE", "int8_float16")
        self.model = None

    def _load_model(self):
        if self.model is not None:
            return
        try:
            logger.info(f"Loading model {self.model_name} on {self.device}...")
            self.model = WhisperModel(self.model_name, device=self.device, compute_type=self.compute_type)
        except Exception as e:
            logger.warning(f"GPU load failed: {e}. Falling back to CPU...")
            self.model = WhisperModel(self.model_name, device="cpu", compute_type="int8")

    def transcribe(self, file_path):
        self._load_model()
        segments, _ = self.model.transcribe(file_path, beam_size=5, language="ja")
        return [{"start": s.start, "end": s.end, "text": s.text} for s in segments]