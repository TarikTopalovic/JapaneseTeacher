import os
import logging
from faster_whisper import WhisperModel
from sudachipy import dictionary, tokenizer
import pykakasi
import urllib.parse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImmersionEngine:
    def __init__(self):
        """
        Optimized for RTX 3070 (8GB VRAM) and 32GB System RAM.
        Uses 'medium' model with int8_float16 to prevent OOM.
        """
        # Configuration from environment or defaults
        self.model_name = os.getenv("WHISPER_MODEL", "medium")
        self.preferred_device = os.getenv("WHISPER_DEVICE", "cuda")
        self.compute_type = os.getenv("WHISPER_COMPUTE", "int8_float16")

        self.st_model = None
        self.tokenizer_obj = dictionary.Dictionary().create()
        self.kks = pykakasi.kakasi()

    def _init_model(self):
        """Initializes the Whisper model with GPU-to-CPU fallback logic."""
        if self.st_model is not None:
            return

        try:
            logger.info(f"Attempting GPU load: {self.model_name} on {self.preferred_device}")
            self.st_model = WhisperModel(
                self.model/name, 
                device=self.preferred_device, 
                compute_type=self.compute_type
            )
            logger.info("GPU load successful.")
        except Exception as e:
            logger.warning(f"GPU failed or Out of Memory ({e}). Switching to System RAM (CPU)...")
            try:
                # Fallback to CPU with int8 to maximize 32GB RAM efficiency
                self.st_model = WhisperModel(
                    self.model_name, 
                    device="cpu", 
                    compute_type="int8"
                )
                logger.info("CPU load successful.")
            except Exception as e2:
                logger.error(f"Critical failure: {e2}")
                raise e2

    def transcribe(self, file_path):
        """Transcribes audio using faster-whisper with VRAM monitoring."""
        if self.st_model is None:
            self._init_model()

        try:
            segments, _ = self.st_model.transcribe(file_path, beam_size=5, language="ja")
            return [{"start": s.start, "end": s.end, "text": s.text} for s in segments]
        except Exception as e:
            # Check if error is related to CUDA/VRAM
            error_msg = str(e).lower()
            if "out of memory" in error_msg or "cuda" in error_msg:
                logger.warning("VRAM peak detected. Forcing immediate move to System RAM...")
                # Reset model to trigger fallback in next attempt
                self.st_model = None
                self.preferred_device = "cpu" 
                # Retry once
                return self.transcribe_fallback(file_path)
            else:
                raise e

    def transcribe_fallback(self, file_path):
        """Fallback method specifically for CPU processing."""
        self.preferred_device = "cpu"
        self.st_model = WhisperModel(file_path, device="cpu")
        return self.st_model.transcribe(file_path)

    def analyze_text(self, text):
        """
        Analyzes the text for dictionary lookups.
        Instead of a heavy local DB, we generate deep-links to search engines.
        """
        words = text.split()
        results = []

        for word in words:
            # Clean word (remove punctuation)
            clean_word = "".join(char for char in word if char.isalnum())
            if not clean_word:
                continue

            # Create a Google Search Link for the word
            # This is much lighter and more accurate than a broken local dictionary
            search_url = f"https://www.google.com/search?q={clean_word}+meaning"
            
            # Create a Jisho link for Japanese learners
            jisho_url = f"https://jisho.org/search/{clean_word}"

            results.append({
                "word": clean_word,
                "google_link": search_url,
                "jisho_link": jisho_url
            })
            
        return results