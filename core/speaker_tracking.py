import logging
import os
import time
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

DEFAULT_SPEAKER = "SPEAKER_1"


class SpeakerTracker:
    def __init__(self):
        self.enabled = os.getenv("SPEAKER_TRACKING_ENABLED", "1") == "1"
        self.model_id = os.getenv("PYANNOTE_MODEL", "pyannote/speaker-diarization-3.1")
        self.device = os.getenv("DIARIZATION_DEVICE", os.getenv("WHISPER_DEVICE", "cpu"))
        self.hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        self._pipeline = None
        self._retry_seconds = int(os.getenv("SPEAKER_TRACKING_RETRY_SECONDS", "30"))
        self._next_retry_ts = 0.0

    def _load_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        now = time.time()
        if now < self._next_retry_ts:
            return None
        self._next_retry_ts = now + self._retry_seconds

        if not self.enabled:
            logger.info("Speaker tracking disabled via SPEAKER_TRACKING_ENABLED=0")
            return None
        self.hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        if not self.hf_token:
            logger.warning("Speaker tracking disabled: set HF_TOKEN for pyannote model access")
            return None

        try:
            from pyannote.audio import Pipeline
        except Exception as e:
            logger.warning(f"Speaker tracking unavailable (pyannote import failed): {e}")
            return None

        try:
            try:
                pipeline = Pipeline.from_pretrained(self.model_id, token=self.hf_token)
            except TypeError:
                pipeline = Pipeline.from_pretrained(self.model_id, use_auth_token=self.hf_token)

            try:
                import torch

                use_cuda = self.device == "cuda" and torch.cuda.is_available()
                pipeline.to(torch.device("cuda" if use_cuda else "cpu"))
            except Exception as e:
                logger.warning(f"Could not move diarization pipeline to requested device: {e}")

            self._pipeline = pipeline
            logger.info("Speaker diarization enabled")
        except Exception as e:
            logger.warning(f"Speaker diarization setup failed: {e}")
            self._pipeline = None
        return self._pipeline

    @staticmethod
    def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
        return max(0.0, min(a_end, b_end) - max(a_start, b_start))

    @classmethod
    def _pick_label(
        cls, seg_start: float, seg_end: float, turns: List[Tuple[float, float, str]]
    ) -> str:
        if not turns:
            return DEFAULT_SPEAKER
        best_label = DEFAULT_SPEAKER
        best_overlap = 0.0
        for t_start, t_end, label in turns:
            overlap = cls._overlap(seg_start, seg_end, t_start, t_end)
            if overlap > best_overlap:
                best_overlap = overlap
                best_label = label
        return best_label if best_overlap > 0 else DEFAULT_SPEAKER

    @staticmethod
    def _normalize_labels(labels: List[str]) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        next_id = 1
        for label in labels:
            if label not in mapping:
                mapping[label] = f"SPEAKER_{next_id}"
                next_id += 1
        return mapping

    def assign(self, file_path: str, segments: List[dict]) -> List[dict]:
        if not segments:
            return []

        pipeline = self._load_pipeline()
        if pipeline is None:
            return [{**seg, "speaker": DEFAULT_SPEAKER} for seg in segments]

        try:
            diarization = pipeline(file_path)
            turns: List[Tuple[float, float, str]] = []
            for turn, _, raw_label in diarization.itertracks(yield_label=True):
                turns.append((float(turn.start), float(turn.end), str(raw_label)))
        except Exception as e:
            logger.warning(f"Speaker diarization failed, falling back to single speaker: {e}")
            return [{**seg, "speaker": DEFAULT_SPEAKER} for seg in segments]

        raw_labels = [
            self._pick_label(float(seg["start"]), float(seg["end"]), turns)
            for seg in segments
        ]
        normalized = self._normalize_labels(raw_labels)
        return [{**seg, "speaker": normalized.get(raw, DEFAULT_SPEAKER)} for seg, raw in zip(segments, raw_labels)]
