import os
import logging
import threading
from core.speaker_tracking import SpeakerTracker
try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioEngine:
    AVAILABLE_MODELS = ('tiny', 'base', 'small', 'medium', 'large-v3')

    def __init__(self):
        self.model_name = os.getenv('WHISPER_MODEL', 'medium')
        self.device = os.getenv('WHISPER_DEVICE', 'cuda')
        self.compute_type = os.getenv('WHISPER_COMPUTE', 'float16')
        fallback_raw = os.getenv('WHISPER_GPU_COMPUTE_FALLBACKS', 'int8_float16')
        self.gpu_compute_fallbacks = [c.strip() for c in fallback_raw.split(',') if c.strip()]
        self.allow_cpu_fallback = os.getenv('WHISPER_ALLOW_CPU_FALLBACK', '0') == '1'
        self.model = None
        self.speaker_tracker = SpeakerTracker()
        self._lock = threading.RLock()

    def runtime_settings(self):
        return {
            'model_name': self.model_name,
            'compute_type': self.compute_type,
            'allow_cpu_fallback': self.allow_cpu_fallback,
            'available_models': list(self.AVAILABLE_MODELS),
        }

    def configure(
        self,
        model_name: str | None = None,
        compute_type: str | None = None,
        allow_cpu_fallback: bool | None = None,
    ):
        with self._lock:
            if model_name:
                self.model_name = model_name
            if compute_type:
                self.compute_type = compute_type
            if allow_cpu_fallback is not None:
                self.allow_cpu_fallback = bool(allow_cpu_fallback)
            self.device = 'cuda'
            self.model = None

    def _compute_attempts(self):
        attempts = [self.compute_type] + self.gpu_compute_fallbacks
        uniq = []
        for compute in attempts:
            if compute not in uniq:
                uniq.append(compute)
        return uniq

    def _init_cuda_model(self, compute_type):
        self.model = WhisperModel(self.model_name, device='cuda', compute_type=compute_type)
        self.device = 'cuda'
        self.compute_type = compute_type

    def _init_cpu_model(self):
        self.model = WhisperModel(self.model_name, device='cpu', compute_type='int8')
        self.device = 'cpu'
        self.compute_type = 'int8'

    def _load_model(self):
        if self.model is not None:
            return
        if WhisperModel is None:
            raise RuntimeError(
                'faster-whisper is not installed. Run: pip install -r requirements.txt'
            )
        if self.device != 'cuda':
            raise RuntimeError(
                f"WHISPER_DEVICE must be 'cuda' (got '{self.device}'). CPU mode disabled."
            )
        init_errors = []
        for compute in self._compute_attempts():
            try:
                logger.info(f'Loading {self.model_name} on cuda ({compute})...')
                self._init_cuda_model(compute)
                return
            except Exception as e:
                init_errors.append(f'{compute}: {e}')
                logger.warning(f'Whisper CUDA init failed ({compute}): {e}')
        if self.allow_cpu_fallback:
            try:
                logger.warning('CUDA init failed. Falling back to CPU model (RAM-backed, slower).')
                self._init_cpu_model()
                return
            except Exception as e:
                init_errors.append(f'cpu-int8: {e}')
        raise RuntimeError(
            'Whisper GPU initialization failed for all compute modes. '
            f"Tried: {', '.join(self._compute_attempts())}. "
            f'Details: {" | ".join(init_errors)}'
        )

    def _transcribe_segments(self, file_path):
        segments, _ = self.model.transcribe(file_path, beam_size=5, language='ja')
        return [{'start': s.start, 'end': s.end, 'text': s.text} for s in segments]

    def transcribe(self, file_path):
        with self._lock:
            self._load_model()
            if self.device != 'cuda':
                raw_segments = self._transcribe_segments(file_path)
                return self.speaker_tracker.assign(file_path, raw_segments)

            attempts = self._compute_attempts()
            start_idx = attempts.index(self.compute_type) if self.compute_type in attempts else 0
            run_errors = []
            for compute in attempts[start_idx:]:
                if compute != self.compute_type:
                    logger.warning(
                        f'Whisper CUDA runtime fallback: retrying with compute_type={compute}'
                    )
                    self._init_cuda_model(compute)
                try:
                    raw_segments = self._transcribe_segments(file_path)
                    return self.speaker_tracker.assign(file_path, raw_segments)
                except Exception as e:
                    run_errors.append(f'{compute}: {e}')
                    logger.warning(f'Whisper CUDA runtime failed ({compute}): {e}')

            if self.allow_cpu_fallback:
                try:
                    logger.warning('CUDA runtime failed. Falling back to CPU model (RAM-backed, slower).')
                    self._init_cpu_model()
                    raw_segments = self._transcribe_segments(file_path)
                    return self.speaker_tracker.assign(file_path, raw_segments)
                except Exception as e:
                    run_errors.append(f'cpu-int8: {e}')

            raise RuntimeError(
                'Whisper GPU runtime failed for all compute modes. '
                f"Tried: {', '.join(attempts[start_idx:])}. "
                f'Details: {" | ".join(run_errors)}'
            )
