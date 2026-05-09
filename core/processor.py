from core.engine import AudioEngine
from core.analyzer import LinguisticAnalyzer
from core.models import Segment, Lesson

class ImmersionProcessor:
    def __init__(self):
        self.engine = AudioEngine()
        self.analyzer = LinguisticAnalyzer()

    def process_media(self, file_path, filename):
        raw_segments = self.engine.transcribe(file_path)
        final_segments = []
        
        for seg in raw_segments:
            tokens = self.analyzer.analyze(seg['text'])
            final_segments.append(Segment(
                start=seg['start'],
                end=seg['end'],
                text=seg['text'],
                tokens=tokens
            ))
        
        return Lesson(filename=filename, segments=final_segments)