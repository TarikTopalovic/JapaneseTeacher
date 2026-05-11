from core.engine import AudioEngine
from core.linguistics import LinguisticsManager
from core.sentence_translation import SentenceTranslator
from core.database import ImmersionDB

class ImmersionProcessor:
    def __init__(self):
        self.engine = AudioEngine()
        self.ling = LinguisticsManager()
        self.translator = SentenceTranslator()
        self.db = ImmersionDB()

    def process_media(self, file_path, filename):
        # 1. Audio -> Raw Text
        raw_segments = self.engine.transcribe(file_path)
        final_segments = []
        
        for seg in raw_segments:
            # 2. Raw Text -> Tokens with English meanings
            tokens = self.ling.analyze(seg['text'])
            
            # 3. Save each word to Vocab Bank
            for t in tokens:
                self.db.add_word(t['text'], t['reading'], t['meaning'])
                
            final_segments.append({
                'start': seg['start'],
                'end': seg['end'],
                'text': seg['text'],
                'speaker': seg.get('speaker', 'SPEAKER_1'),
                'translation': self.translator.translate(seg['text']),
                'tokens': tokens
            })
        
        return final_segments
