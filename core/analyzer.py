from sudachipy import dictionary, tokenizer
import pykakasi
from core.models import Token

class LinguisticAnalyzer:
    def __init__(self):
        self.tokenizer_obj = dictionary.Dictionary().create()
        self.kks = pykakasi.kakasi()

    def get_romaji(self, text):
        result = self.kks.convert(text)
        return "".join([item['hepburn'] for item in result])

    def analyze(self, text):
        mode = tokenizer.Tokenizer.SplitMode.C
        tokens_raw = self.tokenizer_obj.tokenize(text, mode)
        
        processed_tokens = []
        for t in tokens_raw:
            surface = t.surface()
            processed_tokens.append(Token(
                text=surface,
                reading=self.get_romaji(surface),
                url=f"https://jisho.org/search/{surface}"
            ))
        return processed_tokens