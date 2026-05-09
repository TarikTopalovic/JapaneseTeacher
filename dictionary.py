from sudachipy import dictionary, tokenizer
from jamdict import Jamdict
import pykakasi


class JpAnalyzer:
    def __init__(self):
        # Initialize Sudachi for splitting text
        self.tokenizer_obj = dictionary.Dictionary().create()
        # Initialize Jamdict for meanings
        self.jam = Jamdict()
        # Initialize Kakasi for Romaji conversion
        self.kks = pykakasi.kakasi()

    def get_romaji(self, text):
        """Converts a string of Japanese text to Romaji."""
        result = self.kks.convert(text)
        return "".join([item['hepburn'] for item in result])

    def analyze_sentence(self, text):
        mode = tokenizer.Tokenizer.SplitMode.C
        tokens = self.tokenizer_obj.tokenize(text, mode)

        results = []
        for token in tokens:
            surface = token.surface()
            # Convert the specific token to Romaji
            romaji_word = self.get_romaji(surface)

            # Lookup the word in the local dictionary
            lookup = self.jam.lookup(surface)
            meaning = lookup.entries[0].definitions[0].contents[0] if lookup.entries else "No definition found"

            results.append({
                "word": surface,
                "romaji": romaji_word,
                "reading": token.reading_form(),
                "meaning": meaning
            })
        return results