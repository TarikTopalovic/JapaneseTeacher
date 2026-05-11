import logging
import re
import threading
from typing import Dict

import pykakasi
from jamdict import Jamdict
from sudachipy import dictionary, tokenizer

logger = logging.getLogger(__name__)

NO_MEANING = "-"


class LinguisticsManager:
    def __init__(self):
        self._dict = dictionary.Dictionary()
        self._local = threading.local()
        self._meaning_cache: Dict[str, str] = {}
        self._cache_lock = threading.Lock()

    def _resources(self):
        if not hasattr(self._local, "tokenizer_obj"):
            self._local.tokenizer_obj = self._dict.create()
            self._local.kks = pykakasi.kakasi()
            self._local.jamdict = Jamdict()
        return self._local.tokenizer_obj, self._local.kks, self._local.jamdict

    def get_romaji(self, text):
        _, kks, _ = self._resources()
        res = kks.convert(text)
        return "".join([i["hepburn"] for i in res])

    @staticmethod
    def _normalize_lemma(token) -> str:
        base = (token.dictionary_form() or "").strip()
        if base and base != "*":
            return base
        return token.surface().strip()

    @staticmethod
    def _clean_meaning(text: str) -> str:
        if not text:
            return NO_MEANING
        cleaned = re.sub(r"\s+", " ", text).strip()
        cleaned = re.sub(r"\(\(.*?\)\)", "", cleaned).strip()
        cleaned = re.sub(r"^\d+\.\s*", "", cleaned).strip()
        if "/" in cleaned:
            cleaned = cleaned.split("/", 1)[0].strip()
        return cleaned or NO_MEANING

    @classmethod
    def _extract_meaning(cls, entry) -> str:
        senses = getattr(entry, "senses", None) or []
        for sense in senses:
            glosses = getattr(sense, "gloss", None) or getattr(sense, "glosses", None) or []
            for gloss in glosses:
                if isinstance(gloss, str):
                    meaning = gloss
                else:
                    meaning = (
                        getattr(gloss, "text", None)
                        or getattr(gloss, "value", None)
                        or str(gloss)
                    )
                cleaned = cls._clean_meaning(meaning)
                if cleaned != NO_MEANING:
                    return cleaned

        entry_text = str(entry)
        if ":" in entry_text:
            entry_text = entry_text.split(":", 1)[1]
        return cls._clean_meaning(entry_text)

    def _lookup_meaning(self, lemma: str, jamdict: Jamdict) -> str:
        with self._cache_lock:
            cached = self._meaning_cache.get(lemma)
        if cached is not None:
            return cached

        try:
            result = jamdict.lookup(lemma)
            entries = getattr(result, "entries", None) or []
            meaning = self._extract_meaning(entries[0]) if entries else NO_MEANING
        except Exception as e:
            logger.warning(f"Jamdict lookup failed for '{lemma}': {e}")
            meaning = NO_MEANING

        with self._cache_lock:
            self._meaning_cache[lemma] = meaning
        return meaning

    def analyze(self, text: str) -> list:
        tokenizer_obj, _, jamdict = self._resources()
        mode = tokenizer.Tokenizer.SplitMode.C
        tokens_raw = tokenizer_obj.tokenize(text, mode)
        if not tokens_raw:
            return []

        processed = []
        for t in tokens_raw:
            surface = t.surface().strip()
            if not surface:
                continue
            lemma = self._normalize_lemma(t)
            processed.append(
                {
                    "text": surface,
                    "reading": self.get_romaji(surface),
                    "meaning": self._lookup_meaning(lemma, jamdict),
                    "url": f"https://jisho.org/search/{lemma}",
                }
            )
        return processed
