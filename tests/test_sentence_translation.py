from core.sentence_translation import SentenceTranslator


def test_translation_disabled_returns_placeholder(monkeypatch):
    monkeypatch.setenv("SENTENCE_TRANSLATION_ENABLED", "0")
    translator = SentenceTranslator()
    assert translator.translate("テストです") == "-"


def test_empty_sentence_returns_empty_string():
    translator = SentenceTranslator()
    assert translator.translate("") == ""
