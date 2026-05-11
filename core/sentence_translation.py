import logging
import os
import threading
import time
from typing import Dict

logger = logging.getLogger(__name__)
logging.getLogger("argostranslate").setLevel(logging.WARNING)
logging.getLogger("argostranslate.utils").setLevel(logging.WARNING)

NO_TRANSLATION = "-"


class SentenceTranslator:
    def __init__(self):
        self.enabled = os.getenv("SENTENCE_TRANSLATION_ENABLED", "1") == "1"
        self.source_lang = os.getenv("TRANSLATE_SOURCE_LANG", "ja")
        self.target_lang = os.getenv("TRANSLATE_TARGET_LANG", "en")
        self._translation = None
        self._cache: Dict[str, str] = {}
        self._lock = threading.Lock()
        self._retry_seconds = int(os.getenv("TRANSLATE_RETRY_SECONDS", "30"))
        self._next_retry_ts = 0.0

    def _resolve_translation(self, installed_languages):
        source = next((lang for lang in installed_languages if lang.code == self.source_lang), None)
        target = next((lang for lang in installed_languages if lang.code == self.target_lang), None)
        if source is None or target is None:
            return None
        try:
            return source.get_translation(target)
        except Exception:
            return None

    def _install_language_pair(self):
        import argostranslate.package

        logger.info("Installing Argos language package (ja -> en)...")
        argostranslate.package.update_package_index()
        available_packages = argostranslate.package.get_available_packages()
        package = next(
            (
                pkg
                for pkg in available_packages
                if pkg.from_code == self.source_lang and pkg.to_code == self.target_lang
            ),
            None,
        )
        if package is None:
            raise RuntimeError(
                f"Argos package {self.source_lang}->{self.target_lang} not found in index"
            )
        download_path = package.download()
        argostranslate.package.install_from_path(download_path)

    def _ensure_ready(self) -> bool:
        now = time.time()
        with self._lock:
            if self._translation is not None:
                return True
            if now < self._next_retry_ts:
                return False
            self._next_retry_ts = now + self._retry_seconds

        if not self.enabled:
            logger.info("Sentence translation disabled via SENTENCE_TRANSLATION_ENABLED=0")
            return False

        try:
            import argostranslate.translate
        except Exception as e:
            logger.warning(f"Argos Translate import failed: {e}")
            return False

        try:
            installed_languages = argostranslate.translate.get_installed_languages()
            translation = self._resolve_translation(installed_languages)
            if translation is None:
                self._install_language_pair()
                installed_languages = argostranslate.translate.get_installed_languages()
                translation = self._resolve_translation(installed_languages)
            if translation is None:
                raise RuntimeError(
                    f"Argos language pair unavailable ({self.source_lang}->{self.target_lang})"
                )
            with self._lock:
                self._translation = translation
            logger.info("Sentence translation enabled (Argos)")
            return True
        except Exception as e:
            logger.warning(f"Sentence translation unavailable: {e}")
            return False

    def translate(self, text: str) -> str:
        cleaned = (text or "").strip()
        if not cleaned:
            return ""

        with self._lock:
            cached = self._cache.get(cleaned)
        if cached is not None:
            return cached

        if not self._ensure_ready():
            return NO_TRANSLATION

        with self._lock:
            translation = self._translation
        try:
            translated = (translation.translate(cleaned) or "").strip()
            result = translated or NO_TRANSLATION
        except Exception as e:
            logger.warning(f"Sentence translation failed: {e}")
            result = NO_TRANSLATION

        with self._lock:
            self._cache[cleaned] = result
        return result
