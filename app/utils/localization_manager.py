import json
import os
import logging

logger = logging.getLogger("LocalizationManager")

class LocalizationManager:
    def __init__(self, locales_dir="app/locales", default_lang="en"):
        self.locales_dir = locales_dir
        self.default_lang = default_lang
        self.current_lang = default_lang
        self.translations = {}
        self.load_language(default_lang)

    def load_language(self, lang_code):
        """Belirtilen dil dosyasını yükler."""
        try:
            file_path = os.path.join(self.locales_dir, f"{lang_code}.json")
            if not os.path.exists(file_path):
                logger.warning(f"Dil dosyası bulunamadı: {file_path}. Varsayılan dil kullanılacak.")
                if lang_code != self.default_lang:
                    self.load_language(self.default_lang)
                return

            with open(file_path, "r", encoding="utf-8") as f:
                self.translations = json.load(f)
            self.current_lang = lang_code
            logger.info(f"Dil yüklendi: {lang_code}")
        except Exception as e:
            logger.error(f"Dil yüklenirken hata: {e}")
            # Hata durumunda boş bırakma, varsayılana dönmeyi dene
            if lang_code != self.default_lang:
                 self.load_language(self.default_lang)

    def get(self, key):
        """Anahtara karşılık gelen çeviriyi döndürür."""
        return self.translations.get(key, key)

    def set_language(self, lang_code):
        """Dili değiştirir ve yükler."""
        if self.current_lang != lang_code:
            self.load_language(lang_code)

# Global instance
_loc_manager = None

def get_manager(locales_dir="app/locales", default_lang="en"):
    global _loc_manager
    if _loc_manager is None:
        _loc_manager = LocalizationManager(locales_dir, default_lang)
    return _loc_manager
