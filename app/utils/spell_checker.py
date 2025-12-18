import os
import logging
from symspellpy import SymSpell, Verbosity

logger = logging.getLogger("SpellChecker")

class SpellChecker:
    _instance = None
    _dictionaries = {}
    _symspells = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SpellChecker, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        # Varsayılan ayarlar
        self.max_edit_distance = 2
        self.prefix_length = 7
        
        # Sözlük yolları
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.dict_paths = {
            "en": os.path.join(base_path, "assets", "dictionaries", "frequency_dictionary_en.txt"),
            "tr": os.path.join(base_path, "assets", "dictionaries", "frequency_dictionary_tr.txt")
        }

    def _load_language(self, lang_code):
        """İlgili dilin sözlüğünü yükler (lazy loading)."""
        # Dil kodunu normalize et (English -> en, Turkish -> tr)
        # app/languages.json'daki yapıya göre "English" veya "en" gelebilir.
        code_map = {"English": "en", "Turkish": "tr"}
        lang = code_map.get(lang_code, lang_code).lower()

        if lang not in self.dict_paths:
            logger.warning(f"Dil desteklenmiyor veya sözlük tanımlı değil: {lang}")
            return None

        if lang in self._symspells:
            return self._symspells[lang]

        path = self.dict_paths[lang]
        if not os.path.exists(path):
            logger.error(f"Sözlük dosyası bulunamadı: {path}")
            return None

        try:
            logger.info(f"Sözlük yükleniyor: {lang} ({path})")
            sym_spell = SymSpell(max_dictionary_edit_distance=self.max_edit_distance, prefix_length=self.prefix_length)
            # Term index 0, count index 1 (standart format)
            if not sym_spell.load_dictionary(path, term_index=0, count_index=1):
                logger.error(f"Sözlük yüklenemedi: {path}")
                return None
            
            self._symspells[lang] = sym_spell
            return sym_spell
        except Exception as e:
            logger.error(f"Sözlük yükleme hatası ({lang}): {e}")
            return None

    def correct(self, text, lang_code):
        """
        Metni denetler ve düzeltilmiş halini döndürür.
        Eğer düzeltme yoksa veya hata varsa None döner.
        """
        if not text or not text.strip():
            return None

        sym_spell = self._load_language(lang_code)
        if not sym_spell:
            return None

        # Compound Check: Cümle bazlı düzeltme (kelime kelime değil, bağlamı da gözetmeye çalışır - ama bigram yoksa sadece kelime bazlıdır)
        # transfer_casing=True: Büyük/küçük harf yapısını korur.
        try:
            suggestions = sym_spell.lookup_compound(text, max_edit_distance=2, transfer_casing=True)
            
            if suggestions:
                best_suggestion = suggestions[0].term
                # Eğer öneri orijinal metinle aynıysa None dön
                if best_suggestion == text:
                    return None
                return best_suggestion
        except Exception as e:
            logger.error(f"Düzeltme hatası: {e}")
            return None

        return None
