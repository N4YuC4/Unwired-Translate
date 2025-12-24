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
        Metni denetler, kelime hatalarını düzeltir ve noktalama işaretlerini yönetir.
        """
        if not text or not text.strip():
            return None

        sym_spell = self._load_language(lang_code)
        if not sym_spell:
            return None

        import re

        # 1. Metni parçalara ayır: Kelimeler ve diğer karakterler (noktalama, boşluk vs.)
        # (\w+): Kelimeler, ([^\w\s]+): Noktalama işaretleri, (\s+): Boşluklar
        # Bu regex grubu her şeyi yakalar ve sırasını korur.
        tokens = re.split(r'(\w+)', text)
        
        corrected_tokens = []
        
        for token in tokens:
            # Sadece kelimeyse ve tamamen rakam değilse düzeltme dene
            if re.match(r'^\w+$', token) and not token.isdigit():
                # Verbosity.TOP: En iyi tek sonucu döndür
                suggestions = sym_spell.lookup(token, Verbosity.TOP, max_edit_distance=2, transfer_casing=True)
                if suggestions:
                    corrected_tokens.append(suggestions[0].term)
                else:
                    corrected_tokens.append(token) # Öneri yoksa orijinali koru
            else:
                # Kelime değilse (noktalama, boşluk) olduğu gibi ekle
                corrected_tokens.append(token)
        
        corrected_text = "".join(corrected_tokens)

        # 2. Temel Dil Bilgisi Kuralları
        
        # Baş harfi büyüt (Eğer ilk karakter harfse)
        corrected_text = corrected_text.strip()
        if corrected_text and corrected_text[0].islower():
             corrected_text = corrected_text[0].upper() + corrected_text[1:]
        
        # Cümle sonu noktalama işareti kontrolü
        # Eğer cümle bir harf veya rakamla bitiyorsa nokta ekle.
        # Zaten . ! ? : ; gibi bir şey varsa dokunma.
        if corrected_text and corrected_text[-1].isalnum():
             corrected_text += "."

        # Eğer sonuç orijinalle aynıysa (boşluk temizliği hariç) None dön
        if corrected_text.strip() == text.strip():
            return None
            
        return corrected_text
