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
        self.dict_paths = {}
        
        # app/languages.json dosyasını oku ve sözlük yollarını dinamik oluştur
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        langs_json_path = os.path.join(base_path, "languages.json")
        dicts_dir = os.path.join(base_path, "assets", "dictionaries")

        try:
            if os.path.exists(langs_json_path):
                import json
                with open(langs_json_path, "r", encoding="utf-8") as f:
                    languages = json.load(f)
                    for lang in languages:
                        iso = lang.get("iso_code")
                        if iso:
                            iso = iso.lower()
                            # Sadece Pickle yolu
                            self.dict_paths[iso] = os.path.join(
                                dicts_dir, f"frequency_dictionary_{iso}.pickle"
                            )
            else:
                logger.error(f"Dinamik sözlük yükleme başarısız: {langs_json_path} bulunamadı.")
        except Exception as e:
            logger.error(f"Dinamik sözlük yapılandırma hatası: {e}")

    def _load_language(self, lang_code):
        """İlgili dilin Pickle sözlüğünü yükler."""
        # Dil kodunu normalize et
        code_map = {"English": "en", "Turkish": "tr"}
        lang = code_map.get(lang_code, lang_code).lower()

        if lang not in self.dict_paths:
            logger.warning(f"Dil desteklenmiyor veya sözlük tanımlı değil: {lang}")
            return None

        if lang in self._symspells:
            return self._symspells[lang]

        pickle_path = self.dict_paths[lang]
        
        if not os.path.exists(pickle_path):
            logger.error(f"Sözlük dosyası bulunamadı (Pickle): {pickle_path}. Lütfen önce 'scripts/generate_frequency_dict.py' scriptini çalıştırın.")
            return None

        try:
            logger.info(f"Sözlük yükleniyor (Pickle): {lang}")
            sym_spell = SymSpell(max_dictionary_edit_distance=self.max_edit_distance, prefix_length=self.prefix_length)
            
            if sym_spell.load_pickle(pickle_path):
                self._symspells[lang] = sym_spell
                logger.info(f"Sözlük başarıyla yüklendi: {lang}")
                return sym_spell
            else:
                logger.error(f"Pickle dosyası yüklenemedi: {pickle_path}")
                return None
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
            # Sadece harf içeren kelimeler için işlem yap
            if re.match(r'^\w+$', token) and not token.isdigit():
                # 1. Önce kelime sözlükte var mı bak (Edit distance 0)
                exact_match = sym_spell.lookup(token, Verbosity.TOP, max_edit_distance=0)
                
                if exact_match:
                    corrected_tokens.append(token)
                else:
                    # 2. Sözlükte yoksa, önce "Kelime Bölümlendirme" (Word Segmentation) dene
                    # Bu 'yada' -> 'ya da' gibi durumları yakalamak için en iyi yoldur.
                    seg_result = sym_spell.word_segmentation(token, max_edit_distance=0)
                    
                    # Eğer bölme işlemi sonucunda 1'den fazla kelime çıktıysa ve 
                    # bu kelimeler sözlükte güçlüyse bunu kullan.
                    if seg_result.segmented_string and " " in seg_result.segmented_string:
                        corrected_tokens.append(seg_result.segmented_string)
                    else:
                        # 3. Bölme işlemi sonuç vermediyse klasik düzeltme (lookup_compound) dene
                        suggestions = sym_spell.lookup_compound(token, max_edit_distance=2, transfer_casing=True)
                        if suggestions:
                            corrected_tokens.append(suggestions[0].term)
                        else:
                            corrected_tokens.append(token)
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
