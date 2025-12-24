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
        Metni denetler ve olası düzeltmeleri (en fazla 3) liste olarak döndürür.
        """
        if not text or not text.strip():
            return []

        sym_spell = self._load_language(lang_code)
        if not sym_spell:
            return []

        import re
        import itertools

        # Regex ile ayır
        tokens = re.split(r'(\w+)', text)
        
        # Her token için olası düzeltmelerin listesini tutacağız
        # Örn: [['Merhaba'], [' '], ['dunya', 'dünya'], ['.']]
        token_options = []
        
        for token in tokens:
            options = []
            
            # Sadece harf içeren kelimeler için işlem yap
            if re.match(r'^\w+$', token) and not token.isdigit():
                # 1. Tam eşleşme var mı?
                exact_match = sym_spell.lookup(token, Verbosity.TOP, max_edit_distance=0)
                if exact_match:
                    options.append(token)
                else:
                    # 2. Kelime Bölümlendirme (Segmentation) - "yada" -> "ya da"
                    seg_result = sym_spell.word_segmentation(token, max_edit_distance=0)
                    if seg_result.segmented_string and " " in seg_result.segmented_string:
                        options.append(seg_result.segmented_string)
                    
                    # 3. Klasik Düzeltme (Lookup) - "yada" -> "yana", "yara"
                    # lookup_compound sadece en iyi sonucu verir, alternatifleri vermez.
                    # Bu yüzden tek kelime için 'lookup' kullanıyoruz.
                    suggestions = sym_spell.lookup(token, Verbosity.CLOSEST, max_edit_distance=2, transfer_casing=True)
                    
                    # En iyi 3 farklı öneriyi al
                    for suggestion in suggestions:
                        if len(options) >= 3:
                            break
                        if suggestion.term not in options:
                            options.append(suggestion.term)
                    
                    # Eğer hiç öneri yoksa orijinali ekle
                    if not options:
                        options.append(token)
            else:
                # Kelime değilse olduğu gibi ekle
                options.append(token)
            
            token_options.append(options)

        # Cartesian Product: Tüm olasılıkları birleştir
        # Bu işlem çok sayıda kombinasyon üretebilir, bu yüzden sınırlayalım.
        # Sadece ilk 3 kombinasyonu alacağız.
        # itertools.product(*token_options) tüm kombinasyonları verir.
        
        results = []
        # En fazla 50 kombinasyon dene, en iyi 3 farklıyı seç
        for combination in itertools.islice(itertools.product(*token_options), 50):
            sentence = "".join(combination)
            
            # Dil Bilgisi Kuralları
            sentence = sentence.strip()
            if sentence and sentence[0].islower():
                sentence = sentence[0].upper() + sentence[1:]
            
            if sentence and sentence[-1].isalnum():
                sentence += "."
            
            if sentence.strip() != text.strip() and sentence not in results:
                results.append(sentence)
                if len(results) >= 3:
                    break
        
        return results
