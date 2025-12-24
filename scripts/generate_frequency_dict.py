import os
import json
import pandas as pd
from collections import Counter
import re
from symspellpy import SymSpell

# Ayarlar
PROCESSED_DATA_DIR = "artifacts/processed_data"
OUTPUT_DIR = "app/assets/dictionaries"
LANGUAGES_JSON = "app/languages.json"
MIN_FREQUENCY = 1

def generate_dictionaries_from_parquet():
    # 1. Desteklenen dilleri ve ISO kodlarını yükle
    if not os.path.exists(LANGUAGES_JSON):
        print(f"Hata: {LANGUAGES_JSON} bulunamadı.")
        return
    
    with open(LANGUAGES_JSON, "r", encoding="utf-8") as f:
        languages_data = json.load(f)
    
    # Haritalama: "Turkish" -> "tr"
    name_to_iso = {l['name']: l['iso_code'] for l in languages_data}
    iso_counters = {l['iso_code']: Counter() for l in languages_data}

    print(f"Tanımlı {len(languages_data)} dil için Parquet taraması başlatılıyor...")

    # 2. Parquet dosyalarını işle
    if not os.path.exists(PROCESSED_DATA_DIR):
        print(f"Hata: {PROCESSED_DATA_DIR} dizini bulunamadı.")
        return

    pq_files = [f for f in os.listdir(PROCESSED_DATA_DIR) if f.endswith('.parquet')]
    if not pq_files:
        print(f"Bilgi: {PROCESSED_DATA_DIR} içinde parquet dosyası bulunamadı.")
        return

    for file_name in pq_files:
        path = os.path.join(PROCESSED_DATA_DIR, file_name)
        print(f"İşleniyor: {file_name}...")
        try:
            df = pd.read_parquet(path)
            
            # Hem 'source' hem 'target' sütunlarını ilgili dillerle eşleştirerek tara
            for text_col, lang_col in [('source', 'source_lang'), ('target', 'target_lang')]:
                if text_col in df.columns and lang_col in df.columns:
                    for lang_name, group in df.groupby(lang_col):
                        if lang_name in name_to_iso:
                            iso = name_to_iso[lang_name]
                            # Pandas ile hızlı kelime ayıklama ve frekans güncelleme
                            words = group[text_col].str.lower().str.replace(r'[^\w\s]', '', regex=True).str.split().explode()
                            iso_counters[iso].update(words.dropna().tolist())
        except Exception as e:
            print(f"Parquet okuma hatası ({file_name}): {e}")

    # 3. Sonuçları kaydet (Parquet ve Pickle olarak)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    total_updated = 0
    
    for iso, counter in iso_counters.items():
        if not counter:
            continue
            
        # Boş string temizliği
        if '' in counter: del counter['']
            
        # 3a. Parquet Kaydı (Yedek/Analiz için)
        output_parquet = os.path.join(OUTPUT_DIR, f"frequency_dictionary_{iso}.parquet")
        df_dict = pd.DataFrame(counter.most_common(), columns=['term', 'count'])
        df_dict = df_dict[df_dict['count'] >= MIN_FREQUENCY]
        df_dict.to_parquet(output_parquet, index=False, compression='snappy')
        
        # 3b. SymSpell Pickle Kaydı (Hızlı Yükleme için)
        output_pickle = os.path.join(OUTPUT_DIR, f"frequency_dictionary_{iso}.pickle")
        print(f"[{iso}] SymSpell Pickle oluşturuluyor (Bu işlem biraz sürebilir)...")
        
        try:
            # Geçici SymSpell nesnesi oluştur
            sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
            
            # Veriyi yükle
            for term, count in counter.most_common():
                if count >= MIN_FREQUENCY:
                    sym_spell.create_dictionary_entry(str(term), int(count))
            
            # Pickle olarak kaydet
            sym_spell.save_pickle(output_pickle)
            print(f"TAMAMLANDI: {output_pickle}")
            total_updated += 1
        except Exception as e:
            print(f"Pickle oluşturma hatası ({iso}): {e}")

    print(f"\nİşlem bitti. {total_updated} adet dil için sözlükler güncellendi.")

if __name__ == "__main__":
    generate_dictionaries_from_parquet()
