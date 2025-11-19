import os
import yaml
import pandas as pd
import re
import ftfy
from sklearn.model_selection import train_test_split
import logging
from datetime import datetime
from bs4 import BeautifulSoup

# --- LOGLAMA AYARLARI ---
log_dir = "logs/data_preprocessing"
os.makedirs(log_dir, exist_ok=True)
log_filename = f"{log_dir}/data_preprocessing_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(log_filename),
                        logging.StreamHandler()
                    ])

# --- VERİ YÜKLEME FONKSİYONU ---
def load_parquet_dataset(source_path, target_path, source_lang, target_lang, start_line=0, num_lines=None):
    """
    Parquet formatındaki kaynak ve hedef veri setlerini okur ve birleştirir.
    """
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Kaynak dosya bulunamadı: {source_path}")
    if not os.path.exists(target_path):
        raise FileNotFoundError(f"Hedef dosya bulunamadı: {target_path}")

    source_df = pd.read_parquet(source_path)
    target_df = pd.read_parquet(target_path)

    df = pd.DataFrame({
        source_lang: source_df.iloc[:, 0],
        target_lang: target_df.iloc[:, 0]
    })

    if num_lines is not None:
        df = df.iloc[start_line:start_line + num_lines]

    return df


# --- METİN NORMALLEŞTİRME FONKSİYONU ---
def normalize_text(text):
    text = ftfy.fix_text(text)
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    text = text.strip()
    return text

# --- ANA İŞLEM FONKSİYONU ---
def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    logging.info("Veri seti yükleniyor...")
    
    veri_seti_config = config['dataset']
    lang_config = config['language']
    
    source_lang = lang_config['source']
    target_lang = lang_config['target']

    source_path = os.path.join(veri_seti_config['path'], veri_seti_config['source_lang_file'].format(source_lang=source_lang))
    target_path = os.path.join(veri_seti_config['path'], veri_seti_config['target_lang_file'].format(target_lang=target_lang))

    df = load_parquet_dataset(
        source_path,
        target_path,
        source_lang,
        target_lang,
        start_line=veri_seti_config['start_line'],
        num_lines=veri_seti_config['max_lines']
    )
    logging.info(f"Toplam {len(df)} satır veri yüklendi.")
    
    df.dropna(subset=[source_lang, target_lang], inplace=True)
    df = df[(df[source_lang].str.len() > 0) & (df[target_lang].str.len() > 0)]
    logging.info(f"Eksik veriler temizlendikten sonra {len(df)} satır kaldı.")

    logging.info("Metin normalizasyonu uygulanıyor...")
    df[source_lang] = df[source_lang].apply(normalize_text)
    df[target_lang] = df[target_lang].apply(normalize_text)

    logging.info("Yinelenen veriler kaldırılıyor...")
    df.drop_duplicates(subset=[source_lang, target_lang], keep='first', inplace=True)
    logging.info(f"Yinelenenler kaldırıldıktan sonra {len(df)} satır kaldı.")
    logging.info("-" * 40)

    logging.info("Eğitim ve test verileri ayrılıyor...")
    train_df, test_df = train_test_split(
        df,
        test_size=veri_seti_config['test_size'],
        random_state=veri_seti_config['random_state']
    )

    logging.info("İşlenmiş veriler kaydediliyor...")
    output_dir = config['artifacts']['processed_data_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    # Sütun isimlerini config'e göre değil, sabit olarak kaydediyoruz.
    # Bu, train script'inin belediği formatla tutarlılık sağlar.
    train_df.columns = ['source', 'target']
    test_df.columns = ['source', 'target']

    train_df.to_parquet(os.path.join(output_dir, "train.parquet"))
    test_df.to_parquet(os.path.join(output_dir, "test.parquet"))
    
    logging.info(f"Veriler başarıyla '{output_dir}' dizinine kaydedildi.")

if __name__ == "__main__":
    main()