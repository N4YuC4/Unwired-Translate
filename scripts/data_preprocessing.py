import os
import yaml
import pandas as pd
import re
import ftfy
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split
import logging
from datetime import datetime
from bs4 import BeautifulSoup
import gc

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

# --- VERİ YÜKLEME FONKSİYONU (NOTEBOOK'TAN) ---
def load_parquet(source_path, target_path, start=0, num=None):
    if not os.path.exists(source_path) or not os.path.exists(target_path):
        logging.warning(f"Dosya bulunamadı: {source_path} veya {target_path}")
        return pd.DataFrame(columns=["source", "target"])
    
    pf_src = pq.ParquetFile(source_path)
    pf_tgt = pq.ParquetFile(target_path)
    
    # Satır sayılarını kontrol et
    if pf_src.metadata.num_rows != pf_tgt.metadata.num_rows:
        logging.warning(f"Satır sayıları eşleşmiyor: {source_path} ({pf_src.metadata.num_rows}) vs {target_path} ({pf_tgt.metadata.num_rows})")
        # Güvenlik için minimum olanı alabiliriz veya hata verebiliriz. 
        # Notebook mantığı iter_batches ile eşleştiriyor.
        
    iter_src = pf_src.iter_batches()
    iter_tgt = pf_tgt.iter_batches()
    dfs = []
    collected = 0
    scanned = 0
    
    for b_src, b_tgt in zip(iter_src, iter_tgt):
        if num is not None and collected >= num: break
        b_rows = b_src.num_rows
        
        # Başlangıç satırına gelene kadar atla
        if scanned + b_rows < start:
            scanned += b_rows
            continue
            
        offset = max(0, start - scanned)
        
        # Batch'leri pandas'a çevir
        p_src = b_src.to_pandas()
        p_tgt = b_tgt.to_pandas()
        
        sl_src = p_src.iloc[offset:]
        sl_tgt = p_tgt.iloc[offset:]
        
        if num is not None:
            needed = num - collected
            if len(sl_src) > needed:
                sl_src = sl_src.iloc[:needed]
                sl_tgt = sl_tgt.iloc[:needed]
                
        df_b = pd.DataFrame({"source": sl_src.iloc[:, 0], "target": sl_tgt.iloc[:, 0]})
        dfs.append(df_b)
        collected += len(df_b)
        scanned += b_rows
        
    if not dfs: return pd.DataFrame(columns=["source", "target"])
    return pd.concat(dfs, ignore_index=True)

# --- METİN NORMALLEŞTİRME FONKSİYONU ---
def normalize_text(text):
    if text is None: return ""
    text = str(text)
    text = ftfy.fix_text(text)
    # HTML taglerini temizle (BeautifulSoup yavaş olabilir, regex daha hızlıdır büyük veri için)
    text = re.sub(r'<[^>]+>', '', text) 
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'([!?.])\1+', r'\1', text) # Tekrarlayan noktalama işaretlerini tekile indir
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# --- ANA İŞLEM FONKSİYONU ---
def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    logging.info("Veri ön işleme başlatılıyor...")
    
    dataset_config = config['dataset']
    source_lang = config['language']['source']
    target_lang = config['language']['target']
    
    base_path = dataset_config.get('base_path', 'datasets')
    dataset_names = dataset_config.get('names', [])
    max_lines = dataset_config.get('max_lines', 100000)
    start_line = dataset_config.get('start_line', 0)
    read_first_completely = dataset_config.get('read_first_set_completely', False)
    
    dfs = []
    total_rows = 0
    
    for i, dset_name in enumerate(dataset_names):
        remaining = max_lines - total_rows
        
        if not read_first_completely and remaining <= 0:
            logging.info(f"Maksimum satır sayısına ({max_lines}) ulaşıldı. {dset_name} atlanıyor.")
            continue
            
        if read_first_completely and i > 0 and remaining <= 0:
            continue
            
        # Yol oluşturma: datasets/NAME/NAME-Lang.parquet varsayımı
        ds_path = os.path.join(base_path, dset_name)
        src_filename = f"{dset_name}-{source_lang}.parquet"
        tgt_filename = f"{dset_name}-{target_lang}.parquet"
        
        src_p = os.path.join(ds_path, src_filename)
        tgt_p = os.path.join(ds_path, tgt_filename)
        
        num_to_read = None if (read_first_completely and i == 0) else remaining
        
        logging.info(f"Yükleniyor: {dset_name} | Hedeflenen: {num_to_read if num_to_read else 'Tümü'}")
        
        df_temp = load_parquet(src_p, tgt_p, start=start_line, num=num_to_read)
        
        if df_temp.empty:
            logging.warning(f"{dset_name} boş veya yüklenemedi.")
            continue

        df_temp['source_lang'] = source_lang
        df_temp['target_lang'] = target_lang
        dfs.append(df_temp)
        total_rows += len(df_temp)
        logging.info(f"{dset_name} setinden {len(df_temp)} satır yüklendi.")

    if not dfs:
        logging.error("Hiçbir veri yüklenemedi!")
        return

    df = pd.concat(dfs, ignore_index=True)
    del dfs
    gc.collect()
    
    logging.info(f"Toplam Ham Veri: {len(df)}")

    # --- MIRRORING (ÇİFT YÖNLÜ VERİ ÇOĞALTMA) ---
    # Notebook'taki mantık: Her zaman çift yönlü yapılıyor.
    logging.info("Veri seti çift yönlü (bidirectional) hale getiriliyor...")
    
    df_reverse = df.copy()
    # Sütunları değiştir
    df_reverse = df_reverse.rename(columns={'source': 'target', 'target': 'source'})
    # Dilleri değiştir
    df_reverse['source_lang'] = target_lang
    df_reverse['target_lang'] = source_lang
    
    df = pd.concat([df, df_reverse], ignore_index=True)
    
    # Karıştır
    df = df.sample(frac=1, random_state=dataset_config['random_state']).reset_index(drop=True)
    
    logging.info(f"Mirroring sonrası toplam veri: {len(df)}")
    
    # --- NORMALİZASYON ---
    logging.info("Metin normalizasyonu uygulanıyor...")
    df['source'] = df['source'].apply(normalize_text)
    df['target'] = df['target'].apply(normalize_text)
    
    # Boş satırları temizle
    df.dropna(subset=['source', 'target'], inplace=True)
    df = df[(df['source'].str.len() > 0) & (df['target'].str.len() > 0)]
    
    # --- SPLIT VE KAYDETME ---
    logging.info("Eğitim ve test olarak ayrılıyor...")
    train_df, test_df = train_test_split(
        df,
        test_size=dataset_config['test_size'],
        random_state=dataset_config['random_state'],
        stratify=df["source_lang"] # Dil dengesini korumak için stratify önemli
    )
    
    output_dir = config['artifacts']['processed_data_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, "train.parquet")
    test_path = os.path.join(output_dir, "test.parquet")
    
    train_df.to_parquet(train_path)
    test_df.to_parquet(test_path)
    
    logging.info(f"Train Seti: {len(train_df)} satır -> {train_path}")
    logging.info(f"Test Seti: {len(test_df)} satır -> {test_path}")
    logging.info("Veri ön işleme tamamlandı.")

if __name__ == "__main__":
    main()