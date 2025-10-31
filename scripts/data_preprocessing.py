# Gerekli kütüphanelerin import edilmesi
import os  # İşletim sistemiyle ilgili işlemler için (dosya yolları, klasör oluşturma vb.)
import yaml  # YAML formatındaki yapılandırma dosyalarını okumak için
import pandas as pd  # Veri işleme ve analizi için temel kütüphane
import re  # Metin içinde desen arama ve değiştirme (RegEx) işlemleri için
import ftfy  # Metinlerdeki bozuk Unicode karakterleri ve kodlama hatalarını düzeltmek için
from sklearn.model_selection import train_test_split  # Veri setini eğitim ve test olarak ayırmak için
import pyarrow.dataset as ds  # Büyük veri setlerini (özellikle Parquet) verimli bir şekilde okumak için
from tqdm import tqdm  # Döngülerde ilerleme çubuğu göstermek için
import logging  # Olayları ve hataları kaydetmek (loglamak) için
from datetime import datetime  # Tarih ve saat bilgileriyle çalışmak için
from bs4 import BeautifulSoup  # HTML ve XML dosyalarını ayrıştırmak (parse) ve içindeki veriyi çekmek için

# --- LOGLAMA AYARLARI ---
# Bu scriptin çalışması sırasında oluşacak olayları kaydetmek için bir loglama sistemi kuruluyor.
log_dir = "logs/data_preprocessing"  # Log dosyalarının kaydedileceği klasör
os.makedirs(log_dir, exist_ok=True)  # Eğer klasör yoksa oluştur
# Her çalıştırmada benzersiz bir log dosyası adı oluşturuluyor (örn: data_preprocessing_2023-10-27_15-30-00.log)
log_filename = f"{log_dir}/data_preprocessing_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
# Loglama temel yapılandırması:
logging.basicConfig(level=logging.INFO,  # Kaydedilecek minimum log seviyesi (INFO ve üstü)
                    format='%(asctime)s - %(levelname)s - %(message)s',  # Log mesajlarının formatı
                    handlers=[
                        logging.FileHandler(log_filename),  # Logları dosyaya yaz
                        logging.StreamHandler()  # Logları konsola (ekrana) da yaz
                    ])

# --- VERİ YÜKLEME FONKSİYONU ---
def load_parquet_dataset(parquet_dir, start_line=0, num_lines=None, show_progress=True):
    """
    Parquet formatındaki bir veri setini, bellek kullanımını optimize ederek okur.
    Bu fonksiyon, çok büyük dosyaların tamamını belleğe yüklemek yerine parçalar halinde (batch) işler.

    Args:
        parquet_dir (str): Parquet dosyalarının bulunduğu klasörün yolu.
        start_line (int, optional): Okumaya başlanacak satır numarası. Varsayılan: 0.
        num_lines (int, optional): Okunacak toplam satır sayısı. Varsayılan: None (tümünü oku).
        show_progress (bool, optional): Okuma sırasında bir ilerleme çubuğu gösterilip gösterilmeyeceği. Varsayılan: True.

    Raises:
        FileNotFoundError: Belirtilen `parquet_dir` klasörü bulunamazsa bu hata fırlatılır.

    Returns:
        pd.DataFrame: Okunan verileri içeren bir Pandas DataFrame.
    """
    if not os.path.exists(parquet_dir):
        raise FileNotFoundError(f"Klasör bulunamadı: {parquet_dir}")

    # PyArrow kütüphanesi ile veri setini aç
    dataset = ds.dataset(parquet_dir, format="parquet")
    # Sadece 'en' (İngilizce) ve 'tr' (Türkçe) sütunlarını oku
    scanner = dataset.scanner(columns=["en", "tr"])
    reader = scanner.to_reader()
    
    dataframes = []  # Okunan her bir parçanın (batch) ekleneceği liste
    rows_collected = 0  # Şu ana kadar toplanan satır sayısı
    total_rows_scanned = 0  # Şu ana kadar taranan toplam satır sayısı
    desc = f"Satır {start_line}'dan itibaren taranıyor"  # İlerleme çubuğunda gösterilecek metin

    # Veri setini parçalar halinde oku
    for batch in tqdm(reader, disable=not show_progress, desc=desc):
        # Eğer istenen satır sayısına ulaşıldıysa döngüyü durdur
        if num_lines is not None and rows_collected >= num_lines:
            break
        
        batch_rows = batch.num_rows
        # Başlangıç satırına gelene kadar parçaları atla
        if total_rows_scanned + batch_rows < start_line:
            total_rows_scanned += batch_rows
            continue
            
        # Başlangıç satırını içeren parçayı doğru yerden dilimle
        slice_offset = max(0, start_line - total_rows_scanned)
        sliced_batch = batch.slice(offset=slice_offset)
        
        # Eğer `num_lines` belirtilmişse, sadece gereken sayıda satırı al
        if num_lines is not None:
            rows_needed = num_lines - rows_collected
            if sliced_batch.num_rows > rows_needed:
                sliced_batch = sliced_batch.slice(length=rows_needed)
                
        # Parçayı Pandas DataFrame'e dönüştür ve listeye ekle
        df_batch = sliced_batch.to_pandas()
        dataframes.append(df_batch)
        rows_collected += df_batch.shape[0]
        total_rows_scanned += batch_rows

    # Eğer hiç veri okunmadıysa boş bir DataFrame döndür
    if not dataframes:
        return pd.DataFrame(columns=["en", "tr"])
    
    # Tüm parçaları birleştirerek tek bir DataFrame oluştur
    return pd.concat(dataframes, ignore_index=True)


# --- METİN NORMALLEŞTİRME FONKSİYONU ---
def normalize_text(text):
    """
    Bir metni, makine çevirisi modeli (MT5) için daha uygun hale getirmek amacıyla temizler ve standartlaştırır.
    Bu işlem, modelin anlamasını zorlaştırabilecek gereksiz yapıları (HTML etiketleri, URL'ler vb.) kaldırırken,
    anlam bütünlüğü için önemli olan unsurları (büyük harfler, emojiler, argo kelimeler vb.) korur.

    Args:
        text (str): Normalleştirilecek metin.

    Returns:
        str: Temizlenmiş ve normalleştirilmiş metin.
    """
    # 1. Adım: Bozuk Unicode ve Kodlama Hatalarını Düzeltme
    # Örn: "GÃ¼naydÄ±n" -> "Günaydın"
    text = ftfy.fix_text(text)

    # 2. Adım: HTML Etiketlerini Temizleme
    # Örn: "<p>Merhaba</p>" -> "Merhaba"
    text = BeautifulSoup(text, "html.parser").get_text()

    # 3. Adım: URL'leri (İnternet Adreslerini) Temizleme
    # Örn: "Daha fazla bilgi için https://example.com adresini ziyaret edin." -> "Daha fazla bilgi için  adresini ziyaret edin."
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # 4. Adım (İsteğe Bağlı): Aşırı Tekrarlanan Noktalama İşaretlerini Azaltma
    # Bu adım şimdilik devre dışı bırakılmıştır.
    # Örn: "Harika!!!" -> "Harika!"
    # text = re.sub(r'([!?.])\1+', r'\1', text)

    # 5. Adım: Düzensiz Boşlukları ve Kontrol Karakterlerini Temizleme
    # Birden fazla boşluğu tek boşluğa indirir ve metin içinde görünmeyen kontrol karakterlerini siler.
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[\x00-\x1f\x7f-' + '\x9f]', '', text)

    # 6. Adım: Metnin Başındaki ve Sonundaki Boşlukları Temizleme
    text = text.strip()
    
    return text

# --- ANA İŞLEM FONKSİYONU ---
def main():
    """
    Veri ön işleme sürecini yöneten ana fonksiyon.
    1. Yapılandırma dosyasını (config.yaml) okur.
    2. Veri setini yükler.
    3. Veriyi temizler (eksik, yinelenen satırlar).
    4. Metin normalizasyonu uygular.
    5. Veri setini eğitim ve test olarak ikiye ayırır.
    6. İşlenmiş verileri Parquet formatında kaydeder.
    """
    # Yapılandırma dosyasını oku
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    logging.info("Veri seti yükleniyor...")
    # `load_parquet_dataset` fonksiyonunu kullanarak veriyi yükle
    df = load_parquet_dataset(
        config['dataset']['path'], 
        start_line=config['dataset']['start_line'], 
        num_lines=config['dataset']['max_lines']
    )
    logging.info(f"Toplam {len(df)} satır veri yüklendi.")
    
    # Eksik verileri ('en' veya 'tr' sütununda boş olan satırları) temizle
    df.dropna(subset=['en', 'tr'], inplace=True)
    # Boş metin içeren satırları temizle
    df = df[df['en'].str.len() > 0 & df['tr'].str.len() > 0]
    logging.info(f"Eksik veriler temizlendikten sonra {len(df)} satır kaldı.")

    logging.info("Metin normalizasyonu uygulanıyor...")
    # 'en' ve 'tr' sütunlarındaki her bir metne `normalize_text` fonksiyonunu uygula
    df['en'] = df['en'].apply(normalize_text)
    df['tr'] = df['tr'].apply(normalize_text)

    # Dil çifti olarak tamamen aynı olan satırları kaldır (yinelenenleri temizle)
    logging.info("Yinelenen veriler kaldırılıyor...")
    df.drop_duplicates(subset=['en', 'tr'], keep='first', inplace=True)
    logging.info(f"Yinelenenler kaldırıldıktan sonra {len(df)} satır kaldı.")
    logging.info("-" * 40)

    logging.info("Eğitim ve test verileri ayrılıyor...")
    # Veri setini, yapılandırma dosyasında belirtilen oranda eğitim ve test setlerine ayır
    train_df, test_df = train_test_split(
        df,
        test_size=config['dataset']['test_size'],
        random_state=config['dataset']['random_state']
    )

    logging.info("İşlenmiş veriler kaydediliyor...")
    # İşlenmiş verilerin kaydedileceği klasörü al ve yoksa oluştur
    output_dir = config['artifacts']['processed_data_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    # Eğitim ve test DataFrame'lerini Parquet formatında kaydet
    train_df.to_parquet(os.path.join(output_dir, "train.parquet"))
    test_df.to_parquet(os.path.join(output_dir, "test.parquet"))
    
    logging.info(f"Veriler başarıyla '{output_dir}' dizinine kaydedildi.")

# --- SCRIPT'İN BAŞLANGIÇ NOKTASI ---
if __name__ == "__main__":
    # Bu script doğrudan çalıştırıldığında `main()` fonksiyonunu çağır.
    # Eğer başka bir script tarafından import edilirse bu blok çalışmaz.
    main()
