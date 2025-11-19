# Gerekli kütüphanelerin import edilmesi
import os  # İşletim sistemiyle ilgili işlemler için (dosya yolları, dizin oluşturma vb.)
import pandas as pd  # Veri işleme ve analizi için güçlü bir kütüphane, özellikle DataFrame yapıları için
import argparse  # Komut satırı argümanlarını işlemek için
import logging  # Olayları ve hataları kaydetmek (loglamak) için
from datetime import datetime  # Tarih ve saat bilgileriyle çalışmak için
import torch  # PyTorch, derin öğrenme ve tensör hesaplamaları için
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, util  # Cümle gömme (embedding) modelleri için
import numpy as np  # Sayısal hesaplamalar ve diziler için
import yaml  # YAML formatındaki yapılandırma dosyalarını okumak için
import gc  # Garbage Collector, belleği manuel olarak temizlemek için
from itertools import islice  # Dosyaları satır satır verimli bir şekilde okumak için

# --- LOGLAMA AYARLARI ---
# Logların kaydedileceği dizini belirle
log_dizini = "logs/convert_to_parquet"
# Eğer bu dizin mevcut değilse oluştur
os.makedirs(log_dizini, exist_ok=True)
# Her çalıştırmada benzersiz bir log dosyası adı oluştur (tarih ve saat içerir)
log_dosya_adi = f"{log_dizini}/convert_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
# Loglama yapılandırmasını ayarla
logging.basicConfig(
    level=logging.INFO,  # Kaydedilecek minimum log seviyesi (INFO ve üstü)
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log mesajlarının formatı
    handlers=[
        logging.FileHandler(log_dosya_adi),  # Logları dosyaya yaz
        logging.StreamHandler()  # Logları konsola (ekrana) yaz
    ])

def dosya_satir_sayisini_al(dosya_yolu):
    """Bir metin dosyasının toplam satır sayısını, dosyayı belleğe tamamen yüklemeden sayar.
    Bu, çok büyük dosyalar için bellek verimliliği sağlar."""
    try:
        with open(dosya_yolu, 'r', encoding='utf-8') as f:
            for i, _ in enumerate(f):
                pass
        return i + 1
    except NameError:
        return 0 # Dosya boşsa
    except Exception as e:
        logging.error(f"'{dosya_yolu}' dosyası okunurken bir hata oluştu: {e}")
        return 0


def parcayi_isle(df, model, kaynak_dil, hedef_dil, benzerlik_esigi, cihaz):
    """
    Veri çerçevesinin (DataFrame) bir parçasını (chunk) işler ve temizler.
    Kaynak ve hedef cümleler arasındaki anlamsal benzerliği hesaplar ve düşük skorlu olanları eler.
    """
    # Cümleleri gömme (embedding) vektörlerine dönüştür. Bu, cümlelerin anlamsal içeriğini temsil eden sayısal vektörlerdir.
    kaynak_gommeleri = model.encode(df[kaynak_dil].tolist(), convert_to_tensor=True, show_progress_bar=True, device=cihaz)
    hedef_gommeleri = model.encode(df[hedef_dil].tolist(), convert_to_tensor=True, show_progress_bar=True, device=cihaz)

    # İki gömme seti arasındaki kosinüs benzerliğini verimli bir şekilde hesapla.
    # Bu yöntem, büyük bir ara matris oluşturmadan doğrudan eşleşen çiftlerin benzerliğini hesaplar.
    benzerlik_skorlari = F.cosine_similarity(kaynak_gommeleri, hedef_gommeleri, dim=1)
    
    # Hesaplanan benzerlik skorlarını DataFrame'e yeni bir sütun olarak ekle.
    df['benzerlik_skoru'] = benzerlik_skorlari.cpu().numpy()
    
    # Belirlenen benzerlik eşiğinden daha yüksek skora sahip olan satırları filtrele.
    temizlenmis_df = df[df['benzerlik_skoru'] >= benzerlik_esigi].copy()

    # Bellek temizliği: Artık ihtiyaç duyulmayan büyük tensörleri bellekten kaldır.
    del kaynak_gommeleri, hedef_gommeleri, benzerlik_skorlari
    if cihaz == 'cuda':
        torch.cuda.empty_cache()  # GPU belleğini temizle
    gc.collect()  # Python'un çöp toplayıcısını çalıştır
    
    return temizlenmis_df

def metin_dosyalarini_parqueta_donustur(kaynak_dosya, hedef_dosya, cikti_dizini, kaynak_dil, hedef_dil, temizle, benzerlik_esigi, cihaz, veri_seti_adi, parca_boyutu):
    """
    Paralel metin dosyalarını (kaynak ve hedef) okur, isteğe bağlı olarak temizler ve Parquet formatında kaydeder.
    Bu işlem, büyük dosyalarla çalışırken belleği verimli kullanmak için parçalar (chunks) halinde yapılır.
    """
    # Kaynak ve hedef dosyaların var olup olmadığını kontrol et
    if not os.path.exists(kaynak_dosya):
        raise FileNotFoundError(f"Kaynak dosya bulunamadı: {kaynak_dosya}")
    if not os.path.exists(hedef_dosya):
        raise FileNotFoundError(f"Hedef dosya bulunamadı: {hedef_dosya}")

    # Çıktı dizinini oluştur (eğer mevcut değilse)
    os.makedirs(cikti_dizini, exist_ok=True)

    # Eğer temizleme işlemi aktifse, anlamsal benzerlik modelini yükle
    model = None
    if temizle:
        logging.info("Veri temizleme için cümle trafo modeli yükleniyor...")
        # Çok dilli, hafif bir model seçimi. Bu model, cümlelerin anlamını vektörlere çevirir.
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2').to(cihaz)
        logging.info("Model yüklendi.")

    # İşlenmiş veri parçalarını depolamak için boş bir liste oluştur
    islenmis_parcalar = []
    
    # Toplam satır sayısını ve işlenecek toplam parça sayısını hesapla
    toplam_satirlar = dosya_satir_sayisini_al(kaynak_dosya)
    toplam_parcalar = (toplam_satirlar + parca_boyutu - 1) // parca_boyutu

    # Dosyaları 'with' bloğu içinde açarak otomatik olarak kapanmalarını sağla
    with open(kaynak_dosya, 'r', encoding='utf-8') as f_kaynak, open(hedef_dosya, 'r', encoding='utf-8') as f_hedef:
        parca_numarasi = 0
        while True:
            parca_numarasi += 1
            # `islice` kullanarak dosyalardan belirli bir sayıda satırı (parca_boyutu kadar) verimli bir şekilde oku
            kaynak_satirlar = list(islice(f_kaynak, parca_boyutu))
            hedef_satirlar = list(islice(f_hedef, parca_boyutu))

            # Eğer okunacak satır kalmadıysa döngüden çık
            if not kaynak_satirlar or not hedef_satirlar:
                break

            logging.info(f"Parça işleniyor: {parca_numarasi}/{toplam_parcalar}...")
            
            # Okunan satırlardan bir pandas DataFrame oluştur
            df_parca = pd.DataFrame({
                kaynak_dil: [satir.strip() for satir in kaynak_satirlar],
                hedef_dil: [satir.strip() for satir in hedef_satirlar]
            })

            # Eğer temizleme aktifse ve model yüklendiyse, parçayı işle ve temizle
            if temizle and model:
                df_parca = parcayi_isle(df_parca, model, kaynak_dil, hedef_dil, benzerlik_esigi, cihaz)
            
            # İşlenmiş (veya temizlenmiş) parçayı listeye ekle
            islenmis_parcalar.append(df_parca)

    logging.info("Tüm işlenmiş parçalar birleştiriliyor...")
    # Tüm parçaları tek bir büyük DataFrame'de birleştir
    son_df = pd.concat(islenmis_parcalar, ignore_index=True)
    
    # --- DEĞİŞİKLİK BURADA BAŞLIYOR ---

    # Son DataFrame'i istendiği gibi iki ayrı Parquet dosyasına bölerek kaydet.
    
    # 1. Kaynak Dil Dosyası
    kaynak_cikti_adi = f"{veri_seti_adi}-{kaynak_dil}.parquet"
    kaynak_cikti_yolu = os.path.join(cikti_dizini, kaynak_cikti_adi)
    logging.info(f"Kaynak dil ({kaynak_dil}) kaydediliyor: {kaynak_cikti_yolu}")
    # Sadece kaynak dil sütununu seç ve kaydet
    son_df[[kaynak_dil]].to_parquet(kaynak_cikti_yolu, index=False)

    # 2. Hedef Dil Dosyası
    hedef_cikti_adi = f"{veri_seti_adi}-{hedef_dil}.parquet"
    hedef_cikti_yolu = os.path.join(cikti_dizini, hedef_cikti_adi)
    logging.info(f"Hedef dil ({hedef_dil}) kaydediliyor: {hedef_cikti_yolu}")
    # Sadece hedef dil sütununu seç ve kaydet
    son_df[[hedef_dil]].to_parquet(hedef_cikti_yolu, index=False)

    logging.info(f"Dosyalar başarıyla dönüştürüldü ve iki ayrı dosyaya kaydedildi.")
    logging.info(f"Başlangıçtaki satır sayısı: {toplam_satirlar}, Sonraki (temizlenmiş) satır sayısı: {len(son_df)}")

    # --- DEĞİŞİKLİK BURADA BİTİYOR ---


def main():
    """
    Betiğin ana giriş noktası. Komut satırı argümanlarını işler, yapılandırmayı yükler ve dönüştürme işlemini başlatır.
    """
    # Komut satırı argümanlarını tanımlamak için bir ayrıştırıcı (parser) oluştur
    ayristirici = argparse.ArgumentParser(description="Paralel metin dosyalarını Parquet formatına dönüştürür.")
    ayristirici.add_argument("kaynak_dil", type=str, help="Kaynak dil (örn: 'en')")
    ayristirici.add_argument("hedef_dil", type=str, help="Hedef dil (örn: 'tr')")
    ayristirici.add_argument("veri_seti_adi", type=str, help="Veri setinin adı (örn: 'lainchan')")
    ayristirici.add_argument("--girdi_dizini", type=str, default=".", help="Girdi metin dosyalarını içeren dizin.")
    ayristirici.add_argument("--cikti_dizini", type=str, default="datasets", help="Çıktı Parquet dosyalarının kaydedileceği dizin.")
    argumanlar = ayristirici.parse_args()

    # Yapılandırma dosyasını (config.yaml) oku
    with open("config.yaml", "r") as f:
        yapilandirma = yaml.safe_load(f)

    # Yapılandırmadan veri temizleme ayarlarını al
    temizleme_yapilandirmasi = yapilandirma['dataset']['cleaning']
    temizle = temizleme_yapilandirmasi['enabled']
    benzerlik_esigi = temizleme_yapilandirmasi['similarity_threshold']
    gpu_kullan = temizleme_yapilandirmasi['gpu']
    # Parça boyutunu yapılandırmadan al, eğer belirtilmemişse varsayılan olarak 10000 kullan
    parca_boyutu = temizleme_yapilandirmasi.get('chunk_size', 10000)

    # Eğer CUDA destekli bir GPU varsa ve yapılandırmada izin verilmişse 'cuda' kullan, yoksa 'cpu'
    cihaz = "cuda" if torch.cuda.is_available() and gpu_kullan else "cpu"
    logging.info(f"Kullanılacak cihaz: {cihaz}")

    # Girdi ve çıktı dosya yollarını oluştur
    kaynak_dosya = os.path.join(argumanlar.girdi_dizini, f"{argumanlar.veri_seti_adi}-{argumanlar.kaynak_dil}.txt")
    hedef_dosya = os.path.join(argumanlar.girdi_dizini, f"{argumanlar.veri_seti_adi}-{argumanlar.hedef_dil}.txt")
    cikti_dizini = os.path.join(argumanlar.cikti_dizini, argumanlar.veri_seti_adi)

    # Ana dönüştürme fonksiyonunu çağır
    metin_dosyalarini_parqueta_donustur(kaynak_dosya, hedef_dosya, cikti_dizini, argumanlar.kaynak_dil, argumanlar.hedef_dil, temizle, benzerlik_esigi, cihaz, argumanlar.veri_seti_adi, parca_boyutu)

# Bu betik doğrudan çalıştırıldığında `main()` fonksiyonunu çağır
if __name__ == "__main__":
    main()