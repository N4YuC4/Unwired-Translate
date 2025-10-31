# Gerekli kütüphanelerin import edilmesi
import os  # İşletim sistemiyle ilgili işlemler için
import yaml  # YAML formatındaki yapılandırma dosyalarını okumak için
import json  # JSON formatındaki verileri işlemek için (sonuçları kaydetmek vb.)
import pandas as pd  # Veri işleme ve analizi için
import torch  # Derin öğrenme ve tensör işlemleri için
import evaluate  # Hugging Face'in model değerlendirme metriklerini yüklemek için (örn: METEOR)
from tqdm import tqdm  # Döngülerde ilerleme çubuğu göstermek için
from transformers import (
    MT5Tokenizer,  # MT5 modeli için tokenizer
    T5ForConditionalGeneration,  # Koşullu metin üretimi için T5/MT5 modeli
    BitsAndBytesConfig,  # 4-bit kuantizasyon ayarları için
)
from peft import PeftModel  # PEFT (LoRA) ile eğitilmiş adaptörleri yüklemek için
from torch.utils.data import Dataset, DataLoader  # PyTorch'ta veri setleri ve yükleyicileri için
import logging  # Olayları ve hataları kaydetmek için
from datetime import datetime  # Tarih ve saat bilgileri için

# --- LOGLAMA AYARLARI ---
# Değerlendirme süreci için loglama sistemi kurulumu
log_dir = "logs/evaluate"
os.makedirs(log_dir, exist_ok=True)
log_filename = f"{log_dir}/evaluate_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(log_filename),
                        logging.StreamHandler()
                    ])

# --- PYTORCH VERİ SETİ SINIFI ---
# Not: Bu sınıf, train.py'deki ile aynıdır. Kodu tekrar kullanmak yerine ortak bir modüle taşınabilir.
class TranslationDataset(Dataset):
    """
    Pandas DataFrame'ini, PyTorch DataLoader tarafından kullanılabilecek bir veri setine dönüştürür.
    """
    def __init__(self, tokenizer, src_texts, tgt_texts, max_len=128):
        self.tokenizer = tokenizer
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.max_len = max_len

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, idx):
        src_text = "translate English to Turkish: " + self.src_texts[idx]
        tgt_text = self.tgt_texts[idx]

        # Kaynak metni sayısallaştır
        source = self.tokenizer(
            src_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        # Hedef metni sayısallaştır
        target = self.tokenizer(
            tgt_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        # Kayıp hesaplaması dışında tutulacak padding token'larını -100 yap
        labels = target["input_ids"]
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": source["input_ids"].squeeze(),
            "attention_mask": source["attention_mask"].squeeze(),
            "labels": labels.squeeze(),
        }

# --- ANA DEĞERLENDİRME FONKSİYONU ---
def main():
    """
    Eğitilmiş modelin çeviri performansını test seti üzerinde METEOR metriği ile değerlendirir.
    """
    # Yapılandırma dosyasını (config.yaml) oku
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Cihazı belirle (GPU veya CPU)
    DEVICE = "cuda" if torch.cuda.is_available() and config['system']['gpu'] else "cpu"

    logging.info("Test verisi ve model yükleniyor...")
    # İşlenmiş test verisini yükle
    processed_data_dir = config['artifacts']['processed_data_dir']
    test_df = pd.read_parquet(os.path.join(processed_data_dir, "test.parquet"))

    # Değerlendirilecek modelin klasör yolunu config'den oluştur
    output_dir = config['system']['output_dir'].format(
        model_mimarisi=config['model_mimarisi'],
        model_teknigi=config['model_teknigi'],
        proje_adi=config['proje_adi'],
        veri_seti=config['veri_seti'],
        versiyon=config['versiyon']
    )
    model_dir = os.path.join("models", output_dir)

    # --- MODEL YÜKLEME ---
    # Temel modeli 4-bit olarak yüklemek için kuantizasyon yapılandırması
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Temel T5 modelini 4-bit kuantizasyon ile yükle
    model = T5ForConditionalGeneration.from_pretrained(
        config['model_adi'],
        quantization_config=bnb_config,
        device_map="auto"
    )

    # Kaydedilmiş tokenizer'ı ve LoRA adaptörlerini yükle
    tokenizer = MT5Tokenizer.from_pretrained(model_dir)
    model = PeftModel.from_pretrained(model, model_dir)

    # Test veri seti ve yükleyicisini oluştur
    MAX_LEN = config['training']['max_len']
    test_dataset = TranslationDataset(tokenizer, test_df["en"].tolist(), test_df["tr"].tolist(), MAX_LEN)
    test_loader = DataLoader(test_dataset, batch_size=config['training']['train_batch_size']*2, shuffle=False)

    # --- METRİK HESAPLAMA ---
    logging.info("METEOR metriği yükleniyor...")
    # Hugging Face `evaluate` kütüphanesinden METEOR metriğini yükle
    meteor = evaluate.load('meteor')

    all_predictions = []  # Modelin ürettiği tüm çeviriler
    all_references = []   # Gerçek (referans) çeviriler

    logging.info("Çeviriler oluşturuluyor...")
    model.eval()  # Modeli değerlendirme moduna al
    with torch.no_grad():  # Gradyan hesaplamasını kapat
        # Test veri yükleyicisindeki her bir batch için
        for batch in tqdm(test_loader, desc="Çeviriler oluşturuluyor"):
            # Veriyi ilgili cihaza gönder
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            
            # Model ile çeviri üret
            # Not: `unsqueeze(0)` ile batch boyutu 1 olan bir tensör oluşturuluyor.
            # Bu, DataLoader'dan gelen tekil örneklerin doğru işlenmesi için bir düzeltmedir.
            # Daha verimli bir yaklaşım, DataLoader'dan gelen batch'leri doğrudan kullanmak olabilir.
            outputs = model.generate(
                input_ids=batch["input_ids"].unsqueeze(0),
                attention_mask=batch["attention_mask"].unsqueeze(0),
                max_length=MAX_LEN,
                num_beams=4,
                early_stopping=True
            )
            # Üretilen token'ları metne dönüştür
            pred_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            # Referans metinleri (etiketleri) de metne dönüştür
            labels = batch["labels"].unsqueeze(0)
            labels[labels == -100] = tokenizer.pad_token_id # -100'leri tekrar pad token ID'sine çevir
            ref_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)

            # Üretilen ve referans metinleri listelere ekle
            all_predictions.extend(pred_texts)
            all_references.extend(ref_texts)

    # --- SONUÇLARI HESAPLAMA VE KAYDETME ---
    logging.info("METEOR skoru hesaplanıyor...")
    # Toplanan tüm tahminler ve referanslar üzerinden METEOR skorunu hesapla
    results = meteor.compute(predictions=all_predictions, references=all_references)

    logging.info(f"Modelin Ortalama METEOR Skoru: {results['meteor']:.4f}")

    # Hesaplanan metrikleri bir JSON dosyasına kaydet
    results_dir = config['artifacts']['results_dir']
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=4)
    logging.info(f"Metrikler '{results_dir}/metrics.json' dosyasına kaydedildi.")

# --- SCRIPT'İN BAŞLANGIÇ NOKTASI ---
if __name__ == "__main__":
    main()
