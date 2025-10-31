# Gerekli kütüphanelerin import edilmesi
import os  # İşletim sistemiyle ilgili işlemler için (dosya yolları, klasör oluşturma vb.)
import yaml  # YAML formatındaki yapılandırma dosyalarını okumak için
import gc  # Garbage Collector: Bellek yönetimi ve temizliği için
import numpy as np  # Sayısal hesaplamalar ve dizi işlemleri için
import pandas as pd  # Veri işleme ve analizi için
import torch  # Derin öğrenme ve tensör işlemleri için temel kütüphane
from torch.utils.data import Dataset, DataLoader  # PyTorch'ta veri setleri ve veri yükleyicileri oluşturmak için
from transformers import (
    MT5Tokenizer,  # MT5 modeli için tokenizer (metni sayısallaştırıcı)
    T5ForConditionalGeneration,  # Koşullu metin üretimi için T5/MT5 modeli
    BitsAndBytesConfig,  # 4-bit kuantizasyon ayarları için yapılandırma sınıfı
    get_cosine_schedule_with_warmup,  # Öğrenme oranını (learning rate) zamanla ayarlamak için bir zamanlayıcı
)
from peft import (
    LoraConfig,  # LoRA (Low-Rank Adaptation) ayarları için yapılandırma sınıfı
    get_peft_model,  # PEFT (Parameter-Efficient Fine-Tuning) yöntemlerini modele uygulamak için
    prepare_model_for_kbit_training,  # Modeli k-bit (örn: 4-bit) eğitimi için hazırlayan fonksiyon
)
from bitsandbytes.optim import PagedAdamW  # Bellek verimli AdamW optimizer'ı (özellikle QLoRA ile kullanılır)
from tqdm import tqdm  # Döngülerde ilerleme çubuğu göstermek için
import matplotlib.pyplot as plt  # Grafikler ve görselleştirmeler oluşturmak için
import matplotlib.ticker as ticker  # Grafik eksenlerindeki işaretleri (ticks) formatlamak için
import logging  # Olayları ve hataları kaydetmek (loglamak) için
from datetime import datetime  # Tarih ve saat bilgileriyle çalışmak için

# --- LOGLAMA AYARLARI ---
log_dir = "logs/train"  # Eğitim loglarının kaydedileceği klasör
os.makedirs(log_dir, exist_ok=True)
log_filename = f"{log_dir}/train_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(log_filename),
                        logging.StreamHandler()
                    ])

# --- PYTORCH VERİ SETİ SINIFI ---
class TranslationDataset(Dataset):
    '''
    Pandas DataFrame'ini, PyTorch DataLoader tarafından kullanılabilecek bir veri setine dönüştürür.
    Her bir veri örneğini (`__getitem__`) modelin anlayacağı formata getirir.
    '''
    def __init__(self, tokenizer, src_texts, tgt_texts, max_len=128):
        '''
        Args:
            tokenizer: Metinleri sayısallaştırmak için kullanılacak tokenizer.
            src_texts (list): Kaynak metinlerin (İngilizce) listesi.
            tgt_texts (list): Hedef metinlerin (Türkçe) listesi.
            max_len (int): Tokenizer'ın metinleri kırpacacağı veya dolduracağı maksimum uzunluk.
        '''
        self.tokenizer = tokenizer
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.max_len = max_len

    def __len__(self):
        '''Veri setindeki toplam örnek sayısını döndürür.'''
        return len(self.src_texts)

    def __getitem__(self, idx):
        '''
        Veri setinden belirli bir indeksteki (idx) tek bir örneği alır ve işler.
        '''
        # MT5 modeline görevin ne olduğunu belirtmek için kaynak metnin başına bir ön ek eklenir.
        src_text = "translate English to Turkish: " + self.src_texts[idx]
        tgt_text = self.tgt_texts[idx]

        # Kaynak metni tokenizer ile sayısallaştır
        source = self.tokenizer(
            src_text,
            truncation=True,  # Metin max_len'den uzunsa kırp
            padding="max_length",  # Metin max_len'den kısaysa doldur
            max_length=self.max_len,
            return_tensors="pt"  # PyTorch tensörleri olarak döndür
        )
        # Hedef metni tokenizer ile sayısallaştır
        target = self.tokenizer(
            tgt_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        # Modelin kayıp (loss) hesaplaması için etiketleri (labels) hazırla.
        # Doldurma (padding) token'larının ID'leri -100 olarak ayarlanır, böylece kayıp hesaplamasına dahil edilmezler.
        labels = target["input_ids"]
        labels[labels == self.tokenizer.pad_token_id] = -100

        # Modelin girdisi olarak hazırlanmış sözlük yapısı
        return {
            "input_ids": source["input_ids"].squeeze(),  # Gereksiz boyutları kaldır
            "attention_mask": source["attention_mask"].squeeze(),
            "labels": labels.squeeze(),
        }

# --- ANA EĞİTİM FONKSİYONU ---
def main():
    '''
    Model eğitim sürecini yöneten ana fonksiyon.
    Yapılandırmayı yükler, veriyi hazırlar, modeli kurar, eğitimi başlatır ve sonuçları kaydeder.
    '''
    # Yapılandırma dosyasını (config.yaml) oku
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Cihazı belirle: CUDA destekli GPU varsa ve config'de izin verilmişse GPU, yoksa CPU kullan.
    DEVICE = "cuda" if torch.cuda.is_available() and config['system']['gpu'] else "cpu"
    logging.info(f"Kullanılacak Cihaz: {DEVICE}")

    # Önceden işlenmiş eğitim ve doğrulama verilerini yükle
    logging.info("İşlenmiş veriler yükleniyor...")
    processed_data_dir = config['artifacts']['processed_data_dir']
    train_df = pd.read_parquet(os.path.join(processed_data_dir, "train.parquet"))
    val_df = pd.read_parquet(os.path.join(processed_data_dir, "test.parquet"))

    # Modeline uygun tokenizer'ı Hugging Face'den yükle
    tokenizer = MT5Tokenizer.from_pretrained(config['model_adi'])

    MAX_LEN = config['training']['max_len']

    # Eğitim ve doğrulama için TranslationDataset nesnelerini oluştur
    train_dataset = TranslationDataset(tokenizer, train_df["en"].tolist(), train_df["tr"].tolist(), MAX_LEN)
    val_dataset = TranslationDataset(tokenizer, val_df["en"].tolist(), val_df["tr"].tolist(), MAX_LEN)

    # Veri yükleyicilerini (DataLoader) oluştur. Bu, veriyi batch'lere ayırır ve eğitimi hızlandırır.
    train_loader = DataLoader(train_dataset, batch_size=config['training']['train_batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['train_batch_size']*2, shuffle=False, num_workers=4, pin_memory=True)

    logging.info(f"Eğitim için {len(train_dataset)}, doğrulama için {len(val_dataset)} örnek oluşturuldu.")

    # --- 4-BIT KUANTİZASYON (QLoRA) AYARLARI ---
    # BitsAndBytesConfig ile modeli 4-bit hassasiyetinde yüklemek için yapılandırma.
    # Bu, modelin bellek kullanımını önemli ölçüde azaltır.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,  # Modeli 4-bit olarak yükle
        bnb_4bit_quant_type="nf4",  # Kuantizasyon tipi: Normal Float 4 (NF4)
        bnb_4bit_compute_dtype=torch.bfloat16,  # Hesaplama sırasında kullanılacak veri tipi
        bnb_4bit_use_double_quant=True,  # Daha fazla bellek tasarrufu için çift kuantizasyon
    )

    # Önceden eğitilmiş temel modeli (MT5) 4-bit kuantizasyon ile yükle
    logging.info("Temel model 4-bit olarak yükleniyor...")
    model = T5ForConditionalGeneration.from_pretrained(
        config['model_adi'],
        quantization_config=bnb_config,
        device_map="auto"  # Modeli mevcut cihazlara (GPU/CPU) otomatik olarak dağıt
    )
    logging.info("Model yüklendi.")

    # Modeli k-bit (4-bit) eğitimi için hazırla
    model = prepare_model_for_kbit_training(model)

    # --- LoRA (LOW-RANK ADAPTATION) AYARLARI ---
    # LoRA, modelin tüm parametrelerini güncellemek yerine sadece küçük "adaptör" katmanlarını eğitir.
    # Bu, ince ayar (fine-tuning) sürecini çok daha verimli hale getirir.
    lora_config = LoraConfig(
        r=config['qlora']['lora_rank'],  # LoRA matrislerinin rank'ı (düşük olması daha az parametre demek)
        lora_alpha=config['qlora']['lora_alpha'],  # LoRA ölçekleme faktörü
        target_modules=config['qlora']['target_modules'],  # LoRA'nın uygulanacağı katmanlar (genellikle attention katmanları)
        lora_dropout=config['qlora']['lora_dropout'],  # LoRA katmanlarındaki dropout oranı
        bias="none",  # Bias'ların nasıl eğitileceği ('none' en yaygın olanıdır)
        task_type="SEQ_2_SEQ_LM"  # Görev tipi (Sequence-to-Sequence Language Modeling)
    )

    # LoRA yapılandırmasını modele uygula
    model = get_peft_model(model, lora_config)
    # Modeldeki eğitilebilir parametre sayısını yazdır (sadece LoRA parametreleri)
    model.print_trainable_parameters()

    # --- OPTIMIZER VE ZAMANLAYICI ---
    # PagedAdamW, 4-bit kuantize edilmiş modellerle verimli çalışan bir optimizer'dır.
    optimizer = PagedAdamW(model.parameters(), lr=config['training']['lr'])
    
    # Toplam optimizasyon adımı sayısını hesapla
    total_optimizer_steps = (len(train_loader) // config['training']['gradient_accumulation_steps']) * config['training']['epochs']
    # Isınma (warmup) adımı sayısını belirle (genellikle toplam adımların %10'u)
    warmup_steps = int(0.1 * total_optimizer_steps)

    # Öğrenme oranını (learning rate) ayarlamak için kosinüs zamanlayıcısı oluştur.
    # Başlangıçta öğrenme oranını yavaşça artırır (warmup), sonra kosinüs eğrisini takip ederek azaltır.
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_optimizer_steps
    )

    # Eğitim ve doğrulama kayıp (loss) değerlerini saklamak için listeler
    train_losses = []
    val_losses = []

    logging.info("4-bit QLoRA Eğitimi başlıyor...")

    # --- EĞİTİM DÖNGÜSÜ ---
    for epoch in range(config['training']['epochs']):
        model.train()  # Modeli eğitim moduna al
        total_loss = 0
        # Eğitim veri yükleyicisi için bir ilerleme çubuğu oluştur
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']} [Eğitim]")

        for i, batch in enumerate(loop):
            # Veri batch'ini ilgili cihaza (GPU/CPU) gönder
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            # Modeli batch ile çalıştır ve çıktıları al
            outputs = model(**batch)
            # Çıktılardan kayıp (loss) değerini al
            loss = outputs.loss
            
            # Gradyan biriktirme (gradient accumulation) için kaybı ölçekle
            loss = loss / config['training']['gradient_accumulation_steps']
            
            # Geriye yayılım (backpropagation) ile gradyanları hesapla
            loss.backward()
            
            # Gerçek (normalize edilmemiş) kayıp değerini hesapla ve ilerleme çubuğunda göster
            unnormalized_loss = loss.item() * config['training']['gradient_accumulation_steps']
            total_loss += unnormalized_loss
            loop.set_postfix(loss=unnormalized_loss)

            # Belirli sayıda adımda bir (gradient_accumulation_steps) optimizer'ı çalıştır
            if (i + 1) % config['training']['gradient_accumulation_steps'] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradyan patlamasını önlemek için kırpma
                optimizer.step()  # Parametreleri güncelle
                scheduler.step()  # Öğrenme oranını güncelle
                optimizer.zero_grad()  # Gradyanları sıfırla

        # Epoch sonu ortalama eğitim kaybını hesapla ve kaydet
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        logging.info(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f}")

        # --- DOĞRULAMA DÖNGÜSÜ ---
        model.eval()  # Modeli değerlendirme moduna al
        val_loss = 0
        with torch.no_grad():  # Gradyan hesaplamasını devre dışı bırak
            val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']} [Doğrulama]")
            for batch in val_loop:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                outputs = model(**batch)
                val_loss += outputs.loss.item()

        # Ortalama doğrulama kaybını hesapla ve kaydet
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        logging.info(f"Validation Loss: {avg_val_loss:.4f}")

        # Epoch sonunda GPU belleğini temizle
        logging.info(f"Epoch {epoch+1} sonunda bellek temizleniyor...")
        torch.cuda.empty_cache()
        gc.collect()

    logging.info("\nEğitim tamamlandı!")

    # --- MODELİ VE SONUÇLARI KAYDETME ---
    # Kaydedilecek modelin adını config dosyasından formatla
    output_dir = config['system']['output_dir'].format(
        model_mimarisi=config['model_mimarisi'],
        model_teknigi=config['model_teknigi'],
        proje_adi=config['proje_adi'],
        veri_seti=config['veri_seti'],
        versiyon=config['versiyon']
    )
    model_output_dir = os.path.join("models", output_dir)

    # Sadece eğitilen LoRA adaptör ağırlıklarını ve tokenizer'ı kaydet
    logging.info(f"\nSADECE LoRa adaptör ağırlıkları '{model_output_dir}' dizinine kaydediliyor...")
    model.save_pretrained(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)
    logging.info("Kaydetme işlemi tamamlandı.")

    # --- KAYIP GRAFİĞİNİ OLUŞTURMA VE KAYDETME ---
    results_dir = config['artifacts']['results_dir']
    os.makedirs(results_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(train_losses) + 1), train_losses, label='Eğitim Kaybı (Training Loss)', color='blue', linewidth=2, linestyle='-')
    ax.plot(range(1, len(val_losses) + 1), val_losses, label='Doğrulama Kaybı (Validation Loss)', color='red', linewidth=2, linestyle='--')
    ax.set_title(f'Model Kayıp Grafiği ({output_dir})', fontsize=16, fontweight='bold')
    ax.set_ylabel('Kayıp Değeri (Loss Value)', fontsize=12)
    ax.set_xlabel('Epok Sayısı (Epoch Number)', fontsize=12)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True)) # X eksenini tamsayı yapmak için
    ax.legend(loc='upper right', fontsize=10, frameon=True, shadow=True)
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(os.path.join(results_dir, 'loss_graph.png'))
    logging.info(f"Grafik '{results_dir}' dizinine kaydedildi.")

# --- SCRIPT'İN BAŞLANGIÇ NOKTASI ---
if __name__ == "__main__":
    main()
