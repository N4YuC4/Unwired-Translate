
# Unwired Translate 🌍

**Unwired Translate**, Google'ın **mT5 (Multilingual T5)** modelini temel alan, **4-bit QLoRA** tekniği ile optimize edilmiş ve son kullanıcı için modern bir **Flet** arayüzü sunan açık kaynaklı bir makine çevirisi projesidir.

Bu proje; veri toplama (scraping), anlamsal veri temizleme (semantic cleaning), model eğitimi ve masaüstü uygulaması geliştirme süreçlerinin tamamını kapsayan uçtan uca (end-to-end) bir çözümdür.

---

## 🚀 Performance & Experiments (Latest Run)

Modelin eğitim süreçleri, hiperparametre optimizasyonu ve detaylı performans metrikleri **Kaggle** üzerinde şeffaf bir şekilde dökümante edilmiştir.

📊 **Kaggle Notebook & Eğitim Logları:** [Kaggle Notebook](https://www.kaggle.com/code/n4yuc4/t5-model-based-machine-translation)

---

## 🛠 Features

* **Advanced NLP Pipeline:**
* **Custom Scraper:** `lainchan_veri_kazıma.py` ile hedefe yönelik veri toplama.
* **Semantic Cleaning:** `SentenceTransformers` kullanılarak yapılan anlamsal benzerlik analizi ile düşük kaliteli çeviri çiftlerinin elenmesi.


* **Efficient Fine-Tuning:**
* `bitsandbytes` ve `peft` kütüphaneleri kullanılarak **4-bit Quantization** ve **QLoRA** entegrasyonu.
* Düşük VRAM tüketimi ile yüksek performanslı eğitim.


* **Modern GUI (Flet):**
* Karanlık/Aydınlık mod desteği.
* Çeviri geçmişi yönetimi (History Manager).
* Çoklu dil desteği (Arayüz için 12+ dil).



---

## 📂 Project Structure

```bash
Unwired-Translate/
├── app/
│   ├── main.py              # Flet tabanlı GUI uygulaması
│   ├── locales/             # Arayüz dil dosyaları (JSON)
│   └── utils/               # Yardımcı araçlar (History, Localization, Settings)
├── scripts/
│   ├── train.py             # PyTorch Lightning eğitim döngüsü
│   ├── predict.py           # Model inference ve test betiği
│   ├── eval.py              # METEOR skoru hesaplama
│   ├── lainchan_veri_kazıma.py  # Web scraping aracı
│   └── clean_and_convert...py   # Veri ön işleme ve temizleme
├── config.yaml              # Tüm hiperparametrelerin yönetildiği konfigürasyon
├── requirements.txt         # Proje bağımlılıkları
└── README.md

```

---

## ⚙️ Installation

1. **Repoyu klonlayın:**
```bash
git clone https://github.com/n4yuc4/unwired-translate.git
cd unwired-translate

```


2. **Sanal ortam oluşturun (Önerilen):**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

```


3. **Bağımlılıkları yükleyin:**
```bash
pip install -r requirements.txt

```



---

## 💻 Usage

### 1. Uygulamayı Çalıştırma (GUI)

Eğitilmiş modeli arayüz üzerinden kullanmak için:

```bash
python app/main.py

```

### 2. Model Eğitimi (Training)

Yeni bir model eğitmek için önce `config.yaml` dosyasını düzenleyin, ardından:

```bash
python scripts/train.py

```

### 3. Veri Seti Oluşturma

Kendi veri setinizi oluşturmak için scraper ve temizleme araçlarını kullanabilirsiniz:

```bash
# Veri kazıma
python scripts/lainchan_veri_kazıma.py

# Veriyi temizleme ve Parquet formatına dönüştürme
python scripts/clean_and_convert_to_parquet.py source_lang target_lang dataset_name

```

---

## 🔧 Configuration (`config.yaml`)

Proje modüler bir yapıdadır ve tüm ayarlar `config.yaml` üzerinden yönetilir:

```yaml
model_mimarisi: "mt5-small"
model_teknigi: "4bit-QLoRA"
training:
  epochs: 4
  lr: 0.002
  batch_size: 15
qlora:
  lora_rank: 64
  target_modules: "all-linear"

```

---

## 🤝 Contributing

Katkılarınızı bekliyoruz! Lütfen bir "Issue" açarak veya "Pull Request" göndererek projeye destek olun.

## 📜 License

Bu proje [MIT License](https://www.google.com/search?q=LICENSE) altında lisanslanmıştır.

---

**Developed by [Nazmi Yücel Çan](https://github.com/N4YuC4)**
