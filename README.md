# Unwired Translate 🌍

**Unwired Translate**, Google'ın **mT5 (Multilingual T5)** modelini temel alan, **16-bit LoRA** tekniği ile eğitilmiş ve **8-bit CTranslate2** ile optimize edilmiş, modern bir **Flet** arayüzü sunan açık kaynaklı bir makine çevirisi projesidir.

Bu proje; veri toplama (scraping), anlamsal veri temizleme (semantic cleaning), model eğitimi ve masaüstü uygulaması geliştirme süreçlerinin tamamını kapsayan uçtan uca (end-to-end) bir çözümdür.

---

## 🚀 Performance & Experiments (Latest Run)

Modelin eğitim süreçleri, hiperparametre optimizasyonu ve detaylı performans metrikleri **Kaggle** üzerinde şeffaf bir şekilde dökümante edilmiştir. mT5-small gibi küçük modellerde kararlılığı artırmak için eğitim **16-bit Float16** hassasiyetinde yapılırken, son kullanıcıya sunulan model **int8 (8-bit)** quantization ile optimize edilmiştir.

📊 **Kaggle Notebook & Eğitim Logları:** [Kaggle Notebook](https://www.kaggle.com/code/n4yuc4/t5-model-based-machine-translation)

---

## 🛠 Features

* **Advanced NLP Pipeline:**
    * **Semantic Cleaning:** `SentenceTransformers` kullanılarak yapılan anlamsal benzerlik analizi ile düşük kaliteli çeviri çiftlerinin elenmesi.
    * **Data Preprocessing:** Parquet formatında optimize edilmiş veri yükleme ve temizleme süreçleri.

* **Efficient Fine-Tuning & Optimization:**
    * **16-bit LoRA Training:** Model kararlılığı için 16-bit Float16/Mixed-Precision eğitimi.
    * **8-bit CTranslate2 Inference:** Çıkarım (inference) aşamasında int8 quantization ile maksimum hız ve minimum CPU/GPU kullanımı.

* **Modern GUI (Flet):**
    * **Responsive Tasarım:** Masaüstü ve mobil ekran boyutlarına tam uyum.
    * **Akıllı Metin Düzeltme:** `SymSpell` entegrasyonu ile "Bunu mu demek istediniz?" önerileri.
    * **Gelişmiş Geçmiş Yönetimi:** Tıklanabilir geçmiş öğeleri ile hızlı tekrar çeviri.
    * **Karanlık/Aydınlık mod** ve 12+ dil desteği.

---

## 📂 Project Structure

```bash
Unwired-Translate/
├── app/
│   ├── main.py              # Flet tabanlı GUI uygulaması
│   ├── locales/             # Arayüz dil dosyaları (JSON)
│   ├── assets/dictionaries/ # Yazım denetimi sözlükleri
│   └── utils/               # Spell Checker, History, Localization, Settings
├── scripts/
│   ├── train.py             # 16-bit LoRA eğitim ve CTranslate2 dönüşüm betiği
│   ├── predict.py           # 8-bit CTranslate2 tabanlı hızlı inference betiği
│   ├── eval.py              # METEOR skoru hesaplama
│   ├── data_preprocessing.py # Veri birleştirme ve train/test ayırma
│   └── clean_and_convert...py # Veri temizleme ve Parquet formatına dönüştürme
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
Yeni bir model eğitmek, adaptörleri birleştirmek ve 8-bit CTranslate2 formatına dönüştürmek için:
```bash
python scripts/train.py
```

### 3. Veri Seti Hazırlama
```bash
# 1. Ham metinleri temizleme ve Parquet formatına dönüştürme
# Kullanım: python scripts/clean_and_convert_to_parquet.py <kaynak_dil> <hedef_dil> <veri_seti_adi>
python scripts/clean_and_convert_to_parquet.py en tr my_dataset

# 2. Farklı veri setlerini birleştirme ve train/test setlerini oluşturma
python scripts/data_preprocessing.py
```

### 4. CLI Üzerinden Çeviri ve Değerlendirme
```bash
# Tekil çeviri testi
python scripts/predict.py "Hello, how are you?" --src English --tgt Turkish

# Model performansını METEOR skoru ile test etme
python scripts/eval.py
```

---

## 🔧 Configuration (`config.yaml`)

Proje modüler bir yapıdadır ve tüm ayarlar `config.yaml` üzerinden yönetilir:
```yaml
model_name: "google/mt5-small"
training:
  precision: "16-mixed" # 16-bit hassasiyet
  epochs: 5
  learning_rate: 0.0001
```

---

## 🛡️ Git Ignore & Local Files
Aşağıdaki dizinler çalışma anında üretilir ve repo boyutunu korumak için `.gitignore` kapsamındadır:
* `/models/`: Eğitilmiş ve 8-bit'e dönüştürülmüş CTranslate2 model dosyaları.
* `/artifacts/`: Uygulama ayarları (`app_settings.json`) ve çeviri geçmişi (`translation_history.json`).
* `/datasets/`: Ham ve işlenmiş eğitim verileri.
* `logs/`: Uygulama ve eğitim logları.

---

## 🤝 Contributing
Katkılarınızı bekliyoruz! Lütfen bir "Issue" açarak veya "Pull Request" göndererek projeye destek olun.

## 📜 License
Bu proje [MIT License](LICENSE) altında lisanslanmıştır.

---
**Developed by [Nazmi Yücel Çan](https://github.com/N4YuC4)**