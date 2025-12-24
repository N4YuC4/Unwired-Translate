# Unwired Translate

<div align="center">
  <img src="app/assets/unwired-logo.png" alt="Unwired Translate Logo" width="128" height="128">
  <br>
  <h3>Yapay Zeka Destekli, Modern ve Hızlı Masaüstü Çeviri Uygulaması</h3>
  <p>EraneX Technology Tarafından Geliştirilmiştir.</p>
</div>

---

**Unwired Translate**, en son yapay zeka teknolojilerini (Google mT5, LoRA, CTranslate2) modern bir arayüzle (Flet) birleştiren, yüksek performanslı ve kullanıcı dostu bir masaüstü çeviri aracıdır. Düşük donanım kaynaklarında bile hızlı ve akıcı çalışacak şekilde optimize edilmiştir.

## 🚀 Öne Çıkan Özellikler

*   **⚡ Yüksek Performans:** 16-bit LoRA eğitimi ve CTranslate2 (int8 quantization) motoru ile şimşek hızında çeviri.
*   **🎨 Modern Arayüz (Material 3):** Flet ile geliştirilmiş, göz yormayan, şık ve responsive tasarım. Aydınlık ve Karanlık mod desteği.
*   **✨ Akıllı Metin Düzeltme (Spell Checker):**
    *   Yazarken anlık denetim.
    *   **Hibrit Algoritma:** Yazım hatalarını düzeltir ("yanlız" -> "yalnız") ve bitişik kelimeleri ayırır ("yada" -> "ya da").
    *   **Alternatifli Öneriler:** Size en uygun 3 alternatifi sunar.
    *   **Noktalama Koruma:** Metninizin yapısını bozmadan düzeltme yapar.
*   **🌍 Çok Dilli Destek:** Arayüz dili otomatik olarak algılanır ve dinamik olarak çevrilir (Babel entegrasyonu).
*   **📜 Gelişmiş Geçmiş:** Çevirileriniz kaydedilir, tek tıkla geri yüklenebilir.
*   **🛠️ MLOps ve Otomasyon:** Eğitim verilerinden otomatik sözlük oluşturma ve optimize etme araçları.

## 📦 Kurulum

1.  **Depoyu Klonlayın:**
    ```bash
    git clone https://github.com/KullaniciAdiniz/Unwired-Translate.git
    cd Unwired-Translate
    ```

2.  **Sanal Ortam Oluşturun (Önerilen):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/Mac
    # venv\Scripts\activate   # Windows
    ```

3.  **Bağımlılıkları Yükleyin:**
    ```bash
    pip install -r requirements.txt
    ```

## 🛠️ Kullanım

> **Not:** `.gitignore` ayarları gereği `artifacts/`, `datasets/` ve `models/` klasörleri Git deposuna dahil edilmemiştir. Uygulamayı kullanmadan önce kendi veri setinizi hazırlamanız veya eğitilmiş model dosyalarını ilgili dizine yerleştirmeniz gerekmektedir.

### 1. Sözlükleri Oluşturma (İlk Kurulum)
Uygulamanın akıllı yazım denetimi özelliğinin çalışması için frekans sözlüklerinin oluşturulması gerekir. Bu işlem `artifacts/processed_data/` altındaki eğitim verilerini tarar ve optimize edilmiş `.pickle` dosyaları üretir.

```bash
python scripts/generate_frequency_dict.py
```

### 2. Uygulamayı Başlatma
Arayüzü başlatmak için:

```bash
python app/main.py
```

### 3. Model Eğitimi (Geliştiriciler İçin)
Kendi modelinizi eğitmek veya ince ayar yapmak isterseniz:

```bash
python scripts/train.py
```

### 4. Komut Satırı Çevirisi (CLI)
Arayüz olmadan hızlı test yapmak için:

```bash
python scripts/predict.py "Merhaba dünya" --src Turkish --tgt English
```

## 📂 Proje Yapısı

*   `app/`: Uygulama kaynak kodları (UI, mantık, yerelleştirme).
*   `scripts/`: Yapay zeka eğitimi, veri işleme ve yardımcı araçlar.
*   `models/`: Eğitilmiş model dosyaları.
*   `artifacts/`: Eğitim verileri ve işlenmiş dosyalar.
*   `config.yaml`: Proje yapılandırma dosyası.

## 🤝 Katkıda Bulunma

Katkılarınızı bekliyoruz! Lütfen bir "Issue" açarak veya "Pull Request" göndererek projeye destek olun.

## 📄 Lisans

Bu proje [MIT Lisansı](LICENSE) altında lisanslanmıştır.
