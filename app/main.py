# app/main.py
import flet as ft
import os
import threading
import tensorflow as tf
from transformers import TFMT5ForConditionalGeneration, MT5Tokenizer

tf.config.set_visible_devices([], 'GPU')

# TensorFlow'un bilgi mesajlarını ve uyarılarını bastırarak daha temiz bir çıktı sağlar
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# --- 1. Model ve Tokenizer'ı Yükleme ---
# Bu bölüm, uygulama başlamadan önce modeli belleğe yükler.
MODEL_DIR = "models/mt5-small-Translation_HPLT_EN-TR_V2" # Projenin ana dizininden çalıştırıldığını varsayıyoruz
tokenizer = None
model = None
model_loaded = False

print("Model ve Tokenizer yükleniyor... Lütfen bekleyin.")
try:
    if os.path.exists(MODEL_DIR):
        tokenizer = MT5Tokenizer.from_pretrained(os.path.join(MODEL_DIR, "tokenizer"))
        model = TFMT5ForConditionalGeneration.from_pretrained(os.path.join(MODEL_DIR, "model"))
        model_loaded = True
        print("Model başarıyla yüklendi. Uygulama başlatılıyor.")
    else:
        print(f"HATA: Model dizini bulunamadı: '{MODEL_DIR}'")
        print("Lütfen önce 'scripts/train.py' script'i ile modeli eğitin.")
except Exception as e:
    print(f"Model yüklenirken bir hata oluştu: {e}")

# --- 2. Çeviri Fonksiyonu ---
def translate_sentence(text_to_translate):
    """Verilen metni çevirir."""
    if not model_loaded or tokenizer is None or model is None:
        return "Model yüklenemediği için çeviri yapılamıyor."

    input_text = "translate English to Turkish: " + text_to_translate
    input_ids = tokenizer(input_text, return_tensors="tf").input_ids
    outputs = model.generate(input_ids, max_length=128, num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# --- 3. Flet Uygulamasının Ana Fonksiyonu ---
def main(page: ft.Page):
    # Sayfa ayarları
    page.title = "İngilizce - Türkçe Çeviri Uygulaması"
    page.vertical_alignment = ft.MainAxisAlignment.START
    page.window_width = 700
    page.window_height = 600
    page.theme_mode = ft.ThemeMode.LIGHT

    # UI (Kullanıcı Arayüzü) Elemanları
    input_text = ft.TextField(
        label="Çevrilecek İngilizce Metin",
        multiline=True,
        min_lines=5,
        max_lines=5,
    )
    translate_button = ft.ElevatedButton(text="Çevir", icon=ft.Icons.TRANSLATE)
    output_text = ft.TextField(
        label="Türkçe Çeviri",
        multiline=True,
        min_lines=5,
        max_lines=5,
        read_only=True, # Sadece okunabilir
    )
    progress_ring = ft.ProgressRing(visible=False, width=20, height=20)
    status_text = ft.Text()

    # Çeviri işlemini ayrı bir thread'de çalıştıracak fonksiyon
    def run_translation_in_thread(e):
        # Kullanıcıya işlem başladığını bildir
        progress_ring.visible = True
        translate_button.disabled = True
        output_text.value = "Çeviri yapılıyor, lütfen bekleyin..."
        page.update()

        # Asıl çeviri işlemini yap
        translated = translate_sentence(input_text.value)

        # İşlem bitince sonuçları UI'a yansıt
        output_text.value = translated
        progress_ring.visible = False
        translate_button.disabled = False
        page.update()

    # "Çevir" butonuna tıklandığında ne olacağını belirleyen fonksiyon
    def translate_click(e):
        if not input_text.value:
            output_text.value = "Lütfen çevirmek için bir metin girin."
            page.update()
            return
        
        # Çeviri işlemini arayüzü dondurmamak için yeni bir thread'de başlat
        thread = threading.Thread(target=run_translation_in_thread, args=(e,))
        thread.start()

    # Butonun tıklama olayına fonksiyonu ata
    translate_button.on_click = translate_click

    # Eğer model yüklenmemişse, kullanıcıyı bilgilendir ve butonu pasif yap
    if not model_loaded:
        status_text.value = f"Model '{MODEL_DIR}' dizininde bulunamadı. Lütfen modeli eğitin."
        status_text.color = ft.colors.RED
        input_text.disabled = True
        translate_button.disabled = True

    # Tüm elemanları sayfaya ekle
    page.add(
        ft.Column(
            [
                ft.Text("Unwired Translate", style=ft.TextThemeStyle.HEADLINE_MEDIUM),
                status_text,
                input_text,
                ft.Row(
                    [
                        translate_button,
                        progress_ring
                    ],
                    alignment=ft.MainAxisAlignment.START
                ),
                output_text
            ],
            spacing=20
        )
    )

# --- 4. Uygulamayı Başlatma ---
if __name__ == "__main__":
    if model_loaded:
        ft.app(target=main)
    else:
        # Flet uygulaması yerine konsola bir bekleme mesajı yazdırılabilir
        # Kullanıcı hatayı görüp programı kapatabilir.
        input("Model yüklenemedi. Programı kapatmak için Enter'a basın...")