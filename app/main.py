import flet as ft
import os
import sys
import threading
import json

# Proje kök dizinini sisteme ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts import predict
from utils import history_manager

# --- Global Değişkenler ---
model = None
tokenizer = None
config = None
model_loaded = False
available_languages = []
lang_options = []
supported_source_lang = ""
supported_target_lang = ""
source_lang_dd = None
target_lang_dd = None

# --- Yardımcı Fonksiyonlar ---
def load_available_languages():
    """`app/languages.json` dosyasından dil listesini yükler."""
    global available_languages
    try:
        lang_file_path = os.path.join(os.path.dirname(__file__), 'languages.json')
        with open(lang_file_path, 'r', encoding='utf-8') as f:
            available_languages = json.load(f)
    except Exception as e:
        print(f"Hata: 'languages.json' dosyası okunamadı: {e}")
        available_languages = [{"name": "English", "code": "English"}, {"name": "Turkish", "code": "Turkish"}]

# --- Arka Plan Görevleri ---
def initialize_app(page: ft.Page, status_text: ft.Text, controls_to_enable: list):
    """Modeli ve tokenizer'ı arka planda yükler ve UI'ı günceller."""
    global model, tokenizer, config, model_loaded, supported_source_lang, supported_target_lang

    def update_ui(message, color, is_loaded):
        status_text.value = message
        status_text.color = color
        for control in controls_to_enable:
            control.disabled = not is_loaded
        # Filtreleme mantığı kaldırıldı
        page.update()

    page.run_thread(lambda: update_ui("Konfigürasyon ve model yükleniyor...", ft.Colors.BLUE_GREY, False))
    
    try:
        model, tokenizer, config = predict.load_model()
        if model and tokenizer:
            model_loaded = True
            supported_source_lang = config.get('language', {}).get('source', 'English')
            supported_target_lang = config.get('language', {}).get('target', 'Turkish')
            page.run_thread(lambda: update_ui("Model (V1) başarıyla yüklendi. Çeviriye hazır.", ft.Colors.GREEN, True))
    except Exception as e:
        model_loaded = False
        page.run_thread(lambda: update_ui(f"Model yüklenirken bir hata oluştu: {e}", ft.Colors.RED, False))

# --- Flet Ana Fonksiyonu ---
def main(page: ft.Page):
    global lang_options, source_lang_dd, target_lang_dd
    page.title = "Unwired Translate"
    page.vertical_alignment = ft.MainAxisAlignment.START
    page.window_width = 800
    page.window_height = 700
    page.theme_mode = ft.ThemeMode.LIGHT

    load_available_languages()
    lang_options = [ft.dropdown.Option(lang['code'], lang['name']) for lang in available_languages]

    # --- Arayüz Olay Yöneticileri ---
    def swap_languages(e):
        """Dilleri takas eder (basit versiyon)."""
        source_lang_dd.value, target_lang_dd.value = target_lang_dd.value, source_lang_dd.value
        page.update()

    # --- Arayüz Bileşenleri ---
    status_text = ft.Text("Uygulama başlatılıyor...", color=ft.Colors.BLUE_GREY, size=12)
    source_lang_dd = ft.Dropdown(
        options=lang_options, 
        value="English", 
        label="Kaynak Dil", 
        disabled=True,
    )
    target_lang_dd = ft.Dropdown(
        options=lang_options, 
        value="Turkish", 
        label="Hedef Dil", 
        disabled=True,
    )
    input_text = ft.TextField(label="Çevrilecek Metin", multiline=True, min_lines=5, max_lines=5, disabled=True)
    output_text = ft.TextField(label="Çeviri Sonucu", multiline=True, min_lines=5, max_lines=5, read_only=True)
    translate_button = ft.ElevatedButton(text="Çevir", icon=ft.Icons.TRANSLATE, disabled=True, height=50)
    progress_ring = ft.ProgressRing(visible=False, width=20, height=20)
    swap_button = ft.IconButton(icon=ft.Icons.SWAP_HORIZ, on_click=swap_languages, tooltip="Dilleri Değiştir", disabled=True)
    history_list = ft.ListView(expand=True, spacing=10, auto_scroll=True)

    # --- Diğer Olay Fonksiyonları ---
    def update_history_display():
        history_list.controls.clear()
        history = history_manager.load_history()
        for entry in reversed(history):
            if 'direction' in entry:
                source_lang_code, target_lang_code = entry['direction'].split('->')
            else:
                source_lang_code = supported_source_lang[:2].lower() if supported_source_lang else "en"
                target_lang_code = supported_target_lang[:2].lower() if supported_target_lang else "tr"
            history_list.controls.append(
                ft.Card(
                    content=ft.Container(
                        padding=15,
                        content=ft.Column([
                            ft.Text(f"{source_lang_code.upper()}: {entry['source_text']}", weight=ft.FontWeight.BOLD),
                            ft.Text(f"{target_lang_code.upper()}: {entry['target_text']}"),
                            ft.Text(f"Zaman: {entry['timestamp']}", size=10, color=ft.Colors.GREY),
                        ])
                    )
                )
            )
        page.update()

    def do_translation():
        source_lang = source_lang_dd.value
        target_lang = target_lang_dd.value
        controls_to_manage = [translate_button, swap_button, source_lang_dd, target_lang_dd, input_text]
        page.run_thread(lambda: [setattr(c, 'disabled', True) for c in controls_to_manage])
        page.run_thread(lambda: setattr(progress_ring, 'visible', True))
        page.run_thread(page.update)
        try:
            if not input_text.value.strip():
                output_text.value = "Lütfen çevirmek için bir metin girin."
            else:
                translated = predict.translate(model, tokenizer, input_text.value, source_lang, target_lang, max_len=config.get('training', {}).get('max_len', 128))
                output_text.value = translated
                history_manager.add_to_history(input_text.value, translated, source_lang, target_lang)
                update_history_display()
        except Exception as e:
            output_text.value = f"Çeviri sırasında bir hata oluştu: {e}"
        page.run_thread(lambda: [setattr(c, 'disabled', False) for c in controls_to_manage])
        page.run_thread(lambda: setattr(progress_ring, 'visible', False))
        page.run_thread(page.update)

    translate_button.on_click = lambda _: threading.Thread(target=do_translation, daemon=True).start()
    
    # --- Sayfa Düzeni ---
    lang_selection_row = ft.Row(
        [
            ft.Column([source_lang_dd], expand=True),
            ft.Column([swap_button], alignment=ft.MainAxisAlignment.CENTER),
            ft.Column([target_lang_dd], expand=True),
        ],
        alignment=ft.MainAxisAlignment.CENTER, vertical_alignment=ft.CrossAxisAlignment.CENTER
    )
    page.add(
        ft.Container(
            padding=20,
            content=ft.Column(
                controls=[
                    status_text,
                    ft.Column([lang_selection_row, input_text, ft.Row([translate_button, progress_ring], alignment=ft.MainAxisAlignment.CENTER), output_text], spacing=15),
                    ft.Divider(height=30),
                    ft.Text("Çeviri Geçmişi", style=ft.TextThemeStyle.HEADLINE_SMALL),
                    ft.Container(content=history_list, border=ft.border.all(1, ft.Colors.BLACK26), border_radius=ft.border_radius.all(5), expand=True)
                ], expand=True
            )
        )
    )
    
    update_history_display()

    # --- Başlangıç ---
    controls_to_enable = [translate_button, swap_button, source_lang_dd, target_lang_dd, input_text]
    threading.Thread(target=initialize_app, args=(page, status_text, controls_to_enable), daemon=True).start()

if __name__ == "__main__":
    ft.app(target=main)