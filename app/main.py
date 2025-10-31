# Gerekli kütüphanelerin import edilmesi
import flet as ft
import os
import sys
import threading
import yaml

# `scripts` klasörünü Python yoluna ekleyerek oradaki modüllerin import edilmesini sağla
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))
from predict import load_model, translate_sentence

# `utils` klasörünü Python yoluna ekleyerek oradaki modüllerin import edilmesini sağla
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))
from history_manager import load_history, add_to_history

# --- 1. Global Değişkenler ---
model_loaded = False
model = None
tokenizer = None
config = None
MODEL_DIR = ""
DEVICE = "cpu"  # Masaüstü uygulaması için CPU kullanımı

# --- 2. Asenkron Model Yükleme ---
def initialize_model():
    """
    Modeli ve tokenizer'ı arka planda yükler.
    """
    global model, tokenizer, model_loaded, config, MODEL_DIR

    print("Konfigürasyon ve model yükleniyor... Lütfen bekleyin.")
    
    try:
        with open("config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        output_dir_format = config['system']['output_dir'].format(
            model_mimarisi=config['model_mimarisi'],
            model_teknigi=config['model_teknigi'],
            proje_adi=config['proje_adi'],
            veri_seti=config['veri_seti'],
            versiyon=config['versiyon']
        )
        MODEL_DIR = os.path.join("models", output_dir_format)

        temp_model, temp_tokenizer = load_model(config, device=DEVICE)

        if temp_model and temp_tokenizer:
            model = temp_model
            tokenizer = temp_tokenizer
            model_loaded = True
            print("Model başarıyla yüklendi.")
        else:
            print(f"HATA: Model '{MODEL_DIR}' dizininden yüklenemedi.")
            model_loaded = False

    except Exception as e:
        print(f"Model yüklenirken bir hata oluştu: {e}")
        model_loaded = False

# --- 3. Flet Uygulamasının Ana Arayüz Fonksiyonu ---
def main(page: ft.Page):
    """
    Flet uygulamasının ana arayüzünü (UI) oluşturur ve olayları yönetir.
    """
    page.title = "İngilizce - Türkçe Çeviri Uygulaması"
    page.vertical_alignment = ft.MainAxisAlignment.START
    page.window_width = 700
    page.window_height = 800
    page.theme_mode = ft.ThemeMode.LIGHT

    history_list = ft.ListView(expand=True, spacing=10, padding=20, auto_scroll=False)

    def update_history_display():
        history_list.controls.clear()
        history = load_history()
        if not history:
            history_list.controls.append(ft.Text("Henüz çeviri geçmişi yok.", italic=True))
        else:
            for entry in history:
                history_list.controls.append(
                    ft.Card(
                        content=ft.Container(
                            padding=10,
                            content=ft.Column([
                                ft.Text(f"İngilizce: {entry['english']}", weight=ft.FontWeight.BOLD),
                                ft.Text(f"Türkçe: {entry['turkish']}"),
                                ft.Text(f"Zaman: {entry['timestamp']}", size=10, color=ft.Colors.GREY_600),
                            ])
                        )
                    )
                )
        page.update()

    update_history_display()

    bottom_sheet_container = ft.Container(padding=10)
    bs = ft.BottomSheet(bottom_sheet_container)
    page.overlay.append(bs)

    def show_menu():
        bottom_sheet_container.content = menu_view
        page.update()

    def show_history(e):
        bottom_sheet_container.content = history_view
        page.update()

    menu_view = ft.Column(
        [
            ft.ListTile(title=ft.Text("Geçmiş"), on_click=show_history),
            ft.ListTile(title=ft.Text("Ayarlar (Yakında)"), disabled=True),
        ]
    )

    history_view = ft.Column(
        [
            ft.Row(
                [
                    ft.IconButton(icon=ft.Icons.ARROW_BACK, on_click=lambda e: show_menu()),
                    ft.Text("Çeviri Geçmişi", style=ft.TextThemeStyle.HEADLINE_SMALL),
                ]
            ),
            history_list,
        ],
        expand=True
    )

    def open_bottom_sheet(e):
        show_menu()
        bs.open = True
        page.update()

    page.appbar = ft.AppBar(
        title=ft.Text("Unwired Translate", style=ft.TextThemeStyle.HEADLINE_MEDIUM),
        center_title=True,
        actions=[
            ft.IconButton(
                icon=ft.Icons.MENU,
                on_click=open_bottom_sheet,
            ),
        ],
    )

    # --- Arayüz Bileşenleri (Widget'lar) ---
    input_text = ft.TextField(
        label="Çevrilecek İngilizce Metin",
        multiline=True,
        min_lines=5,
        max_lines=5,
        disabled=True # Başlangıçta devre dışı
    )
    translate_button = ft.ElevatedButton(
        text="Çevir", 
        icon=ft.Icons.TRANSLATE, 
        disabled=True # Başlangıçta devre dışı
    )
    output_text = ft.TextField(
        label="Türkçe Çeviri",
        multiline=True,
        min_lines=5,
        max_lines=5,
        read_only=True,
    )
    progress_ring = ft.ProgressRing(visible=False, width=20, height=20)
    status_text = ft.Text("Model yükleniyor, lütfen bekleyin...", color=ft.Colors.BLUE)

    def model_loader_thread_target(st, it, tb):
        """
        Arka plan thread'inin hedef fonksiyonu. Modeli yükler ve UI'ı günceller.
        """
        initialize_model()
        
        # UI güncellemelerini doğrudan yap
        if model_loaded:
            st.value = "Model başarıyla yüklendi. Çeviri yapmaya hazır."
            st.color = ft.Colors.GREEN
            it.disabled = False
            tb.disabled = False
        else:
            st.value = f"Model '{MODEL_DIR}' yüklenemedi. Lütfen modeli eğitin."
            st.color = ft.Colors.RED
            it.disabled = True
            tb.disabled = True
        
        page.update() # Değişiklikleri arayüze yansıt

    # Modeli arka planda yüklemeye başla
    threading.Thread(target=model_loader_thread_target, args=(status_text, input_text, translate_button), daemon=True).start()

    # --- Olay Fonksiyonları ---
    def run_translation_in_thread(e):
        progress_ring.visible = True
        translate_button.disabled = True
        output_text.value = "Çeviri yapılıyor, lütfen bekleyin..."
        page.update()

        translated = translate_sentence(
            input_text.value,
            model,
            tokenizer,
            max_length=config['training']['max_len'],
            device=DEVICE
        )

        output_text.value = translated
        add_to_history(input_text.value, translated)
        update_history_display()
        bs.open = False
        page.update()

        progress_ring.visible = False
        translate_button.disabled = False
        page.update()

    def translate_click(e):
        if not input_text.value:
            output_text.value = "Lütfen çevirmek için bir metin girin."
            page.update()
            return
        
        threading.Thread(target=run_translation_in_thread, args=(e,), daemon=True).start()

    translate_button.on_click = translate_click

    page.add(
        ft.Column(
            [
                status_text,
                ft.ResponsiveRow(
                    [
                        ft.Column(
                            col={"sm": 12, "md": 6},
                            controls=[
                                input_text,
                                ft.Row([translate_button, progress_ring]),
                            ],
                        ),
                        ft.Column(
                            col={"sm": 12, "md": 6},
                            controls=[
                                output_text,
                            ],
                        ),
                    ],
                    alignment=ft.MainAxisAlignment.START,
                ),
            ],
            spacing=20,
            expand=True,
        )
    )

# --- 4. Uygulamayı Başlatma ---
if __name__ == "__main__":
    ft.app(target=main)