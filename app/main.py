import flet as ft
import os
import sys
import threading
import json

# Proje kök dizinini sisteme ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts import predict
from utils import history_manager, settings_manager

# --- Global Değişkenler ---
model = None
tokenizer = None
config = None
model_loaded = False
available_languages = []
lang_options = []
supported_source_lang = ""
supported_target_lang = ""

# --- Yardımcı Fonksiyonlar ---
def load_available_languages():
    """`app/languages.json` dosyasından dil listesini yükler."""
    global available_languages, lang_options
    try:
        lang_file_path = os.path.join(os.path.dirname(__file__), 'languages.json')
        with open(lang_file_path, 'r', encoding='utf-8') as f:
            available_languages = json.load(f)
    except Exception as e:
        print(f"Hata: 'languages.json' dosyası okunamadı: {e}")
        available_languages = [{"name": "English", "code": "English"}, {"name": "Turkish", "code": "Turkish"}]
    finally:
        lang_options = [ft.dropdown.Option(lang['code'], lang['name']) for lang in available_languages]


# --- Arka Plan Görevleri ---
def initialize_app(page: ft.Page, status_text: ft.Text, controls_to_enable: list):
    """Modeli ve tokenizer'ı arka planda yükler ve UI'ı günceller."""
    global model, tokenizer, config, model_loaded, supported_source_lang, supported_target_lang

    def update_ui_threadsafe(message, color, is_loaded):
        status_text.value = message
        status_text.color = color
        for control in controls_to_enable:
            control.disabled = not is_loaded
        page.update()

    page.run_thread(lambda: update_ui_threadsafe("Konfigürasyon ve model yükleniyor...", ft.Colors.BLUE_GREY, False))
    
    try:
        model, tokenizer, config = predict.load_model()
        if model and tokenizer:
            model_loaded = True
            supported_source_lang = config.get('language', {}).get('source', 'English')
            supported_target_lang = config.get('language', {}).get('target', 'Turkish')
            page.run_thread(lambda: update_ui_threadsafe("Model başarıyla yüklendi. Çeviriye hazır.", ft.Colors.GREEN, True))
        else:
             model_loaded = False
             page.run_thread(lambda: update_ui_threadsafe("Model yüklenemedi. Lütfen modeli eğitin veya yapılandırmayı kontrol edin.", ft.Colors.RED, False))

    except Exception as e:
        model_loaded = False
        page.run_thread(lambda: update_ui_threadsafe(f"Model yüklenirken bir hata oluştu: {e}", ft.Colors.RED, False))


# --- Flet Ana Fonksiyonu ---
def main(page: ft.Page):
    page.title = "Unwired Translate"
    page.vertical_alignment = ft.MainAxisAlignment.START
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.window_width = 1000
    page.window_height = 700

    # Kalıcı ayarları yükle ve temayı ayarla
    settings = settings_manager.load_settings()
    page.theme_mode = ft.ThemeMode.DARK if settings.get("theme_mode") == "dark" else ft.ThemeMode.LIGHT

    load_available_languages()

    # --- Geçmiş Arayüzü ---
    history_list = ft.ListView(expand=True, spacing=10, padding=20, auto_scroll=False)

    def update_history_display():
        history_list.controls.clear()
        history = history_manager.load_history()
        if not history:
            history_list.controls.append(ft.Text("Henüz çeviri geçmişi yok.", italic=True))
        else:
            for entry in reversed(history):
                source_text = entry.get('source_text', entry.get('english', 'N/A'))
                target_text = entry.get('target_text', entry.get('turkish', 'N/A'))
                timestamp = entry.get('timestamp', 'N/A')
                direction = entry.get('direction', f"{supported_source_lang[:2].upper()}->{supported_target_lang[:2].upper()}")

                history_list.controls.append(
                    ft.Card(
                        content=ft.Container(
                            padding=15,
                            content=ft.Column([
                                ft.Text(f"Kaynak ({direction.split('->')[0]}): {source_text}", weight=ft.FontWeight.BOLD),
                                ft.Text(f"Hedef ({direction.split('->')[1]}): {target_text}"),
                                ft.Text(f"Zaman: {timestamp}", size=10, color=ft.Colors.GREY_600),
                            ])
                        )
                    )
                )
        try:
            page.update()
        except Exception:
            pass

    # --- Alt Menü (BottomSheet) ve Ayarlar ---
    bs = ft.BottomSheet(ft.Container(padding=10), open=False)
    page.overlay.append(bs)

    def show_menu(e=None):
        bs.content = menu_view
        bs.open = True
        page.update()

    def show_history(e=None):
        update_history_display()
        bs.content = history_view
        bs.open = True
        page.update()

    def show_settings(e=None):
        bs.content = settings_view
        bs.open = True
        page.update()

    def toggle_theme(e):
        page.theme_mode = ft.ThemeMode.DARK if e.control.value else ft.ThemeMode.LIGHT
        theme_switch.label = "Aydınlık Mod" if e.control.value else "Karanlık Mod"
        
        # Ayarı kaydet
        current_settings = settings_manager.load_settings()
        current_settings['theme_mode'] = 'dark' if e.control.value else 'light'
        settings_manager.save_settings(current_settings)
        
        page.update()

    theme_switch = ft.Switch(
        label="Aydınlık Mod" if page.theme_mode == ft.ThemeMode.DARK else "Karanlık Mod",
        value=(page.theme_mode == ft.ThemeMode.DARK),
        on_change=toggle_theme,
    )

    settings_view = ft.Column([
        ft.Row([
            ft.IconButton(icon=ft.Icons.ARROW_BACK, on_click=show_menu, tooltip="Geri dön"),
            ft.Text("Ayarlar", style=ft.TextThemeStyle.HEADLINE_SMALL),
        ]),
        ft.Row([theme_switch]),
    ])
    
    menu_view = ft.Column([
        ft.ListTile(title=ft.Text("Geçmiş"), leading=ft.Icon(ft.Icons.HISTORY), on_click=show_history),
        ft.ListTile(title=ft.Text("Ayarlar"), leading=ft.Icon(ft.Icons.SETTINGS), on_click=show_settings),
    ])

    history_view = ft.Column([
        ft.Row([
            ft.IconButton(icon=ft.Icons.ARROW_BACK, on_click=show_menu, tooltip="Geri dön"),
            ft.Text("Çeviri Geçmişi", style=ft.TextThemeStyle.HEADLINE_SMALL),
        ]),
        history_list,
    ], expand=True)

    page.appbar = ft.AppBar(
        title=ft.Text("Unwired Translate", style=ft.TextThemeStyle.HEADLINE_MEDIUM),
        center_title=True,
        actions=[ft.IconButton(icon=ft.Icons.MENU, on_click=show_menu, tooltip="Menü")]
    )

    # --- Arayüz Bileşenleri ---
    status_text = ft.Text("Uygulama başlatılıyor...", color=ft.Colors.BLUE_GREY, size=12)
    
    source_lang_dd = ft.Dropdown(options=lang_options, value="English", label="Kaynak Dil", disabled=True, expand=True)
    target_lang_dd = ft.Dropdown(options=lang_options, value="Turkish", label="Hedef Dil", disabled=True, expand=True)
    
    def swap_languages(e):
        source_lang_dd.value, target_lang_dd.value = target_lang_dd.value, source_lang_dd.value
        input_text.value, output_text.value = output_text.value, input_text.value
        page.update()

    swap_button = ft.IconButton(icon=ft.Icons.SWAP_HORIZ, on_click=swap_languages, tooltip="Dilleri Değiştir", disabled=True)

    input_text = ft.TextField(label="Çevrilecek Metin", multiline=True, min_lines=8, max_lines=8, disabled=True)
    output_text = ft.TextField(label="Çeviri Sonucu", multiline=True, min_lines=8, max_lines=8, read_only=True)
    translate_button = ft.ElevatedButton(text="Çevir", icon=ft.Icons.TRANSLATE, disabled=True, height=50)
    progress_ring = ft.ProgressRing(visible=False, width=20, height=20)

    # --- Çeviri İşlevi ---
    def do_translation():
        source_lang = source_lang_dd.value
        target_lang = target_lang_dd.value
        text_to_translate = input_text.value
        
        controls_to_disable = [translate_button, swap_button, source_lang_dd, target_lang_dd, input_text]

        def update_ui_before_translation():
            for c in controls_to_disable: c.disabled = True
            progress_ring.visible = True
            output_text.value = "Çeviri yapılıyor, lütfen bekleyin..."
            page.update()

        def update_ui_after_translation(result_text):
            output_text.value = result_text
            for c in controls_to_disable: c.disabled = False
            progress_ring.visible = False
            if "hata" not in result_text.lower() and text_to_translate.strip():
                history_manager.add_to_history(text_to_translate, result_text, source_lang, target_lang)
                update_history_display()
            page.update()
        
        page.run_thread(update_ui_before_translation)

        try:
            if not text_to_translate.strip():
                page.run_thread(lambda: update_ui_after_translation("Lütfen çevirmek için bir metin girin."))
                return

            translated = predict.translate(
                model, tokenizer, text_to_translate, source_lang, target_lang, 
                max_len=config.get('training', {}).get('max_len', 128)
            )
            page.run_thread(lambda: update_ui_after_translation(translated))

        except Exception as e:
            error_message = f"Çeviri sırasında bir hata oluştu: {e}"
            page.run_thread(lambda: update_ui_after_translation(error_message))

    translate_button.on_click = lambda _: threading.Thread(target=do_translation, daemon=True).start()

    # --- Sayfa Düzeni ---
    lang_selection_row = ft.ResponsiveRow(
        [
            ft.Column(col={"sm": 5, "md": 5}, controls=[source_lang_dd], horizontal_alignment=ft.CrossAxisAlignment.CENTER),
            ft.Column(col={"sm": 2, "md": 2}, controls=[swap_button], horizontal_alignment=ft.CrossAxisAlignment.CENTER),
            ft.Column(col={"sm": 5, "md": 5}, controls=[target_lang_dd], horizontal_alignment=ft.CrossAxisAlignment.CENTER),
        ],
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
        alignment=ft.MainAxisAlignment.CENTER,
    )

    main_layout = ft.ResponsiveRow(
        [
            ft.Column(col={"sm": 12, "md": 5}, controls=[input_text]),
            ft.Column(
                col={"sm": 12, "md": 2},
                controls=[ft.Row([translate_button, progress_ring], alignment=ft.MainAxisAlignment.CENTER)],
                alignment=ft.MainAxisAlignment.CENTER,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            ft.Column(col={"sm": 12, "md": 5}, controls=[output_text]),
        ],
        alignment=ft.MainAxisAlignment.CENTER,
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
    )

    page.add(
        ft.Container(
            padding=10,
            content=ft.Column(
                [
                    status_text,
                    lang_selection_row,
                    main_layout,
                ],
                spacing=15,
                expand=True,
                alignment=ft.MainAxisAlignment.START,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER
            )
        )
    )
    
    # --- Başlangıç ---
    controls_to_enable = [translate_button, swap_button, source_lang_dd, target_lang_dd, input_text]
    threading.Thread(target=initialize_app, args=(page, status_text, controls_to_enable), daemon=True).start()

if __name__ == "__main__":
    ft.app(target=main)