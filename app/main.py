import flet as ft
import os
import sys
import threading
import json
import logging
from datetime import datetime

# Proje kök dizinini sisteme ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts import predict
from utils import history_manager, settings_manager, localization_manager, spell_checker

# --- LOGLAMA AYARLARI ---
log_dir = "logs/app"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"app_{datetime.now().strftime('%Y%m%d')}.log")

logger = logging.getLogger("UnwiredApp")
logger.setLevel(logging.INFO)
if not logger.handlers:
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(console_handler)

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
    global available_languages, lang_options
    try:
        lang_file_path = os.path.join(os.path.dirname(__file__), 'languages.json')
        with open(lang_file_path, 'r', encoding='utf-8') as f:
            available_languages = json.load(f)
    except Exception as e:
        logger.error(f"Dil dosyası yüklenirken hata: {e}")
        available_languages = [{"name": "English", "code": "English"}, {"name": "Turkish", "code": "Turkish"}]
    finally:
        lang_options = [ft.dropdown.Option(lang['code'], lang['name']) for lang in available_languages]

def initialize_app(page: ft.Page, status_indicator: ft.Container, controls_to_enable: list, loc_manager):
    global model, tokenizer, config, model_loaded, supported_source_lang, supported_target_lang

    def update_status(message_key, color, is_loaded, raw_message=None):
        msg = raw_message if raw_message else loc_manager.get(message_key)
        status_indicator.content.value = msg
        status_indicator.bgcolor = color
        status_indicator.visible = True
        if is_loaded:
             status_indicator.bgcolor = ft.Colors.GREEN_700
        
        for control in controls_to_enable:
            control.disabled = not is_loaded
        page.update()

    page.run_thread(lambda: update_status("model_loading", ft.Colors.AMBER_800, False))
    
    try:
        logger.info("Model yükleme işlemi başlatıldı.")
        model, config = predict.load_model()
        if model:
            model_loaded = True
            supported_source_lang = config.get('language', {}).get('source', 'English')
            supported_target_lang = config.get('language', {}).get('target', 'Turkish')
            logger.info("Model başarıyla yüklendi.")
            page.run_thread(lambda: update_status("system_ready", ft.Colors.GREEN, True))
        else:
             model_loaded = False
             logger.error("Model nesnesi boş döndü.")
             page.run_thread(lambda: update_status("model_load_failed", ft.Colors.RED_700, False))

    except Exception as e:
        model_loaded = False
        logger.critical(f"Model yüklenirken kritik hata: {e}")
        # Capture 'e' in default arg to avoid NameError in lambda
        page.run_thread(lambda err=e: update_status(None, ft.Colors.RED_700, False, raw_message=f"{loc_manager.get('error_prefix')}{err}"))

# --- Flet Ana Fonksiyonu ---
def main(page: ft.Page):
    logger.info("Uygulama başlatıldı.")
    
    # Ayarları yükle
    settings = settings_manager.load_settings()
    ui_lang = settings.get("ui_language", "en")
    
    # Localization Manager Başlat
    loc_manager = localization_manager.get_manager(default_lang="en")
    loc_manager.set_language(ui_lang)

    page.title = loc_manager.get("app_title")
    page.padding = 0
    
    # --- Modern Tema Ayarları (Material 3) ---
    page.theme = ft.Theme(
        color_scheme_seed=ft.Colors.INDIGO,
        use_material3=True,
        font_family="Roboto"
    )
    page.dark_theme = ft.Theme(
        color_scheme_seed=ft.Colors.INDIGO,
        use_material3=True,
        font_family="Roboto"
    )
    
    page.theme_mode = ft.ThemeMode.DARK if settings.get("theme_mode") == "dark" else ft.ThemeMode.LIGHT
    
    initial_source_lang = settings.get("source_lang", "English")
    initial_target_lang = settings.get("target_lang", "Turkish")

    load_available_languages()

    # --- Arayüz Bileşenleri ---
    
    # Durum Çubuğu
    status_text = ft.Text(loc_manager.get("initializing"), color=ft.Colors.WHITE, size=12, weight=ft.FontWeight.BOLD)
    status_indicator = ft.Container(
        content=status_text,
        padding=ft.padding.symmetric(horizontal=20, vertical=5),
        border_radius=ft.border_radius.only(top_left=10, top_right=10),
        bgcolor=ft.Colors.BLUE_GREY_700,
        visible=False,
        animate=ft.Animation(300, ft.AnimationCurve.EASE_OUT),
    )

    # Çeviri Bileşenleri
    def save_language_settings(e=None):
        try:
            current_settings = settings_manager.load_settings()
            current_settings["source_lang"] = source_lang_dd.value
            current_settings["target_lang"] = target_lang_dd.value
            settings_manager.save_settings(current_settings)
        except Exception as err:
            logger.error(f"Ayarlar kaydedilirken hata: {err}")

    source_lang_dd = ft.Dropdown(
        options=lang_options, value=initial_source_lang, label=loc_manager.get("source_label"), 
        border_radius=10, content_padding=10, filled=True, dense=True, expand=True,
        on_change=save_language_settings
    )
    target_lang_dd = ft.Dropdown(
        options=lang_options, value=initial_target_lang, label=loc_manager.get("target_label"), 
        border_radius=10, content_padding=10, filled=True, dense=True, expand=True,
        on_change=save_language_settings
    )
    
    input_text = ft.TextField(
        label=loc_manager.get("input_label"), multiline=True, min_lines=5, max_lines=10, 
        border_color=ft.Colors.OUTLINE, border_radius=15, text_size=16,
        expand=True, hint_text=loc_manager.get("input_hint")
    )
    
    # Öneri Bileşeni (Spell Checker)
    suggestion_text = ft.Text("", color=ft.Colors.BLUE_400, weight=ft.FontWeight.BOLD)
    did_you_mean_label = ft.Text(loc_manager.get("did_you_mean"), color=ft.Colors.GREY_500, size=12)
    
    suggestion_container = ft.Container(
        content=ft.Row([
            did_you_mean_label,
            ft.TextButton(content=suggestion_text, style=ft.ButtonStyle(padding=0), on_click=lambda e: apply_suggestion(e)),
        ], spacing=5),
        padding=ft.padding.only(left=10, top=5),
        visible=False
    )
    
    def apply_suggestion(e):
        new_text = suggestion_text.value
        input_text.value = new_text
        suggestion_container.visible = False
        page.update()
        start_translation_thread() # Düzeltilmiş metinle tekrar çevir

    output_text = ft.TextField(
        label=loc_manager.get("output_label"), multiline=True, min_lines=5, max_lines=10, 
        read_only=True, border=ft.InputBorder.NONE, text_size=16,
        expand=True
    )
    
    output_container = ft.Container(
        content=ft.Stack([
            output_text,
            ft.Container(
                content=ft.IconButton(
                    icon=ft.Icons.COPY,
                    icon_size=20,
                    tooltip=loc_manager.get("copied_to_clipboard"),
                    on_click=lambda _: copy_to_clipboard(output_text.value),
                    style=ft.ButtonStyle(padding=5),
                ),
                right=0,
                bottom=0,
            )
        ]),
        padding=10,
        border_radius=15,
        bgcolor=ft.Colors.GREY_900 if page.theme_mode == ft.ThemeMode.DARK else ft.Colors.GREY_200,
        expand=True,
        animate_opacity=300,
    )

    translate_btn = ft.FloatingActionButton(
        icon=ft.Icons.TRANSLATE, text=loc_manager.get("translate_btn"),
        on_click=lambda _: start_translation_thread()
    )
    
    loading_indicator = ft.ProgressBar(width=None, color=ft.Colors.INDIGO, visible=False)

    def swap_languages(e):
        s = source_lang_dd.value
        source_lang_dd.value = target_lang_dd.value
        target_lang_dd.value = s
        in_txt = input_text.value
        input_text.value = output_text.value
        output_text.value = in_txt
        save_language_settings()
        page.update()

    swap_btn = ft.IconButton(icon=ft.Icons.SWAP_HORIZ, on_click=swap_languages, tooltip=loc_manager.get("swap_tooltip"))

    # Geçmiş Listesi
    history_list = ft.ListView(expand=True, spacing=10, padding=20)
    
    def restore_from_history(entry):
        try:
            input_text.value = entry.get('source_text', '')
            output_text.value = entry.get('target_text', '')
            
            direction = entry.get('direction', '')
            if "->" in direction:
                parts = direction.split("->")
                if len(parts) == 2:
                    source_lang_dd.value = parts[0].strip()
                    target_lang_dd.value = parts[1].strip()
                    save_language_settings()
            
            change_nav(0)
        except Exception as e:
            logger.error(f"Geçmişten yükleme hatası: {e}")

    def load_history_items():
        try:
            history_list.controls.clear()
            history = history_manager.load_history()
            if not history:
                history_list.controls.append(
                    ft.Container(
                        content=ft.Column([
                            ft.Icon(ft.Icons.HISTORY_TOGGLE_OFF, size=64, color=ft.Colors.GREY_400),
                            ft.Text(loc_manager.get("history_empty"), color=ft.Colors.GREY_500)
                        ], horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                        alignment=ft.alignment.center,
                        padding=50
                    )
                )
            else:
                for entry in history:
                    direction = entry.get('direction', "Unknown")
                    source = entry.get('source_text', '')
                    target = entry.get('target_text', '')
                    
                    history_list.controls.append(
                        ft.Card(
                            elevation=2,
                            content=ft.Container(
                                padding=15,
                                ink=True,
                                on_click=lambda e, ent=entry: restore_from_history(ent),
                                content=ft.Column([
                                    ft.Row([
                                        ft.Container(
                                            content=ft.Text(direction, size=12, color=ft.Colors.INDIGO_900, weight=ft.FontWeight.BOLD),
                                            bgcolor=ft.Colors.INDIGO_100,
                                            padding=ft.padding.symmetric(horizontal=8, vertical=4),
                                            border_radius=5,
                                        ),
                                        ft.Text(entry.get('timestamp', ''), size=12, color=ft.Colors.GREY),
                                    ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                                    ft.Divider(height=10, color=ft.Colors.TRANSPARENT),
                                    ft.Text(source, size=15, weight=ft.FontWeight.W_500),
                                    ft.Divider(height=1, color=ft.Colors.OUTLINE_VARIANT),
                                    ft.Text(target, size=15, italic=True, color=ft.Colors.PRIMARY)
                                ])
                            )
                        )
                    )
        except Exception as e:
            logger.error(f"Geçmiş yüklenirken hata: {e}")

    # --- Mantıksal İşlemler ---

    def on_input_change(e):
        # Timer özelliğini fonksiyon üzerinde sakla (Static variable simulation)
        if hasattr(on_input_change, "timer") and on_input_change.timer:
            on_input_change.timer.cancel()
        
        if not input_text.value:
            output_text.value = ""
            suggestion_container.visible = False
            page.update()
            return

        # 1.0 saniye gecikme
        on_input_change.timer = threading.Timer(1.0, start_translation_thread)
        on_input_change.timer.start()

    def copy_to_clipboard(text):
        if not text: return
        page.set_clipboard(text)
        page.open(ft.SnackBar(ft.Text(loc_manager.get("copied_to_clipboard")), duration=1000))

    def start_translation_thread():
        if not input_text.value:
            input_text.error_text = loc_manager.get("input_error_empty")
            input_text.update()
            return
        
        input_text.error_text = None
        translate_btn.disabled = True
        loading_indicator.visible = True
        output_container.opacity = 0.5
        page.update()
        
        threading.Thread(target=run_translation, daemon=True).start()

    def run_translation():
        try:
            src = source_lang_dd.value
            tgt = target_lang_dd.value
            txt = input_text.value
            
            # Spell Checker Mantığı
            # Sadece kısa metinlerde ve desteklenen dillerde çalıştır
            if src in ["English", "Turkish"] and txt and len(txt) < 1000:
                try:
                    checker = spell_checker.SpellChecker()
                    corrected = checker.correct(txt, src)
                    if corrected and corrected != txt:
                        suggestion_text.value = corrected
                        suggestion_container.visible = True
                    else:
                        suggestion_container.visible = False
                except Exception as sc_err:
                    logger.warning(f"Spell checker hatası: {sc_err}")
                    suggestion_container.visible = False
            else:
                suggestion_container.visible = False
            
            page.update() # UI güncelle

            logger.info(f"Çeviri isteği: {src} -> {tgt}")
            
            res = predict.translate(
                model, txt, src, tgt, 
                max_len=config.get('training', {}).get('max_len', 128)
            )
            
            output_text.value = res
            history_manager.add_to_history(txt, res, src, tgt)
            load_history_items() # Geçmişi güncelle
            logger.info("Çeviri başarılı.")
            
        except Exception as e:
            error_msg = f"{loc_manager.get('error_occurred')}{e}"
            logger.error(error_msg)
            output_text.value = error_msg
        finally:
            translate_btn.disabled = False
            loading_indicator.visible = False
            output_container.opacity = 1.0
            page.update()

    def toggle_theme(e):
        is_dark = e.control.value
        page.theme_mode = ft.ThemeMode.DARK if is_dark else ft.ThemeMode.LIGHT
        output_container.bgcolor = ft.Colors.GREY_900 if is_dark else ft.Colors.GREY_200
        s = settings_manager.load_settings()
        s['theme_mode'] = 'dark' if is_dark else 'light'
        settings_manager.save_settings(s)
        page.update()

    def change_ui_language(e):
        new_lang = e.control.value
        loc_manager.set_language(new_lang)
        
        # Ayarları kaydet
        s = settings_manager.load_settings()
        s['ui_language'] = new_lang
        settings_manager.save_settings(s)
        
        update_ui_text()
        page.update()

    # UI Metin Güncelleme Fonksiyonu
    def update_ui_text():
        page.title = loc_manager.get("app_title")
        
        # Nav Rail
        nav_rail.destinations[0].label = loc_manager.get("nav_translate")
        nav_rail.destinations[1].label = loc_manager.get("nav_history")
        nav_rail.destinations[2].label = loc_manager.get("nav_settings")
        
        # Nav Bar
        nav_bar.destinations[0].label = loc_manager.get("nav_translate")
        nav_bar.destinations[1].label = loc_manager.get("nav_history")
        nav_bar.destinations[2].label = loc_manager.get("nav_settings")

        # Translate View
        translate_title_text.value = loc_manager.get("new_translation_title")
        did_you_mean_label.value = loc_manager.get("did_you_mean")
        source_lang_dd.label = loc_manager.get("source_label")
        target_lang_dd.label = loc_manager.get("target_label")
        swap_btn.tooltip = loc_manager.get("swap_tooltip")
        input_text.label = loc_manager.get("input_label")
        input_text.hint_text = loc_manager.get("input_hint")
        output_text.label = loc_manager.get("output_label")
        translate_btn.text = loc_manager.get("translate_btn")

        # History View
        history_header.value = loc_manager.get("history_title")
        load_history_items() # Geçmiş boş mesajını güncellemek için

        # Settings View
        settings_header.value = loc_manager.get("settings_title")
        dark_mode_text.value = loc_manager.get("dark_mode")
        clear_history_tile.title.value = loc_manager.get("clear_history_title")
        clear_history_tile.subtitle.value = loc_manager.get("clear_history_subtitle")
        ui_lang_dd.label = loc_manager.get("language_select")


    # --- Dialoglar ---
    def open_clear_history_dialog(e):
        def close_dlg(e):
            page.close(dlg)

        def confirm_clear(e):
            history_manager.clear_history()
            load_history_items()
            page.close(dlg)
            page.open(ft.SnackBar(ft.Text(loc_manager.get("history_cleared_msg"))))
            logger.info("Kullanıcı geçmişi temizledi.")

        dlg = ft.AlertDialog(
            modal=True,
            title=ft.Text(loc_manager.get("confirm_title")),
            content=ft.Text(loc_manager.get("confirm_content")),
            actions=[
                ft.TextButton(loc_manager.get("no_btn"), on_click=close_dlg),
                ft.TextButton(loc_manager.get("yes_btn"), on_click=confirm_clear, style=ft.ButtonStyle(color=ft.Colors.RED)),
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )
        page.open(dlg)

    # --- Views (Sayfa İçerikleri) ---

    # Responsive Çeviri Görünümü
    translate_title_text = ft.Text(
        loc_manager.get("new_translation_title"),
        style=ft.TextThemeStyle.HEADLINE_MEDIUM,
        color=ft.Colors.PRIMARY,
        weight=ft.FontWeight.BOLD,
    )
    translate_header = ft.Row(
        [
            ft.Icon(ft.Icons.TRANSLATE, color=ft.Colors.PRIMARY, size=30),
            translate_title_text,
        ],
        alignment=ft.MainAxisAlignment.CENTER,
    )
    
    input_stack = ft.Stack([
        input_text,
        ft.Container(
            content=ft.IconButton(
                icon=ft.Icons.COPY, 
                icon_size=20, 
                tooltip=loc_manager.get("copied_to_clipboard"),
                on_click=lambda _: copy_to_clipboard(input_text.value),
                style=ft.ButtonStyle(padding=5),
            ),
            right=5,
            bottom=5,
        )
    ])

    translate_view = ft.Container(
        padding=20,
        content=ft.Column([
            translate_header,
            
            # Dil Seçimi
            ft.Container(
                content=ft.Row([source_lang_dd, swap_btn, target_lang_dd], alignment=ft.MainAxisAlignment.CENTER),
                padding=ft.padding.only(bottom=10)
            ),
            
            ft.Divider(),
            
            # Metin Alanları (ResponsiveRow ile)
            ft.ResponsiveRow([
                ft.Column(col={"sm": 12, "md": 6}, controls=[input_stack, suggestion_container]),
                ft.Column(col={"sm": 12, "md": 6}, controls=[output_container]),
            ]),
            
            ft.Row([loading_indicator], alignment=ft.MainAxisAlignment.CENTER),
            ft.Row([translate_btn], alignment=ft.MainAxisAlignment.END)
        ], scroll=ft.ScrollMode.AUTO)
    )

    history_header = ft.Text(loc_manager.get("history_title"), style=ft.TextThemeStyle.HEADLINE_MEDIUM, color=ft.Colors.PRIMARY)

    history_view = ft.Container(
        padding=20,
        content=ft.Column([
            history_header,
            ft.Divider(),
            history_list
        ])
    )
    
    # Ayarlar UI Elemanları
    settings_header = ft.Text(loc_manager.get("settings_title"), style=ft.TextThemeStyle.HEADLINE_MEDIUM)
    dark_mode_text = ft.Text(loc_manager.get("dark_mode"), size=16)
    
    clear_history_tile = ft.ListTile(
        leading=ft.Icon(ft.Icons.DELETE_FOREVER, color=ft.Colors.RED),
        title=ft.Text(loc_manager.get("clear_history_title"), color=ft.Colors.RED),
        subtitle=ft.Text(loc_manager.get("clear_history_subtitle")),
        on_click=open_clear_history_dialog
    )
    
    ui_lang_dd = ft.Dropdown(
        label=loc_manager.get("language_select"),
        value=loc_manager.current_lang,
        options=[
            ft.dropdown.Option("tr", "Türkçe"),
            ft.dropdown.Option("en", "English"),
            ft.dropdown.Option("fr", "Français"),
            ft.dropdown.Option("de", "Deutsch"),
            ft.dropdown.Option("es", "Español"),
            ft.dropdown.Option("it", "Italiano"),
            ft.dropdown.Option("pt", "Português"),
            ft.dropdown.Option("ru", "Русский"),
            ft.dropdown.Option("zh", "中文"),
            ft.dropdown.Option("ja", "日本語"),
            ft.dropdown.Option("ko", "한국어"),
            ft.dropdown.Option("ar", "العربية"),
        ],
        on_change=change_ui_language,
        width=200
    )

    settings_view = ft.Container(
        padding=40,
        content=ft.Column([
            settings_header,
            ft.Divider(),
            ft.Row([
                ft.Icon(ft.Icons.DARK_MODE),
                dark_mode_text,
                ft.Switch(value=(page.theme_mode == ft.ThemeMode.DARK), on_change=toggle_theme)
            ], spacing=20),
            ft.Divider(),
             ft.Row([
                ft.Icon(ft.Icons.LANGUAGE),
                ui_lang_dd
            ], spacing=20),
            ft.Divider(),
            clear_history_tile
        ])
    )

    # --- Navigasyon ---
    
    destinations = [
        ft.NavigationRailDestination(icon=ft.Icons.TRANSLATE_OUTLINED, selected_icon=ft.Icons.TRANSLATE, label=loc_manager.get("nav_translate")),
        ft.NavigationRailDestination(icon=ft.Icons.HISTORY_OUTLINED, selected_icon=ft.Icons.HISTORY, label=loc_manager.get("nav_history")),
        ft.NavigationRailDestination(icon=ft.Icons.SETTINGS_OUTLINED, selected_icon=ft.Icons.SETTINGS, label=loc_manager.get("nav_settings")),
    ]
    
    destinations_bar = [
        ft.NavigationBarDestination(icon=ft.Icons.TRANSLATE_OUTLINED, selected_icon=ft.Icons.TRANSLATE, label=loc_manager.get("nav_translate")),
        ft.NavigationBarDestination(icon=ft.Icons.HISTORY_OUTLINED, selected_icon=ft.Icons.HISTORY, label=loc_manager.get("nav_history")),
        ft.NavigationBarDestination(icon=ft.Icons.SETTINGS_OUTLINED, selected_icon=ft.Icons.SETTINGS, label=loc_manager.get("nav_settings")),
    ]

    content_area = ft.AnimatedSwitcher(
        content=translate_view,
        transition=ft.AnimatedSwitcherTransition.FADE,
        duration=300,
        reverse_duration=300,
        expand=True
    )

    nav_rail = ft.NavigationRail(
        selected_index=0,
        label_type=ft.NavigationRailLabelType.ALL,
        min_width=100,
        min_extended_width=200,
        group_alignment=-0.9,
        destinations=destinations,
        on_change=lambda e: change_nav(e.control.selected_index),
        visible=True
    )

    nav_bar = ft.NavigationBar(
        selected_index=0,
        destinations=destinations_bar,
        on_change=lambda e: change_nav(e.control.selected_index),
        visible=False
    )

    def change_nav(index):
        nav_rail.selected_index = index
        nav_bar.selected_index = index
        if index == 0:
            content_area.content = translate_view
        elif index == 1:
            load_history_items()
            content_area.content = history_view
        elif index == 2:
            content_area.content = settings_view
        page.update()

    # --- Layout & Responsive Logic ---
    
    layout = ft.Row(
        controls=[
            nav_rail,
            ft.VerticalDivider(width=1),
            ft.Column([content_area, status_indicator], expand=True),
        ],
        expand=True,
    )

    page.add(layout)
    # Başlangıçta nav_bar eklenmez, gerekirse handle_resize ekler

    def handle_resize(e):
        if page.width < 700: # Mobil Görünüm
            nav_rail.visible = False
            layout.controls[1].visible = False # Divider gizle
            page.navigation_bar = nav_bar
            nav_bar.visible = True
        else: # Masaüstü Görünüm
            nav_rail.visible = True
            layout.controls[1].visible = True # Divider göster
            page.navigation_bar = None
        page.update()

    page.on_resized = handle_resize
    handle_resize(None) # İlk yükleme

    # Event Atamaları
    input_text.on_change = on_input_change

    # Başlatma
    controls = [translate_btn, input_text, source_lang_dd, target_lang_dd, swap_btn]
    threading.Thread(target=initialize_app, args=(page, status_indicator, controls, loc_manager), daemon=True).start()

if __name__ == "__main__":
    ft.app(target=main)