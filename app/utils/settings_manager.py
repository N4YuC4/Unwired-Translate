import json
import os

# Ayarlar dosyasının yolu
SETTINGS_FILE = os.path.join("artifacts", "app_settings.json")

def load_settings():
    """
    Ayarları JSON dosyasından yükler.
    Dosya yoksa veya boşsa, varsayılan ayarları döndürür.
    """
    # artifacts dizininin var olduğundan emin ol
    os.makedirs(os.path.dirname(SETTINGS_FILE), exist_ok=True)
    
    try:
        if os.path.exists(SETTINGS_FILE) and os.path.getsize(SETTINGS_FILE) > 0:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        print(f"Ayarlar yüklenirken hata oluştu ({e}), varsayılanlara dönülüyor.")
    
    # Varsayılan ayarlar
    return {"theme_mode": "light", "ui_language": "en"}

def save_settings(settings):
    """
    Verilen ayarları JSON dosyasına kaydeder.
    """
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(settings, f, ensure_ascii=False, indent=4)
    except IOError as e:
        print(f"Ayarlar kaydedilirken hata oluştu: {e}")
