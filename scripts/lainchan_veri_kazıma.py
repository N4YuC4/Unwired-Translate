import requests
import re
import html
import time
from tqdm.auto import tqdm
import deepl
import os
from deep_translator import GoogleTranslator
import argparse
import textwrap
import random
import sys
from concurrent.futures import ThreadPoolExecutor
import threading


# =====================================================================
# --- AYARLAR ---
# =====================================================================

BOARDS_TO_SCRAPE = ["sec", "zzz", "drug", "hum", "vis"] 
CIKTI_DOSYASI_EN = "lainchan_veriseti-en.txt"
CIKTI_DOSYASI_TR = "lainchan_veriseti-tr.txt"

TEMEL_URL = "https://lainchan.org"
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

REQUEST_DELAY_SECONDS = 2 
MAX_TRANSLATE_CHUNK = 4500
MAX_WORKERS = 4  # Eşzamanlı çeviri iş parçacığı sayısı

# Çeviri Servisleri Yönetimi
ACTIVE_SERVICES = ["deepl", "google"]
DEEPL_KEY = os.getenv("DEEPL_AUTH_KEY")
DEEPL_INSTANCE = None
service_lock = threading.Lock()

if DEEPL_KEY:
    try:
        DEEPL_INSTANCE = deepl.Translator(DEEPL_KEY)
    except:
        if "deepl" in ACTIVE_SERVICES: ACTIVE_SERVICES.remove("deepl")
else:
    if "deepl" in ACTIVE_SERVICES: ACTIVE_SERVICES.remove("deepl")

print(f"Başlangıçta Aktif Servisler: {ACTIVE_SERVICES}")

# =====================================================================
# --- TEMİZLEME VE BÖLME FONKSİYONLARI ---
# =====================================================================
CENSOR_MAP = [
    (r'so+y+k+a+f', 'shit'), (r'fu+a+r+k+ing', 'fucking'),
    (r'fu+a+r+k*er', 'fucker'), (r'fu+a+r+k*ed', 'fucked'), (r'fu+a+r+k', 'fuck'),
]

def extract_paragraphs(raw_html):
    """
    HTML içeriğini alır, temizler ve PARAGRAF LİSTESİ olarak döndürür.
    """
    # 1. Sansür Kaldır
    text = raw_html
    for pattern, replacement in CENSOR_MAP:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    # 2. HTML Gürültüsünü Sil
    # PGP Bloklarını Temizle
    text = re.sub(r'-----BEGIN PGP SIGNED MESSAGE-----.*?-----END PGP SIGNATURE-----', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<s>.*?</s>', '', text, flags=re.DOTALL)
    text = re.sub(r'\[spoiler\].*?\[/spoiler\]', '', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'\|\|.*?\|\|', '', text, flags=re.DOTALL)
    text = re.sub(r'<a.*?>.*?</a>', '', text, flags=re.DOTALL | re.IGNORECASE) 
    text = re.sub(r'<span class="quote">.*?</span>', '', text, flags=re.DOTALL)
    
    # 3. KRİTİK HAMLE: <br> etiketlerini gerçek satır sonuna (\n) çevir
    text = re.sub(r'<br\s*/?>', '\n', text)
    
    # 4. Diğer tagleri sil ve decode et
    text = re.sub(r'<.*?>', '', text)
    text = html.unescape(text)
    
    # 5. Düz Metin Temizliği
    text = re.sub(r'>>\d+', '', text)
    text = re.sub(r'httpsS?://\S+', '', text) 
    
    # 6. Satırlara Böl ve Temizle
    # Metni \n karakterinden bölüyoruz.
    raw_lines = text.split('\n')
    
    clean_paragraphs = []
    for line in raw_lines:
        # Her satırın başındaki sonundaki boşlukları al
        cleaned_line = re.sub(r'\s+', ' ', line).strip()
        
        # Eğer satır doluysa ve çok kısa değilse listeye ekle
        if cleaned_line and len(cleaned_line) > 3:
            clean_paragraphs.append(cleaned_line)
            
    return clean_paragraphs

# =====================================================================
# --- TEKİL ÇEVİRİ FONKSİYONU ---
# =====================================================================
def translate_single_line(text):
    global ACTIVE_SERVICES, service_lock
    
    if not text: return ""

    if len(text) > MAX_TRANSLATE_CHUNK:
        parts = textwrap.wrap(text, width=MAX_TRANSLATE_CHUNK, break_long_words=True, break_on_hyphens=False)
    else:
        parts = [text]

    translated_parts = []

    for part in parts:
        success = False
        retry_count = 0
        max_retries = 5
        
        while not success:
            # Mevcut aktif servislerin kopyası üzerinde işlem yap
            current_services = list(ACTIVE_SERVICES)
            
            for service in current_services:
                if success: break
                
                try:
                    result_text = ""
                    if service == "deepl" and DEEPL_INSTANCE:
                        res = DEEPL_INSTANCE.translate_text(part, target_lang="TR")
                        result_text = res.text
                    elif service == "google":
                        result_text = GoogleTranslator(source='auto', target='tr').translate(part)
                    
                    if result_text:
                        translated_parts.append(result_text)
                        success = True
                        time.sleep(random.uniform(1.0, 3.0))

                except Exception as e:
                    if service == "deepl":
                        tqdm.write(f"  ⚠️ DeepL Hatası: {e}. Servis devre dışı bırakılıyor.")
                        with service_lock:
                            if "deepl" in ACTIVE_SERVICES:
                                ACTIVE_SERVICES.remove("deepl")
                    else:
                        tqdm.write(f"  ⚠️ Google Hatası: {e}")

            if not success:
                if not any(s in ["deepl", "google"] for s in ACTIVE_SERVICES):
                    tqdm.write("  🛑 Tüm çeviri servisleri devre dışı. Çeviri yapılamıyor.")
                    return f"HATA: Çeviri servisleri kullanılamıyor."
                
                if retry_count >= max_retries:
                    tqdm.write("  🛑 Maksimum deneme sayısına ulaşıldı. Paragraf atlanıyor.")
                    return f"HATA: Maksimum deneme sayısına ulaşıldı."

                retry_delay = 5 * (2 ** retry_count)
                tqdm.write(f"  🛑 Çeviri başarısız. {retry_delay} sn bekleniyor (Deneme {retry_count+1}/{max_retries})...")
                time.sleep(retry_delay)
                retry_count += 1
    
    return " ".join(translated_parts)

# =====================================================================
# --- ANA İŞLEM DÖNGÜSÜ ---
# =====================================================================

def get_thread_ids(board):
    url = f"{TEMEL_URL}/{board}/catalog.json"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        data = r.json()
        ids = []
        for page in data:
            for thread in page['threads']:
                ids.append(thread['no'])
        return ids
    except Exception as e:
        print(f"Katalog hatası ({board}): {e}")
        return []

def process_board_synchronized(board):
    print(f"\n--- Pano İşleniyor: {board} ---")
    thread_ids = get_thread_ids(board)
    if not thread_ids:
        print(f"-> '{board}' panosunda işlenecek başlık bulunamadı veya bir hata oluştu.")
        return

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor, \
         open(CIKTI_DOSYASI_EN, 'a', encoding='utf-8') as f_en, \
         open(CIKTI_DOSYASI_TR, 'a', encoding='utf-8') as f_tr:
        
        thread_pbar = tqdm(thread_ids, desc=f"Pano: {board}", unit="thread", position=0, dynamic_ncols=True, file=sys.stdout)
        
        for thread_no in thread_pbar:
            thread_pbar.set_postfix_str(f"İndiriliyor: {thread_no}")
            t_url = f"{TEMEL_URL}/{board}/res/{thread_no}.json"
            
            try:
                r = requests.get(t_url, headers=HEADERS, timeout=10)
                
                if r.status_code == 200:
                    posts = r.json().get('posts', [])
                    
                    paragraphs_to_process = []
                    for post in posts:
                        if 'com' in post:
                            paragraphs_to_process.extend(extract_paragraphs(post['com']))
                    
                    if not paragraphs_to_process:
                        continue

                    with tqdm(total=len(paragraphs_to_process), desc=f"  -> Başlık: {thread_no}", unit="para", position=1, leave=False, dynamic_ncols=True, file=sys.stdout) as para_pbar:
                        # İşlemleri executor'a gönder ve sonuçları geldikçe işle
                        future_translations = executor.map(translate_single_line, paragraphs_to_process)
                        
                        for original, translated in zip(paragraphs_to_process, future_translations):
                            f_en.write(original + '\n')
                            f_tr.write(translated + '\n')
                            para_pbar.update(1)
                        
                        # Dosyaya yazma işlemlerinin her başlık sonunda senkronize olduğundan emin ol
                        f_en.flush()
                        f_tr.flush()

                thread_pbar.set_postfix_str(f"Bekleniyor... ({REQUEST_DELAY_SECONDS}s)")
                time.sleep(REQUEST_DELAY_SECONDS)
            
            except Exception as e:
                thread_pbar.set_postfix_str(f"HATA: {thread_no}")
                tqdm.write(f"  Başlık hatası ({thread_no}): {e}")
                time.sleep(5)

# =====================================================================
# --- MAIN ---
# =====================================================================

def main():

    print("="*50)
    print("Lainchan Paragraf Bazlı Eşleşmiş Veri Kazıyıcı")
    print(f"Hedef: {BOARDS_TO_SCRAPE}")
    print("="*50)

    for board in BOARDS_TO_SCRAPE:
        process_board_synchronized(board)
    
    print("\n✅ Tüm işlemler tamamlandı.")

if __name__ == "__main__":
    main()