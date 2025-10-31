# Gerekli kütüphanelerin import edilmesi
import os  # İşletim sistemiyle ilgili işlemler için (dosya yolları)
import yaml  # YAML formatındaki yapılandırma dosyalarını okumak için
import argparse  # Komut satırından argüman almak için
import torch  # Derin öğrenme ve tensör işlemleri için temel kütüphane
from transformers import (
    MT5Tokenizer,  # MT5 modeli için tokenizer (metni sayısallaştırıcı)
    T5ForConditionalGeneration,  # Koşullu metin üretimi için T5/MT5 modeli
    BitsAndBytesConfig,  # 4-bit kuantizasyon ayarları için
)
from peft import PeftModel  # PEFT (LoRA) ile eğitilmiş adaptörleri yüklemek için
import logging  # Olayları ve hataları kaydetmek için
from datetime import datetime  # Tarih ve saat bilgileriyle çalışmak için

# --- LOGLAMA AYARLARI ---
# Tahmin (çeviri) işlemi sırasında oluşacak olayları kaydetmek için loglama sistemi.
log_dir = "logs/predict"
os.makedirs(log_dir, exist_ok=True)
log_filename = f"{log_dir}/predict_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(log_filename),
                        logging.StreamHandler()
                    ])

def load_model(config, device="cpu"):
    '''
    Yapılandırma dosyasına göre 4-bit kuantize edilmiş T5 modelini ve LoRA adaptörlerini yükler.

    Args:
        config (dict): 'config.yaml' dosyasından yüklenen yapılandırma.
        device (str): Modelin yükleneceği cihaz ('cuda' veya 'cpu').

    Returns:
        (PeftModel, MT5Tokenizer) or (None, None): Yüklenen model ve tokenizer'ı veya hata durumunda None döndürür.
    '''
    output_dir = config['system']['output_dir'].format(
        model_mimarisi=config['model_mimarisi'],
        model_teknigi=config['model_teknigi'],
        proje_adi=config['proje_adi'],
        veri_seti=config['veri_seti'],
        versiyon=config['versiyon']
    )
    model_dir = os.path.join("models", output_dir)

    if not os.path.exists(model_dir):
        logging.error(f"Hata: Model '{model_dir}' dizininde bulunamadı.")
        return None, None

    logging.info("Kaydedilmiş model ve tokenizer yükleniyor...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    base_model = T5ForConditionalGeneration.from_pretrained(
        config['model_adi'],
        quantization_config=bnb_config,
        device_map={'': device}
    )

    tokenizer = MT5Tokenizer.from_pretrained(model_dir)
    model = PeftModel.from_pretrained(base_model, model_dir)
    model.to(device)
    
    logging.info("Yükleme tamamlandı.")
    return model, tokenizer

# --- ÇEVİRİ FONKSİYONU ---
def translate_sentence(text_to_translate, model, tokenizer, max_length=128, num_beams=4, device="cpu"):
    """
    Verilen bir metni, yüklenmiş olan model ve tokenizer kullanarak çevirir.

    Args:
        text_to_translate (str): Çevrilecek İngilizce metin.
        model: Çeviri için kullanılacak eğitilmiş model (PeftModel).
        tokenizer: Metni sayısallaştırmak için kullanılacak tokenizer.
        max_length (int): Üretilecek çevirinin maksimum token uzunluğu.
        num_beams (int): Beam search algoritması için kullanılacak beam sayısı. Daha yüksek değerler daha iyi sonuç verebilir ama yavaştır.
        device (str): Modelin çalıştırılacağı cihaz ("cuda" veya "cpu").

    Returns:
        str: Çevrilmiş Türkçe metin.
    """
    model.eval()  # Modeli değerlendirme (inference) moduna al. Bu, dropout gibi katmanları devre dışı bırakır.
    
    # Modele görevin ne olduğunu bildiren ön ek (prompt)
    prompt = "translate English to Turkish: " + text_to_translate

    # Girdi metnini tokenizer ile modelin anlayacağı formata dönüştür
    inputs = tokenizer(
        prompt, 
        return_tensors="pt",  # PyTorch tensörleri olarak döndür
        padding=True,  # Gerekirse doldurma yap
        truncation=True,  # Gerekirse metni kırp
        max_length=max_length
    ).to(device)  # Tensörleri ilgili cihaza (GPU/CPU) gönder

    # Gradyan hesaplamasını devre dışı bırakarak bellek kullanımını ve hızı optimize et
    with torch.no_grad():
        # Modeli kullanarak çeviriyi üret
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,  # Üretilecek metnin maksimum uzunluğu
            num_beams=num_beams,  # Beam search için beam sayısı
            early_stopping=True  # Anlamlı bir cümle oluştuğunda üretimi erken durdur
        )
        # Üretilen token ID'lerini tekrar metne dönüştür (decode)
        translation = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return translation

# --- ANA FONKSİYON ---
def main():
    """
    Komut satırından bir metin alarak çeviri işlemini başlatan ve sonucu yazdıran ana fonksiyon.
    """
    # Komut satırı argümanlarını tanımla ve ayrıştır
    parser = argparse.ArgumentParser(description="Eğitilmiş T5 modeli ile İngilizce'den Türkçe'ye çeviri yapın.")
    parser.add_argument("text", type=str, help="Çevirmek istediğiniz İngilizce cümle.")
    args = parser.parse_args()

    # Yapılandırma dosyasını (config.yaml) oku
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Cihazı belirle (GPU varsa ve config'de izin verilmişse GPU, yoksa CPU)
    DEVICE = "cuda" if torch.cuda.is_available() and config['system']['gpu'] else "cpu"

    model, tokenizer = load_model(config, device=DEVICE)

    if model is None or tokenizer is None:
        return # Hata mesajı load_model içinde zaten loglandı.

    logging.info("\nÇeviri yapılıyor...")
    # `translate_sentence` fonksiyonunu çağırarak çeviriyi gerçekleştir
    translated_sentence = translate_sentence(
        args.text, 
        model, 
        tokenizer, 
        config['training']['max_len'],
        device=DEVICE
    )
    
    # Orijinal ve çevrilmiş metni ekrana yazdır
    logging.info(f"\nİngilizce: {args.text}")
    logging.info(f"Türkçe: {translated_sentence}")

# --- SCRIPT'İN BAŞLANGIÇ NOKTASI ---
if __name__ == "__main__":
    # Script doğrudan çalıştırıldığında `main()` fonksiyonunu çağır
    main()
