import os
import yaml
import argparse  # Komut satırından argüman almak için
import tensorflow as tf
from transformers import TFMT5ForConditionalGeneration, MT5Tokenizer

tf.config.set_visible_devices([], 'GPU')

# TensorFlow'un bilgi mesajlarını ve uyarılarını bastırarak daha temiz bir çıktı sağlar
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
tf.get_logger().setLevel('ERROR')

def translate_sentence(text_to_translate, model, tokenizer, max_length=128):
    """
    Verilen model ve tokenizer'ı kullanarak tek bir cümleyi çevirir.
    Notebook'taki translate fonksiyonunun aynısıdır.
    """
    # T5 modelinin gerektirdiği şekilde girdi metnini hazırla
    input_text = "translate English to Turkish: " + text_to_translate
    
    # Girdiyi tokenize et
    input_ids = tokenizer(input_text, return_tensors="tf").input_ids
    
    # Çeviriyi oluştur
    outputs = model.generate(
        input_ids, 
        max_length=max_length, 
        num_beams=4,       # Eğitim ve değerlendirmede kullanılan parametrelerle aynı
        early_stopping=True
    )
    
    # Oluşturulan token'ları metne geri dönüştür
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return translated_text

def main():
    # Komut satırı argümanlarını tanımlamak için bir parser oluştur
    parser = argparse.ArgumentParser(description="Eğitilmiş T5 modeli ile İngilizce'den Türkçe'ye çeviri yapın.")
    parser.add_argument("text", type=str, help="Çevirmek istediğiniz İngilizce cümle.")
    args = parser.parse_args()

    # Proje konfigürasyonunu yükle
    try:
        with open("config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("Hata: 'config.yaml' dosyası bulunamadı. Projenin ana dizininde olduğunuzdan emin olun.")
        return

    # Konfigürasyondan modelin yolunu al
    model_dir = config['artifacts']['model_output_dir']
    model_path = os.path.join(model_dir, "model")
    tokenizer_path = os.path.join(model_dir, "tokenizer")

    # Model dosyalarının varlığını kontrol et
    if not os.path.exists(model_path) or not os.path.exists(tokenizer_path):
        print(f"Hata: Model veya tokenizer '{model_dir}' dizininde bulunamadı.")
        print("Lütfen önce 'scripts/train.py' script'ini çalıştırarak modeli eğitin.")
        return

    print("Kaydedilmiş model ve tokenizer yükleniyor...")
    try:
        model = TFMT5ForConditionalGeneration.from_pretrained(model_path)
        tokenizer = MT5Tokenizer.from_pretrained(tokenizer_path)
        print("Yükleme tamamlandı.")
    except Exception as e:
        print(f"Model yüklenirken bir hata oluştu: {e}")
        return
        
    # Çeviriyi gerçekleştir
    print("\nÇeviri yapılıyor...")
    translated_sentence = translate_sentence(
        args.text, 
        model, 
        tokenizer, 
        config['model']['max_length']
    )
    
    # Sonucu ekrana güzel bir