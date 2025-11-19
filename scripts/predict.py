import os
import yaml
import argparse
import torch
from transformers import (
    MT5Tokenizer,
    T5ForConditionalGeneration,
    BitsAndBytesConfig,
)
from peft import PeftModel
import logging
from datetime import datetime

# --- LOGLAMA AYARLARI ---
log_dir = "logs/predict"
os.makedirs(log_dir, exist_ok=True)
log_filename = f"{log_dir}/predict_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(log_filename),
                        logging.StreamHandler()
                    ])

# --- MODEL YÜKLEME FONKSİYONU (V1 UYUMLU) ---
def load_model(config_path="config.yaml"):
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logging.error(f"'{config_path}' dosyası okunurken hata oluştu: {e}")
        raise ValueError(f"Konfigürasyon dosyası ('{config_path}') okunamadı.") from e

    # Eski model yolu yapısını (V1) kullan
    lora_config = config['system']['lora']
    base_dir = lora_config['base_dir'].format(
        model_mimarisi=config['model_mimarisi'],
        model_teknigi=config['model_teknigi'],
        proje_adi=config['proje_adi'],
        veri_seti=config['veri_seti'],
        versiyon=config['versiyon']
    )
    lang_dir = lora_config['lang_dir'].format(
        source_lang=config['language']['source'],
        target_lang=config['language']['target']
    )
    model_dir = os.path.join(config['system']['output_dir'], base_dir, lang_dir)

    if not os.path.exists(model_dir):
        logging.error(f"Model dizini bulunamadı: '{model_dir}'")
        raise FileNotFoundError(f"Gerekli model dosyaları '{model_dir}' dizininde bulunamadı. Lütfen modelin doğru yerde olduğundan emin olun.")

    try:
        logging.info(f"Model '{model_dir}' dizininden yükleniyor...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        logging.info("1/3: Temel model (base model) yükleniyor...")
        base_model = T5ForConditionalGeneration.from_pretrained(
            config['model_adi'],
            quantization_config=bnb_config,
            device_map="auto"
        )

        logging.info("2/3: Tokenizer yükleniyor...")
        tokenizer = MT5Tokenizer.from_pretrained(model_dir)

        logging.info("3/3: LoRA adaptörleri yükleniyor...")
        model = PeftModel.from_pretrained(base_model, model_dir)
        
        logging.info("Model yükleme tamamlandı.")
        return model, tokenizer, config

    except Exception as e:
        logging.error(f"Model yükleme sırasında beklenmedik bir hata oluştu: {e}", exc_info=True)
        raise RuntimeError(f"Model yüklenirken bir hata meydana geldi. Detaylar için logları kontrol edin. Hata: {e}") from e

# --- ÇEVİRİ FONKSİYONU (V1 UYUMLU - PREFIX YOK) ---
def translate(model, tokenizer, input_texts, source_lang=None, target_lang=None, max_len=128, num_beams=4):
    """
    T5 modeliyle uyumlu çeviri fonksiyonu. Dil prefix'i kullanır.
    """
    model.eval()
    is_single = isinstance(input_texts, str)
    if is_single:
        input_texts = [input_texts]

    prompts = [f"translate {source_lang} to {target_lang}: " + text for text in input_texts]

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len
    ).to(model.device)

    translations = []
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_len,
            num_beams=num_beams,
            early_stopping=True
        )
        translations = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return translations[0] if is_single else translations

# --- ANA FONKSİYON ---
def main():
    parser = argparse.ArgumentParser(description="Eğitilmiş T5 modeli ile çeviri yapın.")
    parser.add_argument("text", type=str, help="Çevirmek istediğiniz cümle.")
    args = parser.parse_args()

    model, tokenizer, config = load_model()

    if model is None or tokenizer is None:
        return

    logging.info("\nÇeviri yapılıyor...")
    max_len = config['training'].get('max_len', 128)

    # V1 modeli tek yönlü olduğu için source/target dilleri config'den gelir, CLI'dan değil.
    translated_sentence = translate(
        model=model,
        tokenizer=tokenizer,
        input_texts=args.text,
        max_len=max_len
    )
    
    source_lang_name = config.get('language', {}).get('source', 'en').upper()
    target_lang_name = config.get('language', {}).get('target', 'tr').upper()

    logging.info(f"\n{source_lang_name}: {args.text}")
    logging.info(f"{target_lang_name}: {translated_sentence}")

if __name__ == "__main__":
    main()