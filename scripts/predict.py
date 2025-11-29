import os
import yaml
import argparse
import torch
import gc
from transformers import (
    MT5Tokenizer,
    MT5ForConditionalGeneration,
    BitsAndBytesConfig,
)
from peft import PeftModel
import logging
from datetime import datetime

# --- LOGLAMA AYARLARI ---
log_dir = "logs/predict"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"predict_{datetime.now().strftime('%Y%m%d')}.log")

# Logger oluştur
logger = logging.getLogger("UnwiredPredict")
logger.setLevel(logging.INFO)

# Dosya Handler
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Konsol Handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
logger.addHandler(console_handler)

# --- MODEL YÜKLEME FONKSİYONU ---
def load_model(config_path="config.yaml"):
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"'{config_path}' dosyası okunurken hata oluştu: {e}")
        raise ValueError(f"Konfigürasyon dosyası okunamadı: {e}")

    lora_config = config['system']['lora']
    
    dataset_names = config['dataset'].get('names', [])
    if isinstance(dataset_names, list) and dataset_names:
        dset_val = "-".join([str(x).upper() for x in dataset_names])
    else:
        dset_val = config.get('veri_seti', 'Unknown')

    base_dir = lora_config['base_dir'].format(
        model_mimarisi=config['model_mimarisi'],
        model_teknigi=config['model_teknigi'],
        proje_adi=config['proje_adi'],
        veri_seti=dset_val,
        versiyon=config['versiyon']
    )
    lang_dir = lora_config['lang_dir'].format(
        source_lang=config['language']['source'],
        target_lang=config['language']['target']
    )
    model_dir = os.path.join(config['system']['output_dir'], base_dir, lang_dir)

    if not os.path.exists(model_dir):
        logger.error(f"Model dizini bulunamadı: '{model_dir}'")
        raise FileNotFoundError(f"Model dizini bulunamadı: {model_dir}")

    try:
        logger.info(f"Model yükleniyor: {model_dir}")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        base_model = MT5ForConditionalGeneration.from_pretrained(
            config['model_adi'],
            quantization_config=bnb_config,
            device_map="auto"
        )

        tokenizer = MT5Tokenizer.from_pretrained(model_dir)
        model = PeftModel.from_pretrained(base_model, model_dir)
        model.eval()
        
        return model, tokenizer, config

    except Exception as e:
        logger.error(f"Model yükleme hatası: {e}")
        raise RuntimeError(f"Model yükleme hatası: {e}")

# --- ÇEVİRİ FONKSİYONU ---
@torch.inference_mode()
def translate(model, tokenizer, input_texts, source_lang, target_lang, max_len=128, num_beams=4):
    is_single = isinstance(input_texts, str)
    if is_single:
        input_texts = [input_texts]

    prompts = [f"translate {source_lang} to {target_lang}: " + text for text in input_texts]

    try:
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len
        ).to(model.device)

        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_len,
            num_beams=num_beams,
            early_stopping=True
        )
        translations = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        result = translations[0] if is_single else translations
        return result

    except Exception as e:
        logger.error(f"Çeviri hatası: {e}")
        return f"Hata: {str(e)}"
    
    finally:
        # Bellek Temizliği
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

# --- ANA FONKSİYONU ---
def main():
    parser = argparse.ArgumentParser(description="MT5 Çeviri Aracı")
    parser.add_argument("text", type=str, help="Çevrilecek metin")
    parser.add_argument("--src", type=str, help="Kaynak dil (Opsiyonel, config'den alır)")
    parser.add_argument("--tgt", type=str, help="Hedef dil (Opsiyonel, config'den alır)")
    args = parser.parse_args()

    try:
        model, tokenizer, config = load_model()

        src_lang = args.src if args.src else config['language']['source']
        tgt_lang = args.tgt if args.tgt else config['language']['target']
        
        logger.info(f"Çeviri Başlatılıyor: {src_lang} -> {tgt_lang}")
        
        max_len = config['training'].get('max_len', 128)

        result = translate(
            model=model,
            tokenizer=tokenizer,
            input_texts=args.text,
            source_lang=src_lang,
            target_lang=tgt_lang,
            max_len=max_len
        )
        
        print(f"\n[{src_lang}] {args.text}")
        print(f"[{tgt_lang}] {result}\n")
        
    except Exception as e:
        logger.critical(f"Kritik Hata: {e}")

if __name__ == "__main__":
    main()