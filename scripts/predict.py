import os
import yaml
import argparse
import sys
import logging
from datetime import datetime
import ctranslate2
from transformers import MT5Tokenizer

# --- LOGLAMA AYARLARI ---
log_dir = "logs/predict"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"predict_{datetime.now().strftime('%Y%m%d')}.log")

logger = logging.getLogger("UnwiredPredict")
logger.setLevel(logging.INFO)

if not logger.handlers:
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(console_handler)

# --- MODEL YÜKLEME ---
def load_model(config_path="config.yaml"):
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Config okuma hatası: {e}")
        raise

    # Model Yolları
    lora_config = config['system']['lora']
    dataset_names = config['dataset'].get('names', [])
    dset_val = "-".join([str(x).upper() for x in dataset_names]) if dataset_names else config.get('veri_seti', 'Unknown')
    
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

    ct2_model_path = os.path.join(config['system']['output_dir'], base_dir, lang_dir)
    
    if not os.path.exists(ct2_model_path):
        logger.error(f"CTranslate2 model bulunamadı: {ct2_model_path}")
        raise FileNotFoundError(f"Model bulunamadı: {ct2_model_path}")

    try:
        logger.info(f"CTranslate2 Model yükleniyor: {ct2_model_path}")
        device = "cuda" if config['system']['gpu'] and ctranslate2.get_cuda_device_count() > 0 else "cpu"
        logger.info(f"Cihaz: {device}")
        
        translator = ctranslate2.Translator(ct2_model_path, device=device)
        
        # Tokenizer'ı base modelden veya convert edilen dizinden yükleyebiliriz.
        # En garantisi model_adi'ndan yüklemek (eğer özel vocab yoksa) veya eğitim çıktısından.
        # Merge işlemi sırasında tokenizer da CT2 klasörüne kopyalanmamış olabilir, 
        # bu yüzden merged veya base model kullanabiliriz. 
        # Train scriptinde merged_model path'e tokenizer save ediliyor, oradan convert ediliyor.
        # Ancak convert scripti tokenizer dosyasını kopyalamıyor olabilir.
        # Güvenlik için config'deki model isminden taze bir tokenizer yükleyelim.
        
        model_name = config['model_adi']
        logger.info(f"Tokenizer yükleniyor: {model_name}")
        tokenizer = MT5Tokenizer.from_pretrained(model_name)
        
        return (translator, tokenizer), config

    except Exception as e:
        logger.critical(f"Model yükleme hatası: {e}")
        raise

# --- ÇEVİRİ ---
def translate(model_data, text, source_lang, target_lang, max_len=128):
    translator, tokenizer = model_data
    
    is_single = isinstance(text, str)
    if is_single:
        text = [text]

    results = []
    
    for txt in text:
        prompt = f"translate {source_lang} to {target_lang}: {txt}"
        
        try:
            # 1. Tokenize
            source_tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(prompt))
            
            # 2. Translate
            translation_result = translator.translate_batch(
                [source_tokens],
                max_batch_size=1,
                beam_size=4,
                max_decoding_length=max_len
            )
            
            # 3. Detokenize
            target_tokens = translation_result[0].hypotheses[0]
            target_text = tokenizer.decode(tokenizer.convert_tokens_to_ids(target_tokens), skip_special_tokens=True)
            
            results.append(target_text)
            
        except Exception as e:
            logger.error(f"Çeviri hatası: {e}")
            results.append(f"Hata: {str(e)}")

    return results[0] if is_single else results

# --- CLI ---
def main():
    parser = argparse.ArgumentParser(description="Unwired Translate (CTranslate2)")
    parser.add_argument("text", type=str, help="Text")
    parser.add_argument("--src", type=str, help="Source Lang")
    parser.add_argument("--tgt", type=str, help="Target Lang")
    args = parser.parse_args()

    try:
        model_data, config = load_model()
        src = args.src if args.src else config['language']['source']
        tgt = args.tgt if args.tgt else config['language']['target']
        
        res = translate(model_data, args.text, src, tgt)
        print(f"[{src}->{tgt}]: {res}")
    except Exception as e:
        print(f"Hata: {e}")

if __name__ == "__main__":
    main()
