import os
import yaml
import json
import pandas as pd
import evaluate
from tqdm import tqdm
import logging
from datetime import datetime
import sys

# Import predict script to reuse loading logic
sys.path.append(os.path.dirname(__file__))
import predict

# --- LOGLAMA AYARLARI ---
log_dir = "logs/evaluate"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # 1. Config Yükle
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 2. Veri Yükle
    logging.info("Test verisi yükleniyor...")
    processed_dir = config['artifacts']['processed_data_dir']
    test_df = pd.read_parquet(os.path.join(processed_dir, "test.parquet"))

    # 3. Model Yükle
    try:
        model_data, _ = predict.load_model("config.yaml")
        translator, tokenizer = model_data # predict.load_model returns (translator, tokenizer), config
        logging.info("Model başarıyla yüklendi.")
    except Exception as e:
        logging.critical(f"Model yüklenemedi: {e}")
        return

    MAX_LEN = config['training']['max_len']
    
    logging.info("METEOR hesaplanıyor...")
    meteor = evaluate.load('meteor')
    all_predictions = []
    all_references = []

    src_langs = test_df["source_lang"].tolist()
    tgt_langs = test_df["target_lang"].tolist()
    src_texts = test_df["source"].tolist()
    tgt_texts = test_df["target"].tolist()

    # CTranslate2 için toplu çeviri yapıyoruz
    batch_size = config['training'].get('eval_batch_size', 8) # config'den al veya varsayılan 8
    
    total = len(src_texts)
    for i in tqdm(range(0, total, batch_size), desc="Tahmin Ediliyor"):
        batch_src_texts = src_texts[i:i+batch_size]
        batch_src_langs = src_langs[i:i+batch_size]
        batch_tgt_langs = tgt_langs[i:i+batch_size]
        batch_ref_texts = tgt_texts[i:i+batch_size]
        
        # Batch olarak prompt oluşturma
        batch_prompts = [
            f"translate {batch_src_langs[j]} to {batch_tgt_langs[j]}: {batch_src_texts[j]}"
            for j in range(len(batch_src_texts))
        ]

        try:
            # Tokenize etme
            batch_source_tokens = [
                tokenizer.convert_ids_to_tokens(tokenizer.encode(p)) for p in batch_prompts
            ]
            
            # Toplu çeviri
            batch_translation_results = translator.translate_batch(
                batch_source_tokens,
                max_batch_size=batch_size, # Mevcut batch boyutu
                beam_size=4,
                max_decoding_length=MAX_LEN
            )
            
            # Detokenize etme
            for j, translation_result in enumerate(batch_translation_results):
                target_tokens = translation_result.hypotheses[0]
                pred_text = tokenizer.decode(tokenizer.convert_tokens_to_ids(target_tokens), skip_special_tokens=True)
                
                all_predictions.append(pred_text)
                all_references.append(batch_ref_texts[j])
            
        except Exception as e:
            logging.error(f"Toplu çeviri hatası (batch {i//batch_size}): {e}")
            # Hata durumunda boş tahmin ekleyerek akışı bozmayız
            for _ in range(len(batch_src_texts)):
                all_predictions.append("")
                all_references.append(batch_ref_texts[_])

    results = meteor.compute(predictions=all_predictions, references=all_references)
    logging.info(f"METEOR Skoru: {results['meteor']:.4f}")

    results_dir = config['artifacts']['results_dir']
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, "metrics_eval_ct2.json"), "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()