# Gerekli kütüphanelerin import edilmesi
import os
import yaml
import json
import pandas as pd
import torch
import evaluate
from tqdm import tqdm
from transformers import (
    MT5Tokenizer,
    T5ForConditionalGeneration,
    BitsAndBytesConfig,
)
from peft import PeftModel
from torch.utils.data import Dataset, DataLoader
import logging
from datetime import datetime

# --- LOGLAMA AYARLARI ---
log_dir = "logs/evaluate"
os.makedirs(log_dir, exist_ok=True)
log_filename = f"{log_dir}/evaluate_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(log_filename),
                        logging.StreamHandler()
                    ])

# --- PYTORCH VERİ SETİ SINIFI (V1 UYUMLU) ---
class TranslationDataset(Dataset):
    def __init__(self, tokenizer, src_texts, tgt_texts, max_len=128):
        self.tokenizer = tokenizer
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.max_len = max_len

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, idx):
        src_text = self.src_texts[idx]
        tgt_text = self.tgt_texts[idx]

        source = self.tokenizer(src_text, truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")
        target = self.tokenizer(tgt_text, truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")

        labels = target["input_ids"]
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": source["input_ids"].squeeze(),
            "attention_mask": source["attention_mask"].squeeze(),
            "labels": labels.squeeze(),
        }

# --- ANA DEĞERLENDİRME FONKSİYONU ---
def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    DEVICE = "cuda" if torch.cuda.is_available() and config['system']['gpu'] else "cpu"

    logging.info("Test verisi ve model yükleniyor...")
    processed_data_dir = config['artifacts']['processed_data_dir']
    test_df = pd.read_parquet(os.path.join(processed_data_dir, "test.parquet"))

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

    # --- MODEL YÜKLEME ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = T5ForConditionalGeneration.from_pretrained(
        config['model_adi'],
        quantization_config=bnb_config,
        device_map="auto"
    )

    tokenizer = MT5Tokenizer.from_pretrained(model_dir)
    model = PeftModel.from_pretrained(model, model_dir)

    MAX_LEN = config['training']['max_len']
    
    test_dataset = TranslationDataset(tokenizer, test_df["source"].tolist(), test_df["target"].tolist(), MAX_LEN)
    test_loader = DataLoader(test_dataset, batch_size=config['training']['train_batch_size']*2, shuffle=False)

    # --- METRİK HESAPLAMA ---
    logging.info("METEOR metriği yükleniyor...")
    meteor = evaluate.load('meteor')

    all_predictions = []
    all_references = []

    logging.info("Çeviriler oluşturuluyor...")
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Çeviriler oluşturuluyor"):
            # Orijinal script'teki batching mantığına geri dönülüyor
            # Dikkat: Bu, 'evaluate' script'inin eski, daha az sağlam hali olabilir.
            # Ancak V1 ile tutarlılık için geri alınıyor.
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=MAX_LEN,
                num_beams=4,
                early_stopping=True
            )
            
            pred_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            labels[labels == -100] = tokenizer.pad_token_id
            ref_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)

            all_predictions.extend(pred_texts)
            all_references.extend(ref_texts)

    # --- SONUÇLARI HESAPLAMA VE KAYDETME ---
    logging.info("METEOR skoru hesaplanıyor...")
    results = meteor.compute(predictions=all_predictions, references=all_references)

    logging.info(f"Modelin Ortalama METEOR Skoru: {results['meteor']:.4f}")

    results_dir = config['artifacts']['results_dir']
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=4)
    logging.info(f"Metrikler '{results_dir}/metrics.json' dosyasına kaydedildi.")

if __name__ == "__main__":
    main()
