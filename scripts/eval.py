import os
import yaml
import json
import pandas as pd
import torch
import evaluate
from tqdm import tqdm
from transformers import (
    MT5Tokenizer,
    MT5ForConditionalGeneration,
    BitsAndBytesConfig,
)
from peft import PeftModel
from torch.utils.data import Dataset, DataLoader
import logging
from datetime import datetime

# --- LOGLAMA AYARLARI ---
log_dir = "logs/evaluate"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- DATASET SINIFI ---
class TranslationDataset(Dataset):
    def __init__(self, tokenizer, src_langs, tgt_langs, src_texts, tgt_texts, max_len=128):
        self.tokenizer = tokenizer
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.src_langs = src_langs
        self.tgt_langs = tgt_langs
        self.max_len = max_len

    def __len__(self): return len(self.src_texts)

    def __getitem__(self, idx):
        src_text = f"translate {self.src_langs[idx]} to {self.tgt_langs[idx]}: " + str(self.src_texts[idx])
        
        source = self.tokenizer(
            src_text, 
            truncation=True, 
            padding="max_length", 
            max_length=self.max_len, 
            return_tensors="pt"
        )
        target = self.tokenizer(
            str(self.tgt_texts[idx]), 
            truncation=True, 
            padding="max_length", 
            max_length=self.max_len, 
            return_tensors="pt"
        )
        
        labels = target["input_ids"]
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": source["input_ids"].squeeze(), 
            "attention_mask": source["attention_mask"].squeeze(), 
            "labels": labels.squeeze()
        }

# --- ANA FONKSİYON ---
def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    DEVICE = "cuda" if torch.cuda.is_available() and config['system']['gpu'] else "cpu"

    logging.info("Test verisi yükleniyor...")
    processed_dir = config['artifacts']['processed_data_dir']
    test_df = pd.read_parquet(os.path.join(processed_dir, "test.parquet"))

    # Model Yolu Oluşturma
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

    # --- MODEL YÜKLEME ---
    logging.info(f"Model yükleniyor: {model_dir}")
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

    MAX_LEN = config['training']['max_len']
    
    test_dataset = TranslationDataset(
        tokenizer, 
        test_df["source_lang"].tolist(), test_df["target_lang"].tolist(),
        test_df["source"].tolist(), test_df["target"].tolist(), 
        MAX_LEN
    )
    # eval_batch_size veya train_batch_size*2
    batch_size = config['training'].get('eval_batch_size', config['training']['train_batch_size'] * 4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    logging.info("METEOR hesaplanıyor...")
    meteor = evaluate.load('meteor')
    all_predictions = []
    all_references = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Tahmin Ediliyor"):
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

    results = meteor.compute(predictions=all_predictions, references=all_references)
    logging.info(f"METEOR Skoru: {results['meteor']:.4f}")

    results_dir = config['artifacts']['results_dir']
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "metrics_eval.json"), "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
