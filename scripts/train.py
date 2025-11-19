# Gerekli kütüphanelerin import edilmesi
import os
import yaml
import gc
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    MT5Tokenizer,
    T5ForConditionalGeneration,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from bitsandbytes.optim import PagedAdamW
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import logging
from datetime import datetime

# --- LOGLAMA AYARLARI ---
log_dir = "logs/train"
os.makedirs(log_dir, exist_ok=True)
log_filename = f"{log_dir}/train_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
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
        # V1 modeli prefix olmadan eğitildiği için, doğrudan metni kullanıyoruz.
        src_text = self.src_texts[idx]
        tgt_text = self.tgt_texts[idx]

        source = self.tokenizer(
            src_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        target = self.tokenizer(
            tgt_text,
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
            "labels": labels.squeeze(),
        }

# --- ANA EĞİTİM FONKSİYONU ---
def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    DEVICE = "cuda" if torch.cuda.is_available() and config['system']['gpu'] else "cpu"
    logging.info(f"Kullanılacak Cihaz: {DEVICE}")

    logging.info("İşlenmiş veriler yükleniyor...")
    processed_data_dir = config['artifacts']['processed_data_dir']
    train_df = pd.read_parquet(os.path.join(processed_data_dir, "train.parquet"))
    val_df = pd.read_parquet(os.path.join(processed_data_dir, "test.parquet"))

    tokenizer = MT5Tokenizer.from_pretrained(config['model_adi'])

    MAX_LEN = config['training']['max_len']

    # Eğitim ve doğrulama için TranslationDataset nesnelerini oluştur (V1 UYUMLU)
    train_dataset = TranslationDataset(tokenizer, train_df["source"].tolist(), train_df["target"].tolist(), MAX_LEN)
    val_dataset = TranslationDataset(tokenizer, val_df["source"].tolist(), val_df["target"].tolist(), MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=config['training']['train_batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['train_batch_size']*2, shuffle=False, num_workers=4, pin_memory=True)

    logging.info(f"Eğitim için {len(train_dataset)}, doğrulama için {len(val_dataset)} örnek oluşturuldu.")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    logging.info("Temel model 4-bit olarak yükleniyor...")
    model = T5ForConditionalGeneration.from_pretrained(
        config['model_adi'],
        quantization_config=bnb_config,
        device_map="auto"
    )
    logging.info("Model yüklendi.")

    model = prepare_model_for_kbit_training(model)

    lora_config_params = config['qlora']
    lora_config = LoraConfig(
        r=lora_config_params['lora_rank'],
        lora_alpha=lora_config_params['lora_alpha'],
        target_modules=lora_config_params['target_modules'],
        lora_dropout=lora_config_params['lora_dropout'],
        bias="none",
        task_type="SEQ_2_SEQ_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_params = config['training']
    optimizer = PagedAdamW(model.parameters(), lr=training_params['lr'])
    
    total_optimizer_steps = (len(train_loader) // training_params['gradient_accumulation_steps']) * training_params['epochs']
    warmup_steps = int(0.1 * total_optimizer_steps)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_optimizer_steps
    )

    train_losses = []
    val_losses = []

    logging.info("4-bit QLoRA Eğitimi başlıyor...")

    for epoch in range(training_params['epochs']):
        model.train()
        total_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{training_params['epochs']} [Eğitim]")

        for i, batch in enumerate(loop):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            
            loss = loss / training_params['gradient_accumulation_steps']
            
            loss.backward()
            
            unnormalized_loss = loss.item() * training_params['gradient_accumulation_steps']
            total_loss += unnormalized_loss
            loop.set_postfix(loss=unnormalized_loss)

            if (i + 1) % training_params['gradient_accumulation_steps'] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        logging.info(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f}")

        model.eval()
        val_loss = 0
        with torch.no_grad():
            val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{training_params['epochs']} [Doğrulama]")
            for batch in val_loop:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                outputs = model(**batch)
                val_loss += outputs.loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        logging.info(f"Validation Loss: {avg_val_loss:.4f}")

        logging.info(f"Epoch {epoch+1} sonunda bellek temizleniyor...")
        torch.cuda.empty_cache()
        gc.collect()

    logging.info("\nEğitim tamamlandı!")

    # --- MODELİ VE SONUÇLARI KAYDETME ---
    lora_sys_config = config['system']['lora']
    base_dir = lora_sys_config['base_dir'].format(
        model_mimarisi=config['model_mimarisi'],
        model_teknigi=config['model_teknigi'],
        proje_adi=config['proje_adi'],
        veri_seti=config['veri_seti'],
        versiyon=config['versiyon']
    )
    lang_dir = lora_sys_config['lang_dir'].format(
        source_lang=config['language']['source'],
        target_lang=config['language']['target']
    )
    model_output_dir = os.path.join(config['system']['output_dir'], base_dir, lang_dir)

    logging.info(f"\nSADECE LoRa adaptör ağırlıkları '{model_output_dir}' dizinine kaydediliyor...")
    model.save_pretrained(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)
    logging.info("Kaydetme işlemi tamamlandı.")

    results_dir = config['artifacts']['results_dir']
    os.makedirs(results_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(train_losses) + 1), train_losses, label='Eğitim Kaybı', color='blue')
    ax.plot(range(1, len(val_losses) + 1), val_losses, label='Doğrulama Kaybı', color='red', linestyle='--')
    ax.set_title(f'Model Kayıp Grafiği ({base_dir}/{lang_dir})', fontsize=16, fontweight='bold')
    ax.set_ylabel('Kayıp Değeri', fontsize=12)
    ax.set_xlabel('Epok Sayısı', fontsize=12)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend(loc='upper right')
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(os.path.join(results_dir, 'loss_graph.png'))
    logging.info(f"Grafik '{results_dir}' dizinine kaydedildi.")

if __name__ == "__main__":
    main()
