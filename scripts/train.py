import os
import yaml
import gc
import time
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import logging
from datetime import datetime
import evaluate

from torch.utils.data import Dataset, DataLoader
from transformers import (
    MT5Tokenizer,
    MT5ForConditionalGeneration,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel
)
from bitsandbytes.optim import PagedAdamW

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import CSVLogger

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# --- LOGLAMA AYARLARI ---
log_dir = "logs/train"
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
        # Prompt formatı: "translate English to Turkish: Hello world"
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

# --- LIGHTNING MODÜLÜ ---
class MT5LightningModule(pl.LightningModule):
    def __init__(self, model, tokenizer, lr, total_steps, warmup_steps, max_len):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.lr = lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.max_len = max_len
        
        self.train_losses_history = []
        self.val_losses_history = []
        self._current_train_epoch_loss = []
        self._current_val_epoch_loss = []

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=False) # Custom bar kullanacağız
        self._current_train_epoch_loss.append(loss.item())
        return loss

    def on_train_epoch_end(self):
        if self._current_train_epoch_loss:
            avg_loss = sum(self._current_train_epoch_loss) / len(self._current_train_epoch_loss)
            self.train_losses_history.append(avg_loss)
            self._current_train_epoch_loss = []

    def validation_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self._current_val_epoch_loss.append(loss.item())
        return loss

    def on_validation_epoch_end(self):
        if self._current_val_epoch_loss:
            avg_loss = sum(self._current_val_epoch_loss) / len(self._current_val_epoch_loss)
            self.val_losses_history.append(avg_loss)
            self._current_val_epoch_loss = []

    def predict_step(self, batch, batch_idx):
        generated_ids = self.model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_length=self.max_len,
            num_beams=4,
            early_stopping=True
        )
        preds = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        labels = batch["labels"]
        labels[labels == -100] = self.tokenizer.pad_token_id
        refs = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        return preds, refs

    def configure_optimizers(self):
        optimizer = PagedAdamW(self.model.parameters(), lr=self.lr)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=self.total_steps
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

# --- CALLBACKLER ---
class AdvancedLoggingCallback(Callback):
    def __init__(self, log_every_n_steps=50):
        self.log_every_n_steps = log_every_n_steps
        self.step_start_time = None
        self.epoch_start_time = None
        self.latest_val_loss = None

    def format_time(self, seconds):
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        if h > 0: return f"{h:d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    def on_train_epoch_start(self, trainer, pl_module):
        print(f"\n{'#'*60}")
        print(f"🚀 EPOCH {trainer.current_epoch + 1}/{trainer.max_epochs} BAŞLATILIYOR...")
        print(f"{'#'*60}\n")
        self.epoch_start_time = time.time()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if batch_idx % self.log_every_n_steps == 0:
            self.step_start_time = time.time()

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        if "val_loss" in metrics:
            self.latest_val_loss = metrics["val_loss"].item()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % self.log_every_n_steps == 0:
            loss = outputs["loss"].item() if isinstance(outputs, dict) else outputs.item()
            val_msg = f"{self.latest_val_loss:.4f}" if self.latest_val_loss is not None else "..."
            current_lr = trainer.optimizers[0].param_groups[0]['lr']
            vram_used = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            
            now = time.time()
            elapsed_epoch = now - self.epoch_start_time
            total_batches = trainer.num_training_batches
            current_step = batch_idx + 1
            progress = (current_step / total_batches) * 100
            
            if current_step > 0:
                avg_time_per_batch = elapsed_epoch / current_step
                remaining_batches = total_batches - current_step
                eta_seconds = remaining_batches * avg_time_per_batch
            else:
                eta_seconds = 0
            
            elapsed_str = self.format_time(elapsed_epoch)
            eta_str = self.format_time(eta_seconds)
            
            print(f"[Ep {trainer.current_epoch + 1}] [{progress:.1f}%] Step {batch_idx}/{total_batches} | "
                  f"Time: {elapsed_str}<{eta_str} | L_Tr: {loss:.4f} | L_Val: {val_msg} | "
                  f"LR: {current_lr:.2e} | VRAM: {vram_used:.2f}GB")

    def on_train_epoch_end(self, trainer, pl_module):
        elapsed = time.time() - self.epoch_start_time
        elapsed_str = self.format_time(elapsed)
        val_msg = f"{self.latest_val_loss:.4f}" if self.latest_val_loss is not None else "N/A"
        print(f"\n✅ Epoch {trainer.current_epoch + 1} Bitti. Süre: {elapsed_str}. Son Val Loss: {val_msg}")

class VRAMCleanupCallback(Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % 20 == 0:
            torch.cuda.empty_cache()
            gc.collect()
    def on_validation_end(self, trainer, pl_module):
        torch.cuda.empty_cache()
        gc.collect()

# --- ANA FONKSİYON ---
def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Config'den okuma
    train_params = config['training']
    dataset_config = config['dataset']
    model_name = config['model_adi']
    
    # 1. Veri Yükleme
    logging.info("İşlenmiş veriler yükleniyor...")
    processed_dir = config['artifacts']['processed_data_dir']
    train_df = pd.read_parquet(os.path.join(processed_dir, "train.parquet"))
    val_df = pd.read_parquet(os.path.join(processed_dir, "test.parquet"))
    logging.info(f"Train: {len(train_df)}, Val: {len(val_df)}")

    tokenizer = MT5Tokenizer.from_pretrained(model_name)

    # 2. Max Len Hesaplama
    MAX_LEN = train_params['max_len']
    if train_params.get('use_dynamic_max_len', False):
        logging.info("Dinamik MAX_LEN hesaplanıyor...")
        # Hız için örneklem al
        sample_texts = pd.concat([train_df["source"], train_df["target"]]).sample(min(2000, len(train_df)*2)).tolist()
        token_lengths = [len(tokenizer(t, truncation=False)["input_ids"]) for t in sample_texts]
        MAX_LEN = int(np.clip(np.percentile(token_lengths, train_params['percentile']), 32, 256))
        logging.info(f"Otomatik Hesaplanan MAX_LEN: {MAX_LEN}")
    
    # 3. Dataset ve Dataloader
    train_dataset = TranslationDataset(
        tokenizer, 
        train_df["source_lang"].tolist(), train_df["target_lang"].tolist(),
        train_df["source"].tolist(), train_df["target"].tolist(), 
        MAX_LEN
    )
    val_dataset = TranslationDataset(
        tokenizer, 
        val_df["source_lang"].tolist(), val_df["target_lang"].tolist(),
        val_df["source"].tolist(), val_df["target"].tolist(), 
        MAX_LEN
    )
    
    # eval_batch_size yoksa train_batch_size * 2 kullan
    eval_bs = train_params.get('eval_batch_size', train_params['train_batch_size'] * 2)
    
    train_loader = DataLoader(train_dataset, batch_size=train_params['train_batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=eval_bs, shuffle=False, num_workers=4, pin_memory=True)

    # 4. Model Hazırlığı
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    logging.info("Model yükleniyor (4-bit)...")
    base_model = MT5ForConditionalGeneration.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    base_model = prepare_model_for_kbit_training(base_model)
    
    qlora_params = config['qlora']
    lora_config = LoraConfig(
        r=qlora_params['lora_rank'],
        lora_alpha=qlora_params['lora_alpha'],
        target_modules=qlora_params['target_modules'],
        lora_dropout=qlora_params['lora_dropout'],
        bias="none",
        task_type="SEQ_2_SEQ_LM"
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    # 5. Lightning Trainer
    total_steps = (len(train_loader) // train_params['gradient_accumulation_steps']) * train_params['epochs']
    warmup_steps = int(0.1 * total_steps)
    
    lightning_model = MT5LightningModule(
        model=model,
        tokenizer=tokenizer,
        lr=train_params['lr'],
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        max_len=MAX_LEN
    )
    
    trainer = pl.Trainer(
        max_epochs=train_params['epochs'],
        accelerator="gpu" if torch.cuda.is_available() and config['system']['gpu'] else "cpu",
        devices=1,
        precision="32-true",
        accumulate_grad_batches=train_params['gradient_accumulation_steps'],
        enable_progress_bar=False,
        callbacks=[AdvancedLoggingCallback(), VRAMCleanupCallback()],
        logger=CSVLogger(save_dir="logs", name=config['proje_adi']),
        enable_model_summary=True
    )
    
    logging.info("Eğitim Başlıyor...")
    trainer.fit(lightning_model, train_loader, val_loader)
    logging.info("Eğitim Tamamlandı.")
    
    # 6. METEOR Hesaplama (Predict)
    logging.info("METEOR Skoru Hesaplanıyor...")
    predictions_and_refs = trainer.predict(lightning_model, dataloaders=val_loader)
    
    all_predictions = []
    all_references = []
    for batch_preds, batch_refs in predictions_and_refs:
        all_predictions.extend(batch_preds)
        all_references.extend(batch_refs)
        
    meteor = evaluate.load('meteor')
    results = meteor.compute(predictions=all_predictions, references=all_references)
    meteor_score = results['meteor']
    logging.info(f"METEOR Skoru: {meteor_score:.4f}")
    
    # 7. Kaydetme
    lora_sys_config = config['system']['lora']
    
    dataset_names = config['dataset'].get('names', [])
    if isinstance(dataset_names, list) and dataset_names:
        veri_seti_str = "-".join([str(x).upper() for x in dataset_names])
    else:
        veri_seti_str = config.get('veri_seti', 'Unknown')

    base_dir = lora_sys_config['base_dir'].format(
        model_mimarisi=config['model_mimarisi'],
        model_teknigi=config['model_teknigi'],
        proje_adi=config['proje_adi'],
        veri_seti=veri_seti_str,
        versiyon=config['versiyon']
    )
    lang_dir = lora_sys_config['lang_dir'].format(
        source_lang=config['language']['source'],
        target_lang=config['language']['target']
    )
    # Output dir düzeltmesi: models/MT5.../Turkish-English
    output_path = os.path.join(config['system']['output_dir'], base_dir, lang_dir)
    
    logging.info(f"Model kaydediliyor: {output_path}")
    model = lightning_model.model
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    # Grafik Kaydetme
    train_losses = lightning_model.train_losses_history
    val_losses = lightning_model.val_losses_history
    
    results_dir = config['artifacts']['results_dir']
    os.makedirs(results_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(train_losses) + 1), train_losses, label='Eğitim Kaybı', color='blue')
    if val_losses:
        ax.plot(range(1, len(val_losses) + 1), val_losses, label='Doğrulama Kaybı', color='red', linestyle='--')
    ax.set_title(f'Model Kayıp Grafiği\nMETEOR: {meteor_score:.4f}')
    ax.legend()
    ax.grid(True, alpha=0.6)
    plt.savefig(os.path.join(results_dir, 'loss_graph.png'))
    
    # Metrics JSON kaydetme
    import json
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump({"meteor": meteor_score, "final_train_loss": train_losses[-1] if train_losses else 0}, f)

if __name__ == "__main__":
    main()