# %% [markdown]
#    # T5 Model Based EN-TR Machine Translation

# %%
!pip3 install -q bitsandbytes accelerate peft transformers evaluate pyarrow ftfy pytorch-lightning torch torchvision ctranslate2

import time
import os
import gc
import re
import ftfy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
import torch
import evaluate
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from bitsandbytes.optim import PagedAdamW

# Transformers ve PEFT
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

# --- PYTORCH LIGHTNING ---
import ctranslate2
import shutil
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import CSVLogger

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# %%
# --- Proje ve Model Ayarları ---
PROJE_ADI = "Translation"
VERI_SETI = ["lainchan-entries","hplt-dataset"] 
MODEL_MIMARISI = "mt5-small"
MODEL_TEKNIGI = "4bit-QLoRA"
VERSIYON = "V1.1"
MODEL_ADI = "google/" + MODEL_MIMARISI

# Veri
START_LINE = 0
MAX_LINES = 250*(10**3)
TEST_SIZE = 0.2
SOURCE_LANG = "English"
TARGET_LANG = "Turkish"
LANG_PAIR = f"{SOURCE_LANG}-{TARGET_LANG}"
ILK_SETI_TAMAMEN_OKU = True

# Eğitim
EPOCH_NUM = 4
LR = 5e-4 # Stabilite için düşürüldü (Eski: 2e-3)
GRADIENT_ACCUMULATION_STEPS = 4
PERCENTILE = 94
MAX_LEN = 128 # Bu aşağıda otomatik güncellenecek
GPU = True

# QLoRA
LORA_RANK = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05
TARGET_MODULES = "all-linear"

# Yollar
VERI_AD_STR = "-".join([x.split('-')[0].upper() for x in VERI_SETI])
OUTPUT_DIR = f"{MODEL_MIMARISI}_{MODEL_TEKNIGI}_{PROJE_ADI}_{VERI_AD_STR}_{VERSIYON}/{LANG_PAIR}"
os.makedirs(OUTPUT_DIR, exist_ok=True)
LOG_DIR = "logs"
# Ortam
if os.path.exists('/kaggle/input'):
    DATASET_PATH = ["/kaggle/input/"+i for i in VERI_SETI]
    TRAIN_BATCH_SIZE = 32
    CURRENT_ENV = 'Kaggle'
else:
    DATASET_PATH = ["../datasets/"+i for i in VERI_SETI]
    TRAIN_BATCH_SIZE = 18
    CURRENT_ENV = 'Local'

VAL_BATCH_SIZE = TRAIN_BATCH_SIZE * 2
DEVICE = "cuda" if torch.cuda.is_available() and GPU else "cpu"
print(f"Ortam: {CURRENT_ENV} | Cihaz: {DEVICE}")

# %%
# --- Veri Yükleme Fonksiyonu ---
def load_parquet(source_path, target_path, start=0, num=None):
    if not os.path.exists(source_path) or not os.path.exists(target_path):
        raise FileNotFoundError(f"Dosya bulunamadı: {source_path}")
    pf_src = pq.ParquetFile(source_path)
    pf_tgt = pq.ParquetFile(target_path)
    iter_src = pf_src.iter_batches()
    iter_tgt = pf_tgt.iter_batches()
    dfs = []
    collected = 0
    scanned = 0
    for b_src, b_tgt in zip(iter_src, iter_tgt):
        if num is not None and collected >= num: break
        b_rows = b_src.num_rows
        if scanned + b_rows < start:
            scanned += b_rows
            continue
        offset = max(0, start - scanned)
        sl_src = b_src.slice(offset=offset)
        sl_tgt = b_tgt.slice(offset=offset)
        if num is not None:
            needed = num - collected
            if sl_src.num_rows > needed:
                sl_src = sl_src.slice(length=needed)
                sl_tgt = sl_tgt.slice(length=needed)
        df_b = pd.DataFrame({"source": sl_src.to_pandas().iloc[:, 0], "target": sl_tgt.to_pandas().iloc[:, 0]})
        dfs.append(df_b)
        collected += df_b.shape[0]
        scanned += b_rows
    if not dfs: return pd.DataFrame(columns=["source", "target"])
    return pd.concat(dfs, ignore_index=True)

print(f"Veri yükleme başlatılıyor...")
dfs = []
total_rows = 0
for i, dset_name in enumerate(VERI_SETI):
    remaining = MAX_LINES - total_rows
    if not ILK_SETI_TAMAMEN_OKU and remaining <= 0: continue
    if ILK_SETI_TAMAMEN_OKU and i > 0 and remaining <= 0: continue
    ds_prefix = dset_name.split('-')[0].upper()
    src_p = DATASET_PATH[i] + f"/{ds_prefix}-{SOURCE_LANG}.parquet"
    tgt_p = DATASET_PATH[i] + f"/{ds_prefix}-{TARGET_LANG}.parquet"
    num_to_read = None if (ILK_SETI_TAMAMEN_OKU and i == 0) else remaining
    df_temp = load_parquet(src_p, tgt_p, start=START_LINE, num=num_to_read)
    df_temp['source_lang'] = SOURCE_LANG
    df_temp['target_lang'] = TARGET_LANG
    dfs.append(df_temp)
    total_rows += len(df_temp)

df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=["source", "target", "source_lang", "target_lang"])
del dfs
gc.collect()

# %%
# ==============================================================================
# EKSİK OLAN PARÇA: VERİYİ TERS ÇEVİRİP EKLEME (ÇİFT YÖNLÜ EĞİTİM)
# ==============================================================================
if not df.empty:
    print(f"Orijinal Veri Sayısı (Tek Yönlü): {len(df)}")
    print("Veri seti ters çevriliyor (Mirroring)...")

    # 1. Verinin kopyasını al
    df_reverse = df.copy()

    # 2. Sütunların ismini değiştir (Source <-> Target yer değişir)
    #    Artık Source sütununda Türkçe, Target sütununda İngilizce olacak
    df_reverse = df_reverse.rename(columns={'source': 'target', 'target': 'source'})

    # 3. Dil etiketlerini değiştir
    #    Artık Source Lang: Turkish, Target Lang: English olacak
    df_reverse['source_lang'] = TARGET_LANG  # Turkish
    df_reverse['target_lang'] = SOURCE_LANG  # English

    # 4. İki tabloyu alt alta birleştir
    df = pd.concat([df, df_reverse], ignore_index=True)

    # 5. Veriyi karıştır (Shuffle) - Bu çok önemli, model ezberlemesin
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Yeni Toplam Veri Sayısı (Çift Yönlü): {len(df)}")
    
    # Kontrol için ekrana bas
    print(f"Örnek 1: {df.iloc[0]['source_lang']} -> {df.iloc[0]['target_lang']}")
    print(f"Girdi: {df.iloc[0]['source']}")
    print(f"Hedef: {df.iloc[0]['target']}")
# ==============================================================================

# %%
print(f"Toplam {len(df)} satır çift yönlü veri yüklendi ({SOURCE_LANG}->{TARGET_LANG} + {TARGET_LANG}->{SOURCE_LANG}).")
print("-" * 40)
print(df.head())
print(f"\nVeri setinin sonundan örnekler ({TARGET_LANG}->{SOURCE_LANG} kontrolü):")
print(df.tail())
print("-" * 40)
print(df.info())

# %%
# --- Veri Önişleme ---
def normalize_text(text):
    text = ftfy.fix_text(text)
    text = re.sub(r'([!?.])\1+', r'\1', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

df['source'] = df['source'].apply(normalize_text)
df['target'] = df['target'].apply(normalize_text)

# --- Tokenizer ve Dataset ---
tokenizer = MT5Tokenizer.from_pretrained(MODEL_ADI)
train_df, val_df = train_test_split(
    df, 
    test_size=TEST_SIZE, 
    random_state=42, 
    stratify=df["source_lang"]
)

print(f"Train Seti Dil Dağılımı:\n{train_df['source_lang'].value_counts(normalize=True)}")
print(f"Val Seti Dil Dağılımı:\n{val_df['source_lang'].value_counts(normalize=True)}")

# Otomatik Max Len
all_texts = pd.concat([train_df["source"], train_df["target"]]).tolist()
token_lengths = [len(tokenizer(t, truncation=False)["input_ids"]) for t in tqdm(all_texts[:1000], desc="Len Calc")] # Hız için ilk 1000
MAX_LEN = int(np.clip(np.percentile(token_lengths, PERCENTILE), 32, 256))
print(f"Otomatik MAX_LEN: {MAX_LEN}")

# %%
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
        src_text = f"translate {self.src_langs[idx]} to {self.tgt_langs[idx]}: " + self.src_texts[idx]
        source = self.tokenizer(src_text, truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")
        target = self.tokenizer(self.tgt_texts[idx], truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")
        labels = target["input_ids"]
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {"input_ids": source["input_ids"].squeeze(), "attention_mask": source["attention_mask"].squeeze(), "labels": labels.squeeze()}

train_dataset = TranslationDataset(tokenizer, train_df["source_lang"].tolist(), train_df["target_lang"].tolist(), train_df["source"].tolist(), train_df["target"].tolist(), MAX_LEN)
val_dataset = TranslationDataset(tokenizer, val_df["source_lang"].tolist(), val_df["target_lang"].tolist(), val_df["source"].tolist(), val_df["target"].tolist(), MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# %%
# 4-bit konfigürasyonu KALDIRILDI (Kökten çözüm için standart LoRA'ya geçildi)
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16,
#     bnb_4bit_use_double_quant=True,
# )

print("Temel model Float16 (16-bit) olarak yükleniyor (No Quantization)...")
base_model = MT5ForConditionalGeneration.from_pretrained(
    MODEL_ADI,
    # quantization_config=bnb_config, # 4-bit iptal
    torch_dtype=torch.float16,       # 16-bit hassasiyet
    device_map="auto",
    low_cpu_mem_usage=True
)
print("Model yüklendi.")

# QLoRA hazırlığı yerine standart LoRA hazırlığı
# base_model = prepare_model_for_kbit_training(base_model) # Bu sadece kbit içindi
base_model.gradient_checkpointing_enable() # Bellek tasarrufu için
base_model.enable_input_require_grads()    # LoRA için gerekli

lora_config = LoraConfig(
    r=LORA_RANK, lora_alpha=LORA_ALPHA, target_modules=TARGET_MODULES, 
    lora_dropout=LORA_DROPOUT, bias="none", task_type="SEQ_2_SEQ_LM"
)
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()

# %%
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
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
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
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self._current_val_epoch_loss.append(loss.item())
        return loss

    def on_validation_epoch_end(self):
        if self._current_val_epoch_loss:
            avg_loss = sum(self._current_val_epoch_loss) / len(self._current_val_epoch_loss)
            self.val_losses_history.append(avg_loss)
            self._current_val_epoch_loss = []

    # --- KRİTİK: METEOR İÇİN PREDICT STEP ---
    def predict_step(self, batch, batch_idx):
        # 1. Üretim (Generation)
        generated_ids = self.model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_length=self.max_len,
            num_beams=4,
            early_stopping=True
        )
        # 2. Çözümleme (Decoding)
        preds = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        # 3. Referansları Çözümleme
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

# %%
# --- 1. GELİŞMİŞ LOGLAMA CALLBACK'İ ---
class AdvancedLoggingCallback(Callback):
    def __init__(self, log_every_n_steps=50):
        self.log_every_n_steps = log_every_n_steps
        self.step_start_time = None
        self.latest_val_loss = None 

    def format_time(self, seconds):
        """Saniyeyi Dakika:Saniye (MM:SS) formatına çevirir."""
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h:d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    def on_train_epoch_start(self, trainer, pl_module):
        print(f"\n{'#'*60}")
        print(f"🚀 EPOCH {trainer.current_epoch + 1}/{trainer.max_epochs} BAŞLATILIYOR...")
        print(f"{'#'*60}\n")
        self.epoch_start_time = time.time()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        # Anlık hız hesabı için adım başlangıcını tut
        if batch_idx % self.log_every_n_steps == 0:
            self.step_start_time = time.time()
            
    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        if "val_loss" in metrics:
            self.latest_val_loss = metrics["val_loss"].item()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % self.log_every_n_steps == 0:
            # --- 1. Metrikler ---
            loss = outputs["loss"].item() if isinstance(outputs, dict) else outputs.item()
            val_msg = f"{self.latest_val_loss:.4f}" if self.latest_val_loss is not None else "..."
            current_lr = trainer.optimizers[0].param_groups[0]['lr']
            vram_used = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            
            # --- 2. Süre Hesaplamaları (YENİ) ---
            # A) Epoch içinde geçen süre
            now = time.time()
            elapsed_epoch = now - self.epoch_start_time
            
            # B) İlerleme ve Toplam Batch
            total_batches = trainer.num_training_batches
            current_step = batch_idx + 1 # 0 tabanlı olduğu için +1
            progress = (current_step / total_batches) * 100
            
            # C) Ortalama Hız ve ETA (Kalan Süre)
            if current_step > 0:
                avg_time_per_batch = elapsed_epoch / current_step
                remaining_batches = total_batches - current_step
                eta_seconds = remaining_batches * avg_time_per_batch
            else:
                eta_seconds = 0
                
            # D) Formatlama (MM:SS)
            elapsed_str = self.format_time(elapsed_epoch)
            eta_str = self.format_time(eta_seconds)
            
            # Anlık adım hızı (saniye/iterasyon)
            step_time = now - self.step_start_time if self.step_start_time else 0
            
            # --- LOG ÇIKTISI ---
            # Örn: Time: 02:15<14:30 (Geçen < Kalan)
            print(
                f"[Ep {trainer.current_epoch + 1}] "
                f"[{progress:.1f}%] "
                f"Step {batch_idx}/{total_batches} | "
                f"Time: {elapsed_str}<{eta_str} | " # <-- YENİ KISIM
                f"L_Tr: {loss:.4f} | "
                f"L_Val: {val_msg} | "
                f"LR: {current_lr:.2e} | "
                f"VRAM: {vram_used:.2f}GB"
            )

    def on_train_epoch_end(self, trainer, pl_module):
        elapsed = time.time() - self.epoch_start_time
        elapsed_str = self.format_time(elapsed)
        val_msg = f"{self.latest_val_loss:.4f}" if self.latest_val_loss is not None else "Hesaplanmadı"
        print(f"\n✅ Epoch {trainer.current_epoch + 1} Bitti. Süre: {elapsed_str}. Son Val Loss: {val_msg}")
        print("-" * 60)

# --- 2. VRAM TEMİZLEME CALLBACK'İ ---
class VRAMCleanupCallback(Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % 20 == 0:
            torch.cuda.empty_cache()
            gc.collect()
            
    def on_validation_end(self, trainer, pl_module):
        torch.cuda.empty_cache()
        gc.collect()

# Adım Hesaplamaları
total_optimizer_steps = (len(train_loader) // GRADIENT_ACCUMULATION_STEPS) * EPOCH_NUM
warmup_steps = int(0.1 * total_optimizer_steps)

# Lightning Modülünü Başlat
lightning_model = MT5LightningModule(
    model=model,
    tokenizer=tokenizer,
    lr=LR,
    total_steps=total_optimizer_steps,
    warmup_steps=warmup_steps,
    max_len=MAX_LEN
)

# --- 3. TRAINER YAPILANDIRMASI ---
trainer = pl.Trainer(
    max_epochs=EPOCH_NUM,
    accelerator="gpu" if torch.cuda.is_available() and GPU else "cpu",
    devices=1,
    precision="16-mixed", # Mixed Precision: Hız ve VRAM avantajı sağlar, NaN riskini azaltır.
    accumulate_grad_batches=GRADIENT_ACCUMULATION_STEPS,
    gradient_clip_val=0.5, # Daha agresif clipping: Patlayan gradyanları önler.
    
    num_sanity_val_steps=0, 
    enable_progress_bar=False, 
    
    callbacks=[
        AdvancedLoggingCallback(log_every_n_steps=50), 
        VRAMCleanupCallback() 
    ],
    
    logger=CSVLogger(save_dir=LOG_DIR, name=PROJE_ADI),
    enable_model_summary=True
)

print("PyTorch Lightning Eğitimi Başlıyor (Commit Log Modu)...")
trainer.fit(model=lightning_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
print("Eğitim Tamamlandı.")

# 1. LOSS'LARI KAYDET
train_losses = lightning_model.train_losses_history
val_losses = lightning_model.val_losses_history

# Modeli Lightning'den geri al (Adapter kaydı için)
model = lightning_model.model

# %% [markdown]
# ## 2. MODELİ BİRLEŞTİR VE CTRANSLATE2 FORMATINA DÖNÜŞTÜR

# %% [markdown]
# ### 2.1 Adapte'ı Geçici Kaydet

# %%
print("Model CTranslate2 formatına dönüştürülüyor...")

temp_adapter_path = "temp_adapter"
print(f"Geçici adapter kaydediliyor: {temp_adapter_path}")
model.save_pretrained(temp_adapter_path)
tokenizer.save_pretrained(temp_adapter_path)

# %% [markdown]
# ### 2.2 Belleği Temizle (Trainer ve eski modelleri sil)

# %%
print("Hafıza temizleniyor...")
try:
    del lightning_model
    del trainer
    del model
    del base_model
except:
    pass

gc.collect()
torch.cuda.empty_cache()
print("RAM ve VRAM temizlendi.")

# %% [markdown]
# ### 2.3 Base Modeli Yükle (CPU, Float32 - Merge için)

# %%
print("Base model (CPU) yükleniyor...")
base_model = MT5ForConditionalGeneration.from_pretrained(
    MODEL_ADI,
    device_map="cpu",
    torch_dtype=torch.float32
)

# %% [markdown]
# ### 2.4 Adapter ile Birleştir

# %%
print("LoRA adapter birleştiriliyor...")
model_to_merge = PeftModel.from_pretrained(base_model, temp_adapter_path)
merged_model = model_to_merge.merge_and_unload()

# %% [markdown]
# ### 2.5 Birleşmiş Modeli Kaydet (Geçici)

# %%
merged_output_path = "temp_merged_model"
print(f"Birleştirilmiş model kaydediliyor: {merged_output_path}")
merged_model.save_pretrained(merged_output_path)
tokenizer.save_pretrained(merged_output_path)

# %% [markdown]
# ### 2.6 CTranslate2 Dönüştürme

# %%
# OUTPUT_DIR artık CT2 modelinin yolu olacak
ct2_output_path = OUTPUT_DIR 
print(f"CTranslate2 formatına dönüştürülüyor (int8): {ct2_output_path}")

converter = ctranslate2.converters.TransformersConverter(merged_output_path)
converter.convert(
    output_dir=ct2_output_path,
    quantization="int8",
    force=True
)
print("Dönüştürme tamamlandı.")

# Final tokenizer'ı da CT2 modeli ile aynı yere kaydet
tokenizer.save_pretrained(ct2_output_path)

# %% [markdown]
# ### 2.7 Temizlik

# %%
print("Geçici dosyalar temizleniyor...")
shutil.rmtree(temp_adapter_path, ignore_errors=True)
shutil.rmtree(merged_output_path, ignore_errors=True)
print("Temizlik tamamlandı.")

# %% [markdown]
# ## 3. CTRANSLATE2 İLE METEOR HESAPLAMA

# %%
print(f"\nCTranslate2 Modeli Yükleniyor: {ct2_output_path}")
translator = ctranslate2.Translator(ct2_output_path, device=DEVICE)
tokenizer = MT5Tokenizer.from_pretrained(ct2_output_path)

print("METEOR Skoru Hesaplanıyor (CTranslate2 ile)...")
meteor = evaluate.load('meteor')

start_time = time.time()
all_predictions = []
all_references = []

# Validation loader üzerinden geç
# Not: CT2 batch işlemi için dataloader'dan raw text almak daha kolay olabilir ama
# burada dataloader tensor döndürüyor. Dataset'ten direkt alalım.

# Val dataset'ten metinleri alalım (Tensor dönüşümü olmadan)
val_src_texts = val_dataset.src_texts
val_tgt_texts = val_dataset.tgt_texts
val_src_langs = val_dataset.src_langs
val_tgt_langs = val_dataset.tgt_langs

# Batch işlemleri için
BATCH_SIZE = 32
total_samples = len(val_src_texts)

for i in tqdm(range(0, total_samples, BATCH_SIZE), desc="CT2 Inference"):
    batch_src = val_src_texts[i:i+BATCH_SIZE]
    batch_tgt = val_tgt_texts[i:i+BATCH_SIZE]
    batch_src_lang = val_src_langs[i:i+BATCH_SIZE]
    batch_tgt_lang = val_tgt_langs[i:i+BATCH_SIZE]
    
    # Prompt hazırlama
    prompts = [f"translate {sl} to {tl}: " + txt for sl, tl, txt in zip(batch_src_lang, batch_tgt_lang, batch_src)]
    
    # Tokenize
    source_tokens = [tokenizer.tokenize(p) for p in prompts]
    
    # Translate
    results = translator.translate_batch(
        source_tokens,
        batch_type="examples",
        max_batch_size=BATCH_SIZE,
        beam_size=4,
        max_decoding_length=MAX_LEN
    )
    
    # Decode
    preds = [tokenizer.convert_tokens_to_string(res.hypotheses[0]) for res in results]
    
    all_predictions.extend(preds)
    all_references.extend(batch_tgt)

end_time = time.time()
elapsed = end_time - start_time
print(f">> Tahmin tamamlandı. Geçen Süre: {elapsed:.1f} saniye ({elapsed/60:.1f} dakika).")

results = meteor.compute(predictions=all_predictions, references=all_references)
meteor_score = results['meteor']
print(f"METEOR Skoru: {meteor_score:.4f}")

# %% [markdown]
# ## 4. GRAFİKLERİ ÇİZ (Kayıp Grafiği)

# %%
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(range(1, len(train_losses) + 1), train_losses, label='Eğitim Kaybı', color='blue')
if val_losses:
    ax.plot(range(1, len(val_losses) + 1), val_losses, label='Doğrulama Kaybı', color='red', linestyle='--')

ax.set_title(f'Model Kayıp Grafiği')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend()
ax.grid(True, alpha=0.6)

# METEOR skorunu grafiğe ekle
try:
    ax.text(0.05, 0.95, f"METEOR (CT2): {meteor_score:.4f}", transform=ax.transAxes, 
            bbox=dict(facecolor='wheat', alpha=0.5), verticalalignment='top')
except: 
    pass

plt.savefig(os.path.join(OUTPUT_DIR, "loss_graph.png"))
plt.show()

# %% [markdown]
# ## 5. INFERENCE FONKSİYONU VE ÖRNEKLER

# %%
def translate(input_texts, source_lang, target_lang, max_length=128):
    # Tekil string gelirse listeye çevir
    is_single = isinstance(input_texts, str)
    if is_single:
        input_texts = [input_texts]

    prompts = [f"translate {source_lang} to {target_lang}: " + text for text in input_texts]
    
    source_tokens = [tokenizer.tokenize(p) for p in prompts]
    
    results = translator.translate_batch(
        source_tokens,
        batch_type="examples",
        max_batch_size=8,
        beam_size=4,
        max_decoding_length=max_length
    )
    
    translations = []
    for res in results:
        decoded = tokenizer.convert_tokens_to_string(res.hypotheses[0])
        translations.append(decoded)

    return translations[0] if is_single else translations

# --- Örnek Kullanım ---

print("\n" + "="*30 + "\n")
print("TEST ÇEVİRİLERİ:")

# Tek bir cümle çevirme
sample_single = "This is a test sentence for translation."
translation_single = translate(sample_single, SOURCE_LANG, TARGET_LANG)
print(f"\n{SOURCE_LANG}: {sample_single}")
print(f"{TARGET_LANG}: {translation_single}")

print("\n" + "-"*30 + "\n")

sample_single = "Bu, çeviri için bir test cümlesidir."
translation_single = translate(sample_single, TARGET_LANG, SOURCE_LANG)
print(f"{TARGET_LANG}: {sample_single}")
print(f"{SOURCE_LANG}: {translation_single}")

print("\n" + "="*30 + "\n")

# Birden fazla cümleyi toplu çevirme
samples_list = [
    "I want to learn machine translation with PyTorch and PEFT.",
    "The weather is very nice today, isn't it?",
    "We are developing a translation model using 4-bit QLoRA.",
    "How can I improve the METEOR score further?",
    "This is a Turkish test sentence for the bidirectional model."
]

print("Toplu çeviri başlıyor (EN->TR)...")
translations_list = translate(samples_list,SOURCE_LANG,TARGET_LANG)

# Sonuçları yazdırma
for source, target in zip(samples_list, translations_list):
    print(f"\n - {SOURCE_LANG}: {source}\n + {TARGET_LANG}: {target}")


print("\n" + "-"*30 + "\n")

# Birden fazla cümleyi toplu çevirme (TR->EN)
samples_list = [
    "PyTorch ve PEFT ile makine çevirisi öğrenmek istiyorum.",
    "Bugün hava çok güzel, değil mi?",
    "4 bit QLoRA kullanarak bir çeviri modeli geliştiriyoruz.",
    "METEOR puanını nasıl daha da artırabilirim?",
    "Bu, çift yönlü model için bir Türkçe test cümlesidir."
]

print("Toplu çeviri başlıyor (TR->EN)...")
translations_list = translate(samples_list, TARGET_LANG, SOURCE_LANG)

# Sonuçları yazdırma
for source, target in zip(samples_list, translations_list):
    print(f"\n - {TARGET_LANG}: {source}\n + {SOURCE_LANG}: {target}")


