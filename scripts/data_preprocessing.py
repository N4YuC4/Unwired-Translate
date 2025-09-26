import os
import yaml
import numpy as np
from transformers import MT5Tokenizer
from sklearn.model_selection import train_test_split

def read_lines_efficiently(filename, max_lines):
    lines = []
    try:
        with open(filename, mode='rt', encoding='utf-8') as file:
            for i, line in enumerate(file):
                if max_lines > 0 and i >= max_lines:
                    break
                lines.append(line.strip())
    except FileNotFoundError:
        print(f"Hata: Dosya {filename} konumunda bulunamadı")
        return None
    return lines

def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    print("Veri setleri okunuyor...")
    en_data = read_lines_efficiently(
        os.path.join(config['dataset']['path'], "HPLT-en.txt"),
        config['dataset']['max_lines']
    )
    tr_data = read_lines_efficiently(
        os.path.join(config['dataset']['path'], "HPLT-tr.txt"),
        config['dataset']['max_lines']
    )

    if not en_data or not tr_data:
        print("Veri okuma hatası. İşlem durduruldu.")
        return

    print("Tokenizasyon başlatılıyor...")
    tokenizer = MT5Tokenizer.from_pretrained(config['model']['base_model'])
    inputs = ["translate English to Turkish: " + text for text in en_data]

    tokenized_inputs = tokenizer(inputs, max_length=config['model']['max_length'], truncation=True, padding="max_length", return_tensors="np")
    tokenized_targets = tokenizer(tr_data, max_length=config['model']['max_length'], truncation=True, padding="max_length", return_tensors="np")

    print("Eğitim ve test verileri ayrılıyor...")
    input_ids_train, input_ids_test, \
    attention_mask_train, attention_mask_test, \
    labels_train, labels_test = train_test_split(
        tokenized_inputs['input_ids'],
        tokenized_inputs['attention_mask'],
        tokenized_targets['input_ids'],
        test_size=config['dataset']['test_size'],
        random_state=config['dataset']['random_state']
    )

    print("İşlenmiş veriler kaydediliyor...")
    output_dir = config['artifacts']['processed_data_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    np.savez(
        os.path.join(output_dir, "processed_data.npz"),
        input_ids_train=input_ids_train,
        input_ids_test=input_ids_test,
        attention_mask_train=attention_mask_train,
        attention_mask_test=attention_mask_test,
        labels_train=labels_train,
        labels_test=labels_test
    )
    print(f"Veriler başarıyla '{output_dir}' dizinine kaydedildi.")

if __name__ == "__main__":
    main()