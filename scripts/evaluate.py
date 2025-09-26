import os
import yaml
import json
import numpy as np
import tensorflow as tf
from transformers import TFMT5ForConditionalGeneration, MT5Tokenizer
import evaluate
from tqdm import tqdm

def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    print("Test verisi ve model yükleniyor...")
    data = np.load(os.path.join(config['artifacts']['processed_data_dir'], "processed_data.npz"))
    model_dir = config['artifacts']['model_output_dir']
    
    model = TFMT5ForConditionalGeneration.from_pretrained(os.path.join(model_dir, "model"))
    tokenizer = MT5Tokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    
    test_dataset = tf.data.Dataset.from_tensor_slices({
        'input_ids': data['input_ids_test'],
        'attention_mask': data['attention_mask_test'],
        'labels': data['labels_test']
    }).batch(config['training']['batch_size'], drop_remainder=True).prefetch(tf.data.AUTOTUNE)

    print("METEOR metriği yükleniyor...")
    meteor = evaluate.load('meteor')
    
    all_predictions = []
    all_references = []
    
    print("Çeviriler oluşturuluyor...")
    for batch in tqdm(test_dataset):
        outputs = model.generate(
            input_ids=batch['input_ids'], 
            attention_mask=batch['attention_mask'],
            max_length=config['model']['max_length'],
            num_beams=4
        )
        pred_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        labels = tf.where(batch['labels'] == -100, tokenizer.pad_token_id, batch['labels'])
        ref_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        all_predictions.extend(pred_texts)
        all_references.extend(ref_texts)

    print("METEOR skoru hesaplanıyor...")
    results = meteor.compute(predictions=all_predictions, references=all_references)
    
    print(f"Modelin Ortalama METEOR Skoru: {results['meteor']:.4f}")
    
    results_dir = config['artifacts']['results_dir']
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=4)
    print(f"Metrikler '{results_dir}/metrics.json' dosyasına kaydedildi.")

if __name__ == "__main__":
    main()