import os
import yaml
import numpy as np
import tensorflow as tf
from transformers import TFMT5ForConditionalGeneration, MT5Tokenizer
import matplotlib.pyplot as plt

def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    print("İşlenmiş veriler yükleniyor...")
    data = np.load(os.path.join(config['artifacts']['processed_data_dir'], "processed_data.npz"))
    
    input_ids_train = data['input_ids_train']
    attention_mask_train = data['attention_mask_train']
    labels_train = data['labels_train']
    input_ids_test = data['input_ids_test']
    attention_mask_test = data['attention_mask_test']
    labels_test = data['labels_test']

    print("tf.data pipeline'ları oluşturuluyor...")
    train_dataset = tf.data.Dataset.from_tensor_slices({
        'input_ids': input_ids_train,
        'attention_mask': attention_mask_train,
        'labels': labels_train
    }).shuffle(len(input_ids_train)).batch(config['training']['batch_size']).prefetch(tf.data.AUTOTUNE)

    test_dataset = tf.data.Dataset.from_tensor_slices({
        'input_ids': input_ids_test,
        'attention_mask': attention_mask_test,
        'labels': labels_test
    }).batch(config['training']['batch_size'], drop_remainder=True).prefetch(tf.data.AUTOTUNE)

    print("Model yükleniyor...")
    model = TFMT5ForConditionalGeneration.from_pretrained(config['model']['base_model'])
    optimizer = tf.keras.optimizers.Adam(learning_rate=config['training']['learning_rate'])
    model.compile(optimizer=optimizer)

    print("Model eğitimi başlıyor...")
    history = model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=config['training']['epochs']
    )
    print("Model eğitimi tamamlandı.")

    print("Model ve tokenizer kaydediliyor...")
    model_output_dir = config['artifacts']['model_output_dir']
    model.save_pretrained(os.path.join(model_output_dir, "model"))
    
    tokenizer = MT5Tokenizer.from_pretrained(config['model']['base_model'])
    tokenizer.save_pretrained(os.path.join(model_output_dir, "tokenizer"))
    print(f"Model '{model_output_dir}' dizinine kaydedildi.")
    
    print("Eğitim grafiği kaydediliyor...")
    results_dir = config['artifacts']['results_dir']
    os.makedirs(results_dir, exist_ok=True)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(os.path.join(results_dir, 'loss_graph.png'))
    print(f"Grafik '{results_dir}' dizinine kaydedildi.")

if __name__ == "__main__":
    main()