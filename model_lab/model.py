import os
import sys
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tf2onnx
import onnx
import librosa 

# Add current dir to path for dsp.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from dsp import DSP
    print("✅ Loaded dsp.py")
except ImportError:
    from model_lab.dsp import DSP

# --- CONFIGURATION ---
SAMPLE_RATE = 16000
EPOCHS = 10
BATCH_SIZE = 64
INPUT_SHAPE = (30, 40, 1)

# The "Golden 10" (Matches C++ App/main.cpp)
TARGET_COMMANDS = ["down", "go", "left", "no", "off", "on", "right", "stop", "up", "yes"]

def get_dataset():
    print(f"\n1. Loading 'speech_commands' via TensorFlow Datasets...")
    dataset, info = tfds.load('speech_commands', with_info=True, as_supervised=True)
    
    all_label_names = info.features['label'].names
    label_map = {}
    for tfds_idx, name in enumerate(all_label_names):
        if name in TARGET_COMMANDS:
            label_map[tfds_idx] = TARGET_COMMANDS.index(name)
        else:
            label_map[tfds_idx] = -1

    dsp = DSP()
    X = []
    y = []
    
    print("2. Processing & Filtering Audio...")
    full_ds = dataset['train'].concatenate(dataset['validation']).concatenate(dataset['test'])
    iterator = tfds.as_numpy(full_ds)
    
    count = 0
    skipped = 0
    
    for audio, label_idx in iterator:
        mapped_label = label_map[label_idx]
        if mapped_label == -1:
            skipped += 1
            continue
            
        audio = audio.astype(np.float32) / 32768.0
        
        if len(audio) < SAMPLE_RATE:
            audio = np.pad(audio, (0, SAMPLE_RATE - len(audio)), 'constant')
        else:
            audio = audio[:SAMPLE_RATE]
            
        try:
            spectrogram = dsp.process(audio)
            if spectrogram.shape == (30, 40):
                X.append(spectrogram)
                y.append(mapped_label)
                count += 1
                if count % 2000 == 0:
                    print(f"   Collected {count} samples...", end="\r")
        except Exception:
            pass

    print(f"\n   Done. Kept {count} samples. Skipped {skipped}.")
    X = np.array(X)[..., np.newaxis]
    y = np.array(y)
    return X, y

def train_and_export():
    X, y = get_dataset()

    # --- MODEL DEFINITION ---
    # We explicitly name the layers to match your C++ expectations.
    model = tf.keras.models.Sequential([
        # Input Name: input_spectrogram
        tf.keras.layers.Input(shape=INPUT_SHAPE, name="input_spectrogram"),
        
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        
        # Output Name: dense_1 (Matches C++ Code)
        tf.keras.layers.Dense(len(TARGET_COMMANDS), activation='softmax', name="dense_1")
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print("\n3. Training Model...")
    model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.15)

    print("\n4. Exporting to ONNX...")
    
    # --- Keras 3 Patch ---
    # Manually tell tf2onnx that the output is "dense_1"
    if not hasattr(model, "output_names"):
        model.output_names = ["dense_1"]

    spec = (tf.TensorSpec((None, *INPUT_SHAPE), tf.float32, name="input_spectrogram"),)
    
    model_proto, _ = tf2onnx.convert.from_keras(
        model, 
        input_signature=spec, 
        opset=13
    )

    # --- Force IR Version 10 for C++ Safety ---
    output_path = "model_lab/model_quantized.onnx"
    model_onnx = onnx.load_model_from_string(model_proto.SerializeToString())
    model_onnx.ir_version = 10 
    
    del model_onnx.opset_import[:]
    opset = model_onnx.opset_import.add()
    opset.domain = ""
    opset.version = 12

    onnx.save(model_onnx, output_path)
    
    print(f"✅ SUCCESS: Model saved to {output_path}")
    print(f"ℹ️  Verified Output Node Name: {model_onnx.graph.output[0].name}")
    print(f"ℹ️  (This should match 'dense_1' for your C++ app)")

if __name__ == "__main__":
    train_and_export()