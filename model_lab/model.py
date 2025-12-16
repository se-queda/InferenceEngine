import os
import sys
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tf2onnx
import onnx
import librosa # Used only for helper functions if needed, mainly we use DSP

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

# The "Golden 10" (Must match your C++ App/main.cpp EXACTLY)
TARGET_COMMANDS = ["down", "go", "left", "no", "off", "on", "right", "stop", "up", "yes"]

def get_dataset():
    print(f"\n1. Loading 'speech_commands' via TensorFlow Datasets...")
    
    # Load the dataset (auto-downloads ~2.4GB if needed)
    dataset, info = tfds.load('speech_commands', with_info=True, as_supervised=True)
    
    # Get the label names from the dataset metadata
    all_label_names = info.features['label'].names
    print(f"   Original Dataset has {len(all_label_names)} labels.")
    
    # Create a mapping: TFDS_Index -> Our_Index
    # If a word is NOT in our target list, we map it to -1 (to skip it)
    label_map = {}
    for tfds_idx, name in enumerate(all_label_names):
        if name in TARGET_COMMANDS:
            label_map[tfds_idx] = TARGET_COMMANDS.index(name)
        else:
            label_map[tfds_idx] = -1

    dsp = DSP()
    X = []
    y = []
    
    # Iterate through Train, Validation, and Test splits
    print("2. Processing & Filtering Audio (Converting to Spectrograms)...")
    
    # We combine all splits because we want maximum data for our 10 words
    # (Since we are filtering out 20+ other words, we can afford to use everything)
    full_ds = dataset['train'].concatenate(dataset['validation']).concatenate(dataset['test'])
    
    # Convert to numpy iterator for processing
    # TFDS returns (audio, label). Audio is Int16.
    iterator = tfds.as_numpy(full_ds)
    
    count = 0
    skipped = 0
    
    for audio, label_idx in iterator:
        # 1. Check if this is a word we care about
        mapped_label = label_map[label_idx]
        if mapped_label == -1:
            skipped += 1
            continue
            
        # 2. INT16 -> FLOAT32 Conversion
        # TFDS gives raw PCM (e.g., 32000). Librosa/DSP expects -1.0 to 1.0
        audio = audio.astype(np.float32) / 32768.0
        
        # 3. Handle Length (Pad or Crop to 16000)
        if len(audio) < SAMPLE_RATE:
            audio = np.pad(audio, (0, SAMPLE_RATE - len(audio)), 'constant')
        else:
            audio = audio[:SAMPLE_RATE]
            
        # 4. DSP Processing
        try:
            spectrogram = dsp.process(audio)
            
            # Ensure shape is correct
            if spectrogram.shape == (30, 40):
                X.append(spectrogram)
                y.append(mapped_label)
                count += 1
                
                if count % 1000 == 0:
                    print(f"   Collected {count} samples...", end="\r")
                    
        except Exception:
            pass

    print(f"\n   Done. Kept {count} samples. Skipped {skipped} (irrelevant words).")
    
    # Add Channel Dimension
    X = np.array(X)[..., np.newaxis]
    y = np.array(y)
    
    print(f"   Final Data Shape: {X.shape}")
    return X, y

def train_and_export():
    X, y = get_dataset()

    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=INPUT_SHAPE, name="input_spectrogram"),
        
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        
        # Output: 10 Neurons (Matches C++ labels exactly)
        tf.keras.layers.Dense(len(TARGET_COMMANDS), activation='softmax', name="dense_output")
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print("\n3. Training Model...")
    # Using 15% for validation now since we merged everything earlier
    model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.15)

    print("\n4. Exporting to ONNX...")
    spec = (tf.TensorSpec((None, *INPUT_SHAPE), tf.float32, name="input_spectrogram"),)
    output_path = "model_lab/model_quantized.onnx"

    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)

    # --- Force IR Version 10 for C++ Safety ---
    model_onnx = onnx.load_model_from_string(model_proto.SerializeToString())
    model_onnx.ir_version = 10 
    
    del model_onnx.opset_import[:]
    opset = model_onnx.opset_import.add()
    opset.domain = ""
    opset.version = 12

    onnx.save(model_onnx, output_path)
    print(f"✅ SUCCESS: Model saved to {output_path}")
    print(f"ℹ️  Output Node Name: {model_onnx.graph.output[0].name}")

if __name__ == "__main__":
    train_and_export()