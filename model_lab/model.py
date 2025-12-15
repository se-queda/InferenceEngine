import os
import pathlib
import numpy as np
import tensorflow as tf
import tf2onnx
import librosa
from dsp import DSP # <--- UPDATED IMPORT to match your new filename

# --- Config ---
import os
import pathlib
import numpy as np
import tensorflow as tf
import tf2onnx
import librosa
from dsp import DSP
from dsp import DSP

# --- Config for Mini Speech Commands ---
# If you downloaded from Kaggle, set this to your folder path.
# Otherwise, the script downloads it for you.
DATASET_PATH = '../data/mini_speech_commands'
TARGET_WORD = 'yes'  # The "Event" we are detecting
SAMPLE_RATE = 16000
EPOCHS = 15
BATCH_SIZE = 32


def load_and_preprocess_data():
    print("1. Setting up Mini Speech Commands dataset...")

    data_dir = pathlib.Path(DATASET_PATH)
    if not data_dir.exists():
        print("   Downloading dataset...")
        tf.keras.utils.get_file(
            'mini_speech_commands.zip',
            origin='http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip',
            extract=True,
            cache_dir='..', cache_subdir='data')

    # Verify classes
    commands = np.array([item.name for item in data_dir.glob('*') if item.is_dir() and item.name != "README.md"])
    print(f"   Available classes: {commands}")

    if TARGET_WORD not in commands:
        raise ValueError(f"Target '{TARGET_WORD}' not found in dataset.")

    dsp = DSP()
    X = []
    y = []

    print(f"2. Processing DSP (Target: '{TARGET_WORD}')...")

    # 1. Process Positive Class
    pos_path = data_dir / TARGET_WORD
    print(f"   Processing positive samples: {TARGET_WORD}")
    for file in os.listdir(pos_path):
        if not file.endswith('.wav'): continue
        audio, _ = librosa.load(os.path.join(pos_path, file), sr=SAMPLE_RATE)
        spectrogram = dsp.process(audio)
        X.append(spectrogram)
        y.append(1)  # 1 = "Yes"

    # 2. Process Negative Classes
    print("   Processing negative samples...")
    neg_count = 0
    target_neg_count = len(X)  # Balance 50/50

    for cmd in commands:
        if cmd == TARGET_WORD: continue
        neg_path = data_dir / cmd

        # Distribute negative samples across other classes
        files = os.listdir(neg_path)
        limit = target_neg_count // (len(commands) - 1)

        for file in files[:limit]:
            if not file.endswith('.wav'): continue
            audio, _ = librosa.load(os.path.join(neg_path, file), sr=SAMPLE_RATE)
            spectrogram = dsp.process(audio)
            X.append(spectrogram)
            y.append(0)  # 0 = Not "Yes"

    X = np.array(X)[..., np.newaxis]
    y = np.array(y)

    print(f"   Final Data shape: {X.shape} (Labels: {y.shape})")
    return X, y


def train_and_export():
    X, y = load_and_preprocess_data()

    input_shape = X.shape[1:]  # Should be (32, 40, 1) based on dsp.py settings

    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        # Conv Block 1
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        # Conv Block 2
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        # Classifier
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(2, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print("3. Training Model...")
    model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2)

    print("4. Exporting to ONNX...")
    spec = (tf.TensorSpec((None, *input_shape), tf.float32, name="input_spectrogram"),)
    output_path = "model_lab/model_quantized.onnx"

    # Export
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)

    with open(output_path, "wb") as f:
        f.write(model_proto.SerializeToString())

    print(f"SUCCESS: Model saved to {output_path}")


if __name__ == "__main__":
    train_and_export()