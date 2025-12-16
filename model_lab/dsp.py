import tritonclient.http as httpclient
import numpy as np
import librosa
import sys

# ==========================================
# 🔧 CONFIGURATION
# ==========================================
SERVER_URL = "localhost:8000"
MODEL_NAME = "audioguard"
INPUT_NAME = "input_spectrogram"
OUTPUT_NAME = "dense_1"

# Standard Google Speech Commands labels
LABELS = ["down", "go", "left", "no", "off", "on", "right", "stop", "up", "yes"]

# ==========================================
# 🧠 YOUR EXACT DSP CLASS
# ==========================================
class DSP:
    def __init__(self, sample_rate=16000, n_fft=1024, hop_length=512, n_mels=40):
        self.sr = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

    def process(self, audio_data):
        """
        Input: 1D numpy array (audio samples)
        Output: 2D numpy array (Log Mel-Spectrogram)
        """
        # 1. Ensure length is exactly 1 second (16000 samples)
        if len(audio_data) > self.sr:
            audio_data = audio_data[:self.sr]
        elif len(audio_data) < self.sr:
            audio_data = np.pad(audio_data, (0, self.sr - len(audio_data)))

        # 2. STFT (Short-Term Fourier Transform)
        stft = np.abs(librosa.stft(
            audio_data, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            center=False
        ))**2

        # 3. Compute Mel Filterbank
        mel_basis = librosa.filters.mel(
            sr=self.sr, 
            n_fft=self.n_fft, 
            n_mels=self.n_mels, 
            htk=True,          
            norm='slaney'      
        )
        
        # 4. Apply Mel Basis
        mel_s = np.dot(mel_basis, stft)

        # 5. Log Scale (Log Mel-Spectrogram)
        log_mel = np.log10(mel_s + 1e-6)
        
        # 6. Normalize (Global Standardization)
        mean = log_mel.mean()
        std = log_mel.std()
        log_mel = (log_mel - mean) / (std + 1e-8)

        # Output shape: (n_mels, time_steps) -> Transpose to (time_steps, n_mels)
        return log_mel.T

# ==========================================
# 🚀 INFERENCE ROUTINE
# ==========================================
def run_inference(file_path):
    # 1. Connect to Triton
    try:
        client = httpclient.InferenceServerClient(url=SERVER_URL)
        if not client.is_server_live():
            print("❌ Server is offline.")
            return
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return

    print(f"📂 Loading audio: {file_path}")

    # 2. Load Raw Audio with Librosa (Just to get PCM data)
    try:
        # Load as 16kHz mono
        audio, _ = librosa.load(file_path, sr=16000, mono=True)
    except Exception as e:
        print(f"❌ Failed to load file: {e}")
        return

    # 3. Run DSP (The exact same logic as your C++)
    processor = DSP()
    spectrogram = processor.process(audio)

    # 4. Reshape for Triton
    # Current shape: (30, 40)
    # Target shape: (1, 30, 40, 1) -> (Batch, Time, Freq, Channels)
    input_tensor = spectrogram[np.newaxis, ..., np.newaxis].astype(np.float32)

    print(f"🧩 Input Shape: {input_tensor.shape}")

    # 5. Send Request
    inputs = [httpclient.InferInput(INPUT_NAME, input_tensor.shape, "FP32")]
    inputs[0].set_data_from_numpy(input_tensor)
    outputs = [httpclient.InferRequestedOutput(OUTPUT_NAME)]

    print("🚀 Sending to RTX 4060...")
    try:
        res = client.infer(model_name=MODEL_NAME, inputs=inputs, outputs=outputs)
        
        # 6. Decode Result
        logits = res.as_numpy(OUTPUT_NAME)[0] # Remove batch dim
        prediction_index = np.argmax(logits)
        
        # Simple Softmax for percentage
        probs = np.exp(logits) / np.sum(np.exp(logits))
        confidence = probs[prediction_index] * 100

        print("\n" + "="*30)
        print(f"🎤 Prediction: {LABELS[prediction_index].upper()}")
        print(f"📊 Confidence: {confidence:.2f}%")
        print("="*30)
        print(f"Full Logits: {logits}")

    except Exception as e:
        print(f"❌ Inference failed: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python client_real.py <path_to_wav_file>")
    else:
        run_inference(sys.argv[1])
