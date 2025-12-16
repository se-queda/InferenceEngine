import os
import glob
import numpy as np
import tritonclient.http as httpclient
import librosa
import time

# ==========================================
# 🔧 CONFIGURATION
# ==========================================
DATASET_PATH = "/home/utsab/Downloads/mini_speech_commands"
SERVER_URL = "localhost:8000"
MODEL_NAME = "audioguard"
INPUT_NAME = "input_spectrogram"
OUTPUT_NAME = "dense_1"

LABELS = ["down", "go", "left", "no", "off", "on", "right", "stop", "up", "yes"]

# ==========================================
# 🧠 DSP CLASS
# ==========================================
class DSP:
    def __init__(self, sample_rate=16000, n_fft=1024, hop_length=512, n_mels=40):
        self.sr = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

    def process(self, audio_data):
        if len(audio_data) > self.sr:
            audio_data = audio_data[:self.sr]
        elif len(audio_data) < self.sr:
            audio_data = np.pad(audio_data, (0, self.sr - len(audio_data)))

        stft = np.abs(librosa.stft(audio_data, n_fft=self.n_fft, hop_length=self.hop_length, center=False))**2
        mel_basis = librosa.filters.mel(sr=self.sr, n_fft=self.n_fft, n_mels=self.n_mels, htk=True, norm='slaney')
        mel_s = np.dot(mel_basis, stft)
        log_mel = np.log10(mel_s + 1e-6)
        
        mean = log_mel.mean()
        std = log_mel.std()
        log_mel = (log_mel - mean) / (std + 1e-8)

        return log_mel.T

# ==========================================
# 🚀 BATCH ENGINE
# ==========================================
def run_batch_test():
    try:
        client = httpclient.InferenceServerClient(url=SERVER_URL)
        if not client.is_server_live():
            print("❌ Server is offline.")
            return
    except:
        print("❌ Connection failed.")
        return

    dsp = DSP()
    total_files = 0
    total_correct = 0
    total_inference_time = 0.0  # <--- Accumulator for pure inference time
    
    print(f"{'LABEL':<10} | {'FILE':<20} | {'PRED':<10} | {'CONF':<6} | {'TIME (ms)':<9} | {'STATUS'}")
    print("-" * 90)

    for true_label in LABELS:
        folder_path = os.path.join(DATASET_PATH, true_label)
        
        if not os.path.isdir(folder_path):
            continue

        files = glob.glob(os.path.join(folder_path, "*.wav"))
        selected_files = files[:10] 

        label_correct = 0
        
        for file_path in selected_files:
            filename = os.path.basename(file_path)
            
            try:
                # 1. Preprocessing (Not Timed)
                audio, _ = librosa.load(file_path, sr=16000, mono=True)
                spectrogram = dsp.process(audio)
                input_tensor = spectrogram[np.newaxis, ..., np.newaxis].astype(np.float32)

                inputs = [httpclient.InferInput(INPUT_NAME, input_tensor.shape, "FP32")]
                inputs[0].set_data_from_numpy(input_tensor)
                outputs = [httpclient.InferRequestedOutput(OUTPUT_NAME)]
                
                # 2. INFERENCE (TIMED)
                # -----------------------------------------------
                t0 = time.time()
                res = client.infer(model_name=MODEL_NAME, inputs=inputs, outputs=outputs)
                t1 = time.time()
                # -----------------------------------------------

                inference_ms = (t1 - t0) * 1000
                total_inference_time += inference_ms

                # 3. Post-processing
                logits = res.as_numpy(OUTPUT_NAME)[0]
                pred_idx = np.argmax(logits)
                pred_label = LABELS[pred_idx]
                
                probs = np.exp(logits) / np.sum(np.exp(logits))
                confidence = probs[pred_idx] * 100

                is_correct = (pred_label == true_label)
                status = "✅" if is_correct else "❌"
                
                if is_correct:
                    label_correct += 1
                    total_correct += 1
                total_files += 1

                print(f"{true_label:<10} | {filename[:20]:<20} | {pred_label:<10} | {confidence:5.1f}% | {inference_ms:7.2f} ms | {status}")

            except Exception as e:
                print(f"Error processing {filename}: {e}")

        print("-" * 90)

    # Final Statistics
    avg_inference = total_inference_time / total_files if total_files > 0 else 0

    print("=" * 90)
    print(f"🎉 BATCH TEST COMPLETE")
    print(f"📂 Total Files:      {total_files}")
    print(f"🏆 Total Accuracy:   {(total_correct/total_files)*100:.2f}%")
    print("-" * 30)
    print(f"⚡ Total Inference:  {total_inference_time:.2f} ms")
    print(f"🚀 Avg Latency/Req:  {avg_inference:.2f} ms")
    print(f"🏎️  Throughput:       {(1000/avg_inference):.0f} req/sec")
    print("=" * 90)

if __name__ == "__main__":
    run_batch_test()