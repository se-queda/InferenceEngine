import os
import glob
import sys
import time
import numpy as np
import tritonclient.http as httpclient

# ---------------------------------------------------------
# 1. IMPORT C++ CORE MODULE
# ---------------------------------------------------------
sys.path.append(os.getcwd())

try:
    import audioguard_core
    print("✅ C++ Core Module (audioguard_core) Loaded!")
except ImportError as e:
    print(f"❌ Failed to load audioguard_core: {e}")
    print("Ensure audioguard_core.so is in the current directory.")
    exit(1)

# ---------------------------------------------------------
# 2. CONFIGURATION
# ---------------------------------------------------------
DATASET_PATH = "/home/utsab/Downloads/mini_speech_commands"
SERVER_URL = "localhost:8000"
MODEL_NAME = "audioguard"
INPUT_NAME = "input_spectrogram"
OUTPUT_NAME = "dense_1"

LABELS = ["down", "go", "left", "no", "off", "on", "right", "stop", "up", "yes"]

def run_ultimate_benchmark():
    # -----------------------------------------------------
    # SETUP
    # -----------------------------------------------------
    try:
        client = httpclient.InferenceServerClient(url=SERVER_URL)
        if not client.is_server_live():
            print("❌ Triton Server is offline.")
            return
        print(f"✅ Connected to Triton at {SERVER_URL}")
    except Exception as e:
        print(f"❌ Connection Failed: {e}")
        return

    try:
        preprocessor = audioguard_core.Preprocessor()
    except Exception as e:
        print(f"❌ Failed to init C++ classes: {e}")
        return

    # -----------------------------------------------------
    # 🔥 WARMUP PHASE (Clear Cold Start Spikes)
    # -----------------------------------------------------
    print("🔥 Warming up GPU context...")
    # Create a dummy 1x30x40x1 tensor
    warmup_data = np.zeros((1, 30, 40, 1), dtype=np.float32)
    warmup_input = [httpclient.InferInput(INPUT_NAME, warmup_data.shape, "FP32")]
    warmup_input[0].set_data_from_numpy(warmup_data)
    warmup_output = [httpclient.InferRequestedOutput(OUTPUT_NAME)]
    
    for _ in range(5):
        client.infer(model_name=MODEL_NAME, inputs=warmup_input, outputs=warmup_output)
    print("✅ Warmup complete. Context is hot.\n")

    # Stats Accumulators
    total_files = 0
    total_correct = 0
    accum_dsp_time = 0.0
    accum_inf_time = 0.0

    print("="*105)
    print(f"{'LABEL':<8} | {'FILE':<20} | {'PRED':<8} | {'CONF':<6} | {'DSP(ms)':<8} | {'INF(ms)':<8} | {'TOTAL':<8} | {'STAT'}")
    print("-" * 105)

    # -----------------------------------------------------
    # BATCH LOOP
    # -----------------------------------------------------
    for true_label in LABELS:
        folder_path = os.path.join(DATASET_PATH, true_label)
        if not os.path.isdir(folder_path):
            continue

        files = glob.glob(os.path.join(folder_path, "*.wav"))[:1000]

        for file_path in files:
            filename = os.path.basename(file_path)

            try:
                # --- PHASE 1: C++ LOADING & DSP ---
                t0 = time.time()
                raw_audio = audioguard_core.AudioLoader.load_audio(file_path)
                flat_features = preprocessor.process(raw_audio)
                t1 = time.time()
                
                dsp_time = (t1 - t0) * 1000
                accum_dsp_time += dsp_time

                # --- PHASE 2: RESHAPE ---
                features_np = np.array(flat_features, dtype=np.float32)
                input_tensor = features_np.reshape(1, 30, 40, 1)

                # --- PHASE 3: TRITON INFERENCE ---
                inputs = [httpclient.InferInput(INPUT_NAME, input_tensor.shape, "FP32")]
                inputs[0].set_data_from_numpy(input_tensor)
                outputs = [httpclient.InferRequestedOutput(OUTPUT_NAME)]

                t2 = time.time()
                res = client.infer(model_name=MODEL_NAME, inputs=inputs, outputs=outputs)
                t3 = time.time()
                
                inf_time = (t3 - t2) * 1000
                accum_inf_time += inf_time

                # --- DECODE RESULTS ---
                logits = res.as_numpy(OUTPUT_NAME)[0]
                pred_idx = np.argmax(logits)
                pred_label = LABELS[pred_idx]
                
                probs = np.exp(logits) / np.sum(np.exp(logits))
                confidence = probs[pred_idx] * 100
                is_correct = (pred_label == true_label)
                status = "✅" if is_correct else "❌"

                if is_correct: total_correct += 1
                total_files += 1
                
                total_req_time = dsp_time + inf_time

                print(f"{true_label:<8} | {filename[:20]:<20} | {pred_label:<8} | {confidence:5.1f}% | {dsp_time:8.3f} | {inf_time:8.3f} | {total_req_time:8.3f} | {status}")

            except Exception as e:
                print(f"\nError processing {filename}: {e}")

    # -----------------------------------------------------
    # FINAL REPORT
    # -----------------------------------------------------
    if total_files == 0:
        print("No files processed.")
        return

    avg_dsp = accum_dsp_time / total_files
    avg_inf = accum_inf_time / total_files
    avg_total = avg_dsp + avg_inf
    accuracy = (total_correct / total_files) * 100

    print("=" * 105)
    print(f"🎉 ULTIMATE BENCHMARK COMPLETE")
    print(f"📂 Total Files:      {total_files}")
    print(f"🏆 Final Accuracy:   {accuracy:.2f}%")
    print("-" * 45)
    print(f"⚡ Avg DSP Time (C++):       {avg_dsp:.3f} ms")
    print(f"☁️  Avg Inference (Net+GPU): {avg_inf:.3f} ms")
    print(f"⏱️  AVG TOTAL LATENCY:       {avg_total:.3f} ms")
    print(f"🏎️  End-to-End Throughput:   {1000/avg_total:.1f} files/sec")
    print("=" * 105)

if __name__ == "__main__":
    run_ultimate_benchmark()