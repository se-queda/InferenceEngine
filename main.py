import tritonclient.http as httpclient
import numpy as np
import time

def run_inference():
    # ---------------------------------------------------------
    # 1. CONFIGURATION
    # ---------------------------------------------------------
    SERVER_URL = "localhost:8000"
    MODEL_NAME = "audioguard"
    INPUT_NAME = "input_spectrogram"
    OUTPUT_NAME = "dense_1"
    
    # Input Shape: [Batch, Time, Freq, Channels]
    INPUT_SHAPE = (1, 30, 40, 1) 
    NUM_REQUESTS = 100

    print(f"🔌 Connecting to Triton Server at {SERVER_URL}...")

    # ---------------------------------------------------------
    # 2. CONNECT TO SERVER
    # ---------------------------------------------------------
    try:
        client = httpclient.InferenceServerClient(url=SERVER_URL)
        if not client.is_server_live():
            print("❌ Server is NOT live. Is Docker running?")
            return
        print("✅ Server is online and ready!")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return

    # ---------------------------------------------------------
    # 3. PREPARE DATA
    # ---------------------------------------------------------
    print(f"🎲 Generating dummy input with shape {INPUT_SHAPE}...")
    dummy_input = np.random.randn(*INPUT_SHAPE).astype(np.float32)

    inputs = [httpclient.InferInput(INPUT_NAME, dummy_input.shape, "FP32")]
    inputs[0].set_data_from_numpy(dummy_input)
    outputs = [httpclient.InferRequestedOutput(OUTPUT_NAME)]

    # ---------------------------------------------------------
    # 4. WARM UP (Crucial for GPU)
    # ---------------------------------------------------------
    print("\n🔥 Warming up GPU (Ignoring first request latency)...")
    try:
        # Run once to wake up CUDA/cuDNN
        client.infer(model_name=MODEL_NAME, inputs=inputs, outputs=outputs)
    except Exception as e:
        print(f"❌ Warm-up failed: {e}")
        return

    # ---------------------------------------------------------
    # 5. SPEED RUN (100 Requests)
    # ---------------------------------------------------------
    print(f"🏎️  Running {NUM_REQUESTS} requests on RTX 4060...")
    
    start_time = time.time()
    
    try:
        for i in range(NUM_REQUESTS):
            # We overwrite 'response' every time, we only care about the last one for checking
            response = client.infer(model_name=MODEL_NAME, inputs=inputs, outputs=outputs)
            
        end_time = time.time()
        
        # ---------------------------------------------------------
        # 6. PROCESS RESULTS
        # ---------------------------------------------------------
        total_time = (end_time - start_time) * 1000 # in ms
        avg_latency = total_time / NUM_REQUESTS
        
        # Get the logits from the very last request just to verify correctness
        last_logits = response.as_numpy(OUTPUT_NAME)

        print("\n" + "="*40)
        print("🎉 STRESS TEST COMPLETE!")
        print("="*40)
        print(f"📊 Total Time:      {total_time:.2f} ms")
        print(f"🚀 Average Latency: {avg_latency:.2f} ms / request")
        print(f"⚡ Throughput:      {1000 / avg_latency:.1f} inferences / sec")
        print("-" * 40)
        print(f"Last Output Shape: {last_logits.shape}")
        print("="*40)

    except Exception as e:
        print(f"\n❌ Inference Loop Failed: {e}")

if __name__ == "__main__":
    run_inference()