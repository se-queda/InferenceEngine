Cpp-Audio-Inference-EngineCpp-Audio-Inference-Engine is a production-grade, low-latency keyword spotting system written in Modern C++. It leverages ONNX Runtime with hardware-aware graph optimizations to perform real-time audio classification in under 3ms on standard CPU hardware.Additionally, the project includes a microservice deployment configuration using NVIDIA Triton Inference Server for scalable, GPU-accelerated cloud inference.Key Performance MetricsMetricUnoptimized (Python/Default)C++ Engine (Optimized)SpeedupInference Latency~23.0 ms2.6 ms~8.8xModel Size15 MB (Float32)3.8 MB (Int8 Quantized)4xThroughputSequentialDynamic Batching (GPU)ScalableBenchmarks run on: Intel Core i7 / NVIDIA RTX 4060 Laptop GPU.ArchitectureThe system is designed with a modular "Pimpl" (Pointer to Implementation) pattern to decouple the ONNX Runtime headers from the application logic, ensuring fast compilation and clean APIs.Code snippetgraph LR
    A[Microphone / WAV] -->|Raw Audio| B(AudioLoader)
    B -->|PCM Data| C(Preprocessor / DSP)
    C -->|STFT & Spectrogram| D{Inference Engine}
    D -->|ONNX Runtime API| E[optimized_model.onnx]
    E -->|Logits| F[ArgMax & Confidence]
    F -->|Prediction| G[Terminal / App]
Core ComponentsAudioLoader: Handles WAV file I/O and sample rate conversion.DSP Preprocessor: Custom C++ implementation of Short-Time Fourier Transform (STFT) using KissFFT to generate Log-Mel Spectrograms.Inference Engine: A wrapper around ONNX Runtime C++ API that manages session state, memory allocation, and graph optimization strategies.Tech StackLanguage: C++17Inference Backend: Microsoft ONNX Runtime (CPU) & NVIDIA Triton (GPU)DSP Library: KissFFTBuild System: CMakeDeployment: Docker, NVIDIA Container ToolkitOptimization StrategyTo achieve 2.6ms latency, several engineering decisions were made:Graph Optimization Level 3 (ORT_ENABLE_ALL): Enables constant folding, redundant node elimination, and layer fusion (e.g., Conv+Bias+ReLU).Thread Tuning: Forced SetIntraOpNumThreads(1) to eliminate thread-pool overhead for lightweight models.Memory Arena: Pre-allocated memory for tensors to prevent heap fragmentation during runtime.Hardware Awareness: Utilized AVX/SSE instruction sets automatically via the ONNX native provider.How to Build & Run (Embedded Mode)PrerequisitesCMake 3.10+G++ / ClangONNX Runtime C++ LibraryFFmpeg (for audio decoding)Build StepsBashmkdir build && cd build
cmake ..
make
UsageBash./AudioInferenceApp <path_to_model.onnx> <path_to_audio.wav>
Output:Plaintext[Init] Loading Model... Ready.
[1/3] Loading Audio... Done. (16000 samples)
[2/3] Preprocessing... Done. (1200 features)
[3/3] Running Inference... Done.

>>> PREDICTION: left (Index 2)
Latency (Load+DSP+Infer): 2.6247 ms
How to Deploy (Enterprise Mode)For cloud-scale scenarios, the model is deployed on NVIDIA Triton Inference Server with GPU acceleration.1. Setup NVIDIA DockerEnsure nvidia-container-toolkit is installed and the model repository is structured correctly.2. Launch ServerBashdocker run --gpus all --rm -p 8000:8000 \
  -v $(pwd)/model_repository:/models \
  nvcr.io/nvidia/tritonserver:23.12-py3 \
  tritonserver --model-repository=/models
3. Run ClientA Python client communicates with the server via REST API:Bashpython client_test.py
Configuration: config.pbtxt enables KIND_GPU and Dynamic Batching.Result: High-throughput inference capable of handling concurrent requests.Project StructurePlaintextCppAudioInference/
├── src/
│   ├── main.cpp              # Entry point
│   ├── InferenceEngine.cpp   # ONNX Runtime implementation
│   ├── Preprocessor.cpp      # DSP logic
│   └── AudioLoader.cpp       # File I/O
├── include/
│   └── inference_engine/     # Public headers
├── model_repository/         # Triton Server config
│   └── audio_classifier/
│       ├── config.pbtxt      # GPU configuration
│       └── 1/
│           └── model.onnx    # The model file
├── models/
│   └── model_quantized.onnx  # Optimized Edge model
├── CMakeLists.txt
└── README.md
LicenseMIT License. Free for use and modification