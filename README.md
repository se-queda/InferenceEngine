# AudioGuard: High-Performance Audio Inference Engine

**AudioGuard** is a production-grade Keyword Spotting (KWS) system featuring a hybrid architecture that bridges high-performance C++ digital signal processing (DSP) with scalable AI inference. 

The project demonstrates a "Test What You Fly" philosophy, using identical C++ DSP logic for both ultra-low-latency edge devices and high-throughput GPU cloud clusters via NVIDIA Triton.



## Architecture Overiew



### 1. The C++ Core (`audioguard_core`)
The engine's heart, written in C++17 for maximum efficiency and zero-copy data handling.
* **DSP:** Custom implementation using **KissFFT** for STFT and Log-Mel Spectrogram generation.
* **Loading:** Static WAV loader with 16kHz resampling and mono-mixing using ffmpeg
* **Bindings:** Exposed to Python via **PyBind11** to ensure feature parity between local development and cloud deployment.

### 2. Edge Inference Mode
A standalone C++ deployment using the **ONNX Runtime C++ API**. 
* Optimized with Level 3 Graph Optimizations.
* Designed for embedded systems and offline "always-on" trigger word detection.

### 3. Cloud Hybrid Mode (Triton)
An enterprise-scale deployment using **NVIDIA Triton Inference Server**.
* **Client:** Python-based, utilizing the C++ Core for accelerated preprocessing.
* **Server:** Dockerized environment running on GPU (CUDA), supporting dynamic batching and concurrent model execution.

---

## Project Structure

```text
.
├── App/
│   └── main.cpp                     # Edge Inference Sequential
├── bindings/
│   └── python_bindings.cpp          # PyBind11 bindings for C++ core
├── src/
│   ├── AudioLoader.cpp              # FFMPEG audioloader
│   ├── Preprocessor.cpp             # KissFFT + Mel-spectrogram pipeline
│   ├── InferenceEngine.cpp          # ONNX Runtime C++ wrapper
├── Testers                          # Utility functions used to test the system during various stages of development
├── include/
│   └── audioguard/
│       ├── AudioLoader.h
│       ├── Preprocessor.h
│       └── InferenceEngine.h
├── model_lab/
│   ├── dsp.py                       # Python DSP reference / dev version
│   ├── model.py                     # TF training + ONNX export script
│   └── model.onnx                   # exported graph optimised onxx model
├── model_repository/
│   └── audioguard/
│       ├── config.pbtxt             # Triton server-side config
│       └── 1/
│           └── model.onnx           # deployment model
├── clients/
│   ├── main.py                      # Hybrid C++ + Triton benchmark client
│   └── Trinton_stress_test.py       # Large-scale batch testing using C++ Client and Trinton based inference
├── BENCHMARK.md                     # Detailed performance results
├── README.md                        # Project overview & setup
├── .gitignore
└── CMakeLists.txt                   # Top-level build configuration
