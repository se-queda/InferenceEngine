# Performance Benchmarks: Inference Engine

This document provides a detailed breakdown of the latency and throughput performance for the Keyword Spotting system, comparing the **Embedded Edge** and **Cloud Hybrid** architectures.

## 💻 Hardware Environment
* **CPU:** AMD Ryzen 7 7840HS (8 Cores, 16 Threads)
* **GPU:** NVIDIA GeForce RTX 4060 Laptop (8GB VRAM)
* **OS:** Ubuntu 22.04 LTS


## 🚀 Large-Scale Batch Validation (8,000 Files)
The following metrics were captured during a continuous stress test of 8,000 audio samples using the **Hybrid C++ Preprocessor + Triton GPU** pipeline.



| Metric | Value |
| :--- | :--- |
| **Total Samples Processed** | 8,000 |
| **Final Accuracy** | **96.70%** |
| **Avg. DSP Time (C++ Core)** | 3.340 ms |
| **Avg. Inference Time (Triton GPU)** | 0.653 ms |
| **Total End-to-End Latency** | **3.993 ms** |
| **Throughput** | **250.5 req/sec** |

---

## 📊 Deployment Architecture Comparison

The system is optimized for two distinct use cases. The **Edge** mode prioritizes ultra-low latency for local triggers, while the **Hybrid** mode maximizes throughput via GPU acceleration.

| Metric | Embedded C++ (Edge/CPU) | Hybrid (Triton Cloud/GPU) |
| :--- | :--- | :--- |
| **DSP Latency** | 1.05 ms | 3.34 ms |
| **Inference Latency** | 1.55 ms | 0.65 ms |
| **Total E2E Latency** | **2.60 ms** | **3.99 ms** |
| **Max Throughput** | ~384 files/sec | **~250 files/sec** |
| **Execution Context** | Local C++ Memory | Network + Docker + CUDA |

## 🛠️ Methodology

### **1. Feature Extraction (C++ Core)**
Preprocessing is handled by a custom C++ implementation using **KissFFT**. This includes WAV decoding, resampling to 16kHz, and computing Log-Mel Spectrograms. By using native code, we avoid the heavy overhead associated with Python-based audio libraries.

### **2. Triton GPU Inference**
The model is served via **NVIDIA Triton Inference Server** in a Docker container. The Python client utilizes `PyBind11` to trigger the C++ preprocessor before shipping the resulting tensor via HTTP/REST.

### **3. Warm-up Protocol**
To ensure accurate "steady-state" measurements, a warm-up phase of 5 dummy inferences is executed before every benchmark. This ensures the CUDA context is initialized and the ONNX model is fully loaded into VRAM, eliminating "Cold Start" spikes (which can exceed 400ms on first run).

---
