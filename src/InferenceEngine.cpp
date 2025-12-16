#include "audioguard/InferenceEngine.h"
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>

namespace audioguard {

// Pimpl pattern to hide ONNX Runtime details from the header file
struct InferenceEngine::Impl {
    Ort::Env env;
    Ort::Session session;
    Ort::AllocatorWithDefaultOptions allocator;
    
    // Model metadata
    std::vector<const char*> input_node_names = {"input_spectrogram"};
    std::vector<const char*> output_node_names = {"dense_1"};
    std::vector<int64_t> input_dims;

    Impl(const std::string& model_path) 
        : env(ORT_LOGGING_LEVEL_WARNING, "AudioGuard"), 
          session(env, model_path.c_str(), Ort::SessionOptions()) {
        
        // Auto-detect input shape from the model
        // Note: For simplicity in this specific project, we assume [1, 16000] 
        // or [1, 32, 128] depending on if the model expects raw audio or spectrograms.
        // We will pass that responsibility to the Preprocessor.
    }
};

InferenceEngine::InferenceEngine(const std::string& model_path)
    : pImpl(std::make_unique<Impl>(model_path)) {}

InferenceEngine::~InferenceEngine() = default;

std::vector<float> InferenceEngine::predict(const std::vector<float>& input_data, 
                                            const std::vector<int64_t>& input_shape) {
    
    // 1. Prepare Memory Info
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);

    // 2. Create Input Tensor
    // We must cast away constness because ONNX Runtime API requires non-const pointer, 
    // even though it doesn't modify input.
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, 
        const_cast<float*>(input_data.data()), input_data.size(), 
        input_shape.data(), input_shape.size());

    // 3. Run Inference
    auto output_tensors = pImpl->session.Run(
        Ort::RunOptions{nullptr}, 
        pImpl->input_node_names.data(), 
        &input_tensor, 1, 
        pImpl->output_node_names.data(), 1);

    // 4. Extract Output
    float* floatarr = output_tensors.front().GetTensorMutableData<float>();
    size_t count = output_tensors.front().GetTensorTypeAndShapeInfo().GetElementCount();

    return std::vector<float>(floatarr, floatarr + count);
}

} // namespace audioguard