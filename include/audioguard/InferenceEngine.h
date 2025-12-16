#ifndef AUDIOGUARD_INFERENCEENGINE_H
#define AUDIOGUARD_INFERENCEENGINE_H

#include <vector>
#include <string>
#include <memory> // For std::unique_ptr

namespace audioguard {

class InferenceEngine {
public:
    // Constructor loads the model from disk
    explicit InferenceEngine(const std::string& model_path);
    
    // Destructor must be defined in .cpp where Impl is complete
    ~InferenceEngine();

    /**
     * Runs the ONNX model on the provided input data.
     * * @param input_data Flattened vector of input features (e.g., spectrogram).
     * @param input_shape Dimensions of the input (e.g., {1, 32, 128}).
     * @return std::vector<float> Raw probability scores (logits) for each class.
     */
    std::vector<float> predict(const std::vector<float>& input_data, 
                               const std::vector<int64_t>& input_shape);

private:
    // Pimpl Pattern: Hides ONNX headers from the public API
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace audioguard

#endif // AUDIOGUARD_INFERENCEENGINE_H