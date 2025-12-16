#include <iostream>
#include <vector>
#include <string>
#include <numeric>   // for std::accumulate
#include <algorithm> // for std::max_element
#include <iomanip>   // for std::fixed, std::setprecision

#include "audioguard/AudioLoader.h"
#include "audioguard/Preprocessor.h"
#include "audioguard/InferenceEngine.h"

// Standard Mini Speech Commands Classes (Alphabetical Order)
const std::vector<std::string> LABELS = {
    "down", "go", "left", "no", "off", 
    "on", "right", "stop", "up", "yes"
};

// Helper to find the index of the highest score (ArgMax)
int argmax(const std::vector<float>& scores) {
    return std::distance(scores.begin(), std::max_element(scores.begin(), scores.end()));
}

int main(int argc, char* argv[]) {
    // 1. Argument Check
    if (argc < 3) {
        std::cerr << "Usage: ./AudioGuardApp <path_to_model.onnx> <path_to_audio.wav>\n";
        return 1;
    }

    std::string model_path = argv[1];
    std::string audio_path = argv[2];

    std::cout << "==========================================\n";
    std::cout << "   AudioGuard C++ Inference Engine v1.0   \n";
    std::cout << "==========================================\n";
    std::cout << "Model: " << model_path << "\n";
    std::cout << "Input: " << audio_path << "\n\n";

    try {
        // ---------------------------------------------------------
        // Step 1: The Ear (Load Audio via FFMPEG)
        // ---------------------------------------------------------
        std::cout << "[1/3] Loading Audio... ";
        auto raw_audio = audioguard::AudioLoader::load_audio(audio_path);
        std::cout << "Done. (" << raw_audio.size() << " samples)\n";

        // ---------------------------------------------------------
        // Step 2: The Cortex (Preprocess via KissFFT)
        // ---------------------------------------------------------
        std::cout << "[2/3] Preprocessing... ";
        audioguard::Preprocessor dsp;
        auto features = dsp.process(raw_audio);
        
        // CRITICAL SAFETY CHECK
        // Model expects [1, 30, 40, 1] = 1200 floats.
        // If we don't have exactly this, ONNX Runtime will crash segfault.
        if (features.size() != 1200) {
            throw std::runtime_error("Feature mismatch! Preprocessor produced " + 
                                     std::to_string(features.size()) + 
                                     " features, but model expects 1200 (30x40).");
        }
        
        // Define the shape your model expects (From prepare_model.py)
        // [Batch, Time, Freq, Channels]
        std::vector<int64_t> input_shape = {1, 30, 40, 1}; 
        
        std::cout << "Done. (1200 features generated)\n";

        // ---------------------------------------------------------
        // Step 3: The Brain (Inference via ONNX Runtime)
        // ---------------------------------------------------------
        std::cout << "[3/3] Running Inference... ";
        audioguard::InferenceEngine engine(model_path);
        
        auto logits = engine.predict(features, input_shape);
        std::cout << "Done.\n";

        // ---------------------------------------------------------
        // Step 4: Interpret Results
        // ---------------------------------------------------------
        int predicted_idx = argmax(logits);
        
        // Softmax calculation for display (Optional, but looks professional)
        float max_logit = *std::max_element(logits.begin(), logits.end());
        float sum_exp = 0.0f;
        for (float l : logits) sum_exp += std::exp(l - max_logit);
        
        std::cout << "\n------------------------------------------\n";
        if (predicted_idx >= 0 && predicted_idx < LABELS.size()) {
            std::cout << ">>> PREDICTION: " << LABELS[predicted_idx] << " (Index " << predicted_idx << ")\n";
        } else {
            std::cout << ">>> PREDICTION: Index " << predicted_idx << " (Unknown Label)\n";
        }
        std::cout << "------------------------------------------\n";

        std::cout << "Confidence Scores:\n";
        std::cout << std::fixed << std::setprecision(4);
        for (size_t i = 0; i < logits.size(); ++i) {
            float prob = std::exp(logits[i] - max_logit) / sum_exp;
            std::string label = (i < LABELS.size()) ? LABELS[i] : "Unknown";
            std::cout << "  " << label << ": " << prob * 100.0f << "%\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "\n❌ FATAL ERROR: " << e.what() << "\n";
        return 1;
    }

    return 0;
}