#ifndef AUDIOGUARD_PREPROCESSOR_H
#define AUDIOGUARD_PREPROCESSOR_H

#include <vector>
#include <cmath>
#include <string>

namespace audioguard {

constexpr int SAMPLE_RATE = 16000;
constexpr int N_FFT = 1024;
constexpr int HOP_LENGTH = 512;
constexpr int N_MELS = 40;
constexpr int EXPECTED_SAMPLES = 16000;

class Preprocessor {
public:
    Preprocessor(); // Now this constructor does heavy lifting
    ~Preprocessor();

    std::vector<float> process(const std::vector<float>& input_audio);

private:
    // Core DSP steps
    std::vector<float> pad_signal(const std::vector<float>& input);
    std::vector<std::vector<float>> compute_stft_magnitude(const std::vector<float>& signal);
    std::vector<std::vector<float>> apply_mel_filterbank(const std::vector<std::vector<float>>& stft_mag);
    void apply_log_scale(std::vector<std::vector<float>>& mel_energies);
    void apply_normalization(std::vector<std::vector<float>>& mel_energies);

    // Initialization (Runtime Math)
    void init_hamming_window();
    void init_mel_filters();

    // Helper math for Mel calculation
    float hz_to_mel(float hz);
    float mel_to_hz(float mel);

    // Data structures
    std::vector<float> window_func_;
    std::vector<std::vector<float>> mel_filters_;
};

} // namespace audioguard

#endif // AUDIOGUARD_PREPROCESSOR_H