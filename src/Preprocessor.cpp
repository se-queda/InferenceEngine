#include "audioguard/Preprocessor.h"
#include "kiss_fft.h"
#include <algorithm>
#include <numeric>
#include <iostream>
#include <cmath>
#include <complex> // <--- ADDED THIS FIXED THE ERROR

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace audioguard {

Preprocessor::Preprocessor() {
    // RUNTIME INITIALIZATION
    // These run once when the device boots/app starts.
    init_hamming_window();
    init_mel_filters();
}

Preprocessor::~Preprocessor() {}

// --- Helper Math ---
float Preprocessor::hz_to_mel(float hz) {
    // Slaney/HTK formula approximation used by Librosa
    return 2595.0f * std::log10(1.0f + hz / 700.0f);
}

float Preprocessor::mel_to_hz(float mel) {
    return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f);
}

// --- Init Logic ---
void Preprocessor::init_hamming_window() {
    window_func_.resize(N_FFT);
    // Periodic Hann Window (matches librosa/scipy 'fftbins=True')
    // Formula: 0.5 * (1 - cos(2*pi*n / N))
    for (int i = 0; i < N_FFT; ++i) {
        window_func_[i] = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / N_FFT));
    }
}

void Preprocessor::init_mel_filters() {
    int fft_size_bins = N_FFT / 2 + 1;
    mel_filters_.assign(N_MELS, std::vector<float>(fft_size_bins, 0.0f));

    // 1. Calculate exact Hz of every FFT bin
    // Librosa: fft_freqs = [0, ..., sr/2]
    std::vector<float> fft_freqs(fft_size_bins);
    for (int i = 0; i < fft_size_bins; ++i) {
        fft_freqs[i] = (float)i * SAMPLE_RATE / N_FFT;
    }

    // 2. Calculate Mel Points (Triangle Peaks) in Hz
    float mel_min = hz_to_mel(0.0f);
    float mel_max = hz_to_mel(SAMPLE_RATE / 2.0f);
    
    // We need N_MELS filters, so we need N_MELS + 2 points
    std::vector<float> mel_points_hz(N_MELS + 2);
    float step = (mel_max - mel_min) / (N_MELS + 1);
    
    for (int i = 0; i < N_MELS + 2; ++i) {
        mel_points_hz[i] = mel_to_hz(mel_min + i * step);
    }

    // 3. Construct Filters in Frequency Domain (Matches Librosa)
    for (int m = 0; m < N_MELS; ++m) {
        float f_left = mel_points_hz[m];
        float f_center = mel_points_hz[m+1];
        float f_right = mel_points_hz[m+2];

        // Slaney Area Normalization: 2.0 / (f_right - f_left)
        // This ensures the energy is consistent across bands
        float width = f_right - f_left;
        float norm_factor = (width > 0) ? 2.0f / width : 0.0f;

        for (int i = 0; i < fft_size_bins; ++i) {
            float freq = fft_freqs[i];
            float weight = 0.0f;

            if (freq > f_left && freq < f_center) {
                // Rising edge
                weight = (freq - f_left) / (f_center - f_left);
            } else if (freq >= f_center && freq < f_right) {
                // Falling edge
                weight = (f_right - freq) / (f_right - f_center);
            }

            // Apply Slaney Norm
            mel_filters_[m][i] = weight * norm_factor;
        }
    }
}

// --- Main Process Pipeline ---
std::vector<float> Preprocessor::process(const std::vector<float>& input_audio) {
    // 1. Pad
    std::vector<float> padded = pad_signal(input_audio);
    // 2. STFT
    auto stft_mag = compute_stft_magnitude(padded);
    // 3. Mel
    auto mel_energies = apply_mel_filterbank(stft_mag);
    // 4. Log
    apply_log_scale(mel_energies);
    // 5. Norm
    apply_normalization(mel_energies);

    // 6. Flatten
    std::vector<float> flattened;
    flattened.reserve(mel_energies.size() * mel_energies[0].size());
    for (const auto& row : mel_energies) {
        flattened.insert(flattened.end(), row.begin(), row.end());
    }
    return flattened;
}

std::vector<float> Preprocessor::pad_signal(const std::vector<float>& input) {
    std::vector<float> out = input;
    if (out.size() > EXPECTED_SAMPLES) out.resize(EXPECTED_SAMPLES);
    else if (out.size() < EXPECTED_SAMPLES) out.resize(EXPECTED_SAMPLES, 0.0f);
    return out;
}

std::vector<std::vector<float>> Preprocessor::compute_stft_magnitude(const std::vector<float>& signal) {
    kiss_fft_cfg cfg = kiss_fft_alloc(N_FFT, 0, nullptr, nullptr);
    
    // Complex vectors for input/output
    std::vector<std::complex<float>> fft_in(N_FFT);
    std::vector<std::complex<float>> fft_out(N_FFT);
    
    std::vector<std::vector<float>> spectrogram;

    for (size_t i = 0; i + N_FFT <= signal.size(); i += HOP_LENGTH) {
        for (int j = 0; j < N_FFT; ++j) {
            fft_in[j] = signal[i + j] * window_func_[j];
        }
        
        // Cast std::complex<float>* to kiss_fft_cpx* (they are binary compatible)
        kiss_fft(cfg, (kiss_fft_cpx*)fft_in.data(), (kiss_fft_cpx*)fft_out.data());
        
        std::vector<float> mag_frame;
        for (int j = 0; j < N_FFT / 2 + 1; ++j) {
            float re = fft_out[j].real();
            float im = fft_out[j].imag();
            mag_frame.push_back(re * re + im * im);
        }
        spectrogram.push_back(mag_frame);
    }
    kiss_fft_free(cfg);
    return spectrogram;
}

std::vector<std::vector<float>> Preprocessor::apply_mel_filterbank(const std::vector<std::vector<float>>& stft_mag) {
    int time_steps = stft_mag.size();
    int freq_bins = stft_mag[0].size();
    std::vector<std::vector<float>> mel_spectrogram(time_steps, std::vector<float>(N_MELS, 0.0f));

    for (int t = 0; t < time_steps; ++t) {
        for (int m = 0; m < N_MELS; ++m) {
            float sum = 0.0f;
            for (int f = 0; f < freq_bins; ++f) {
                sum += mel_filters_[m][f] * stft_mag[t][f];
            }
            mel_spectrogram[t][m] = sum;
        }
    }
    return mel_spectrogram;
}

void Preprocessor::apply_log_scale(std::vector<std::vector<float>>& mel_energies) {
    for (auto& row : mel_energies) {
        for (float& val : row) {
            val = std::log10(val + 1e-6f);
        }
    }
}

void Preprocessor::apply_normalization(std::vector<std::vector<float>>& mel_energies) {
    double sum = 0.0;
    size_t count = 0;
    for (const auto& row : mel_energies) {
        for (float val : row) { sum += val; count++; }
    }
    float mean = static_cast<float>(sum / count);

    double sq_sum = 0.0;
    for (const auto& row : mel_energies) {
        for (float val : row) {
            float diff = val - mean;
            sq_sum += diff * diff;
        }
    }
    float std = std::sqrt(static_cast<float>(sq_sum / count));
    if (std < 1e-8f) std = 1e-8f;

    for (auto& row : mel_energies) {
        for (float& val : row) {
            val = (val - mean) / std;
        }
    }
}

} // namespace audioguard