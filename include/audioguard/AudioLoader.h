#ifndef AUDIOGUARD_AUDIOLOADER_H
#define AUDIOGUARD_AUDIOLOADER_H

#include <vector>
#include <string>
#include <stdexcept>

namespace audioguard {

class AudioLoader {
public:
    /**
     * Loads an audio file (WAV, MP3, FLAC, etc.) using FFMPEG.
     * 1. Decodes the audio stream.
     * 2. Downmixes to Mono.
     * 3. Resamples to 16000 Hz.
     * * @param filepath Path to the audio file.
     * @return std::vector<float> Raw audio samples (normalized float).
     * @throws std::runtime_error If file cannot be opened or decoded.
     */
    static std::vector<float> load_audio(const std::string& filepath);
};

} // namespace audioguard

#endif // AUDIOGUARD_AUDIOLOADER_H