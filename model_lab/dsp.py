import numpy as np
import librosa

class DSP:
    def __init__(self, sample_rate=16000, n_fft=1024, hop_length=512, n_mels=40):
        self.sr = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

    def process(self, audio_data):
        """
        Input: 1D numpy array (audio samples)
        Output: 2D numpy array (Log Mel-Spectrogram)
        """
        # 1. Ensure length is exactly 1 second (16000 samples)
        # This matches the C++ pad_signal() function
        if len(audio_data) > self.sr:
            audio_data = audio_data[:self.sr]
        elif len(audio_data) < self.sr:
            audio_data = np.pad(audio_data, (0, self.sr - len(audio_data)))

        # 2. STFT (Short-Term Fourier Transform)
        # center=False is CRITICAL to match C++ sliding window logic
        stft = np.abs(librosa.stft(
            audio_data, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            center=False
        ))**2

        # 3. Compute Mel Filterbank
        # UPDATED: Added htk=True to match C++ implementation
        mel_basis = librosa.filters.mel(
            sr=self.sr, 
            n_fft=self.n_fft, 
            n_mels=self.n_mels, 
            htk=True,          # <--- FIX: Use HTK formula (2595 * log10)
            norm='slaney'      # Keep Slaney normalization to handle area scaling
        )
        
        # 4. Apply Mel Basis
        mel_s = np.dot(mel_basis, stft)

        # 5. Log Scale (Log Mel-Spectrogram)
        # Adding a small epsilon for numerical stability
        log_mel = np.log10(mel_s + 1e-6)
        
        # 6. Normalize (Global Standardization)
        # This matches the apply_normalization() in C++
        mean = log_mel.mean()
        std = log_mel.std()
        log_mel = (log_mel - mean) / (std + 1e-8)

        # Output shape: (n_mels, time_steps) -> Transpose to (time_steps, n_mels)
        return log_mel.T

# Quick test to ensure it runs
if __name__ == "__main__":
    dummy_audio = np.random.uniform(-1, 1, 16000) 
    dsp = DSP()
    result = dsp.process(dummy_audio)
    print(f"DSP Output Shape: {result.shape}")
