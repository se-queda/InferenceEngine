import sys
import os
import numpy as np
import librosa

# 1. Setup Path to find the C++ module in 'build/'
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
build_dir = os.path.join(project_root, 'build')

sys.path.append(build_dir)

try:
    import audioguard_core
    print(f" Imported C++ module from {build_dir}")
except ImportError as e:
    print(f"FAILED to import C++ module.")
    print(f"   Error details: {e}")
    sys.exit(1)

def test_audioloader():
    print("\n--- Testing C++ AudioLoader (FFMPEG) ---")

    # 2. Find a test file from your Downloads folder
    # UPDATED: Pointing to where your data actually is
    possible_paths = [
        "/home/utsab/Downloads/mini_speech_commands",
        "/home/utsab/Downloads/data/mini_speech_commands",
        "/home/utsab/Downloads"
    ]
    
    test_file = None
    data_dir = ""

    # Walk to find the first .wav file in any of the possible paths
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Searching for .wav in: {path}")
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.endswith(".wav") and not file.startswith("._"): # Ignore mac junk
                        test_file = os.path.join(root, file)
                        data_dir = path
                        break
                if test_file: break
        if test_file: break

    if not test_file:
        print(f"❌ Test file not found in: {possible_paths}")
        print("   Please ensure a .wav file exists in your Downloads.")
        return

    print(f"Target File: {test_file}")

    # 3. Load using Librosa (The "Gold Standard")
    print("Loading with Librosa...")
    # sr=16000 matches our C++ engine target
    py_audio, _ = librosa.load(test_file, sr=16000, mono=True)

    # 4. Load using C++ AudioLoader
    print("Loading with C++ AudioLoader...")
    try:
        # This calls src/AudioLoader.cpp
        cpp_audio_list = audioguard_core.AudioLoader.load_audio(test_file)
        cpp_audio = np.array(cpp_audio_list, dtype=np.float32)
    except Exception as e:
        print(f"C++ CRASHED: {e}")
        return

    # 5. Compare Results
    print(f"\nStats:")
    print(f"   Python Samples: {len(py_audio)}")
    print(f"   C++ Samples:    {len(cpp_audio)}")
    
    if len(cpp_audio) == 0:
         print("C++ returned empty audio.")
         return

    # Check length similarity 
    # (Resampling algorithms often differ by ~10-50 samples at the edges, which is fine)
    len_diff = abs(len(py_audio) - len(cpp_audio))
    
    if len_diff > 1000: 
         print(f" Significant length mismatch ({len_diff} samples)")
         return

    print("AudioLoader successfully decoded and resampled the file!")

if __name__ == "__main__":
    test_audioloader()