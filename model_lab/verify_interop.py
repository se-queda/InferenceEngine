import sys
import os
import numpy as np
import time

# 1. Setup Path to find the C++ module in 'build/'
# This assumes you run the script from the project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
build_dir = os.path.join(project_root, 'build')

sys.path.append(build_dir)

try:
    import audioguard_core
    print(f"✅ SUCCESS: Imported C++ module from {build_dir}")
except ImportError as e:
    print(f"❌ FAILED to import C++ module.")
    print(f"   Make sure you built the project and 'audioguard_core.so' exists in:")
    print(f"   {build_dir}")
    print(f"   Error details: {e}")
    sys.exit(1)

# Import your Python DSP for comparison
try:
    from dsp import DSP
except ImportError:
    # Handle running from root vs model_lab folder
    sys.path.append(os.path.join(project_root, 'model_lab'))
    from dsp import DSP

def test_interoperability():
    print("\n--- Phase 3: Interoperability Verification ---")
    
    # 1. Generate Fake Audio (1 second of white noise)
    # Using a fixed seed ensures reproducibility
    np.random.seed(42)
    # Float32 audio between -1.0 and 1.0, exactly 16000 samples
    audio_data = np.random.uniform(-1.0, 1.0, 16000).astype(np.float32)
    
    # 2. Run Python "Gold Standard"
    print("Running Python DSP...")
    py_dsp = DSP()
    
    start_py = time.perf_counter()
    py_result = py_dsp.process(audio_data) 
    end_py = time.perf_counter()
    
    # 3. Run C++ Port
    print("Running C++ DSP...")
    cpp_dsp = audioguard_core.Preprocessor()
    
    # C++ expects a list or 1D array, returns a FLATTENED list
    start_cpp = time.perf_counter()
    cpp_result_flat = cpp_dsp.process(audio_data.tolist())
    end_cpp = time.perf_counter()
    
    # 4. Compare Results
    print(f"\nDimensions Check:")
    print(f"   Python Shape: {py_result.shape}")
    
    # DYNAMIC SHAPE HANDLING
    # We use the Python result to determine the expected shape.
    # Typically (30, 40) for 16k audio with these STFT settings.
    expected_rows = py_result.shape[0]
    expected_cols = py_result.shape[1]
    expected_total = expected_rows * expected_cols
    
    if len(cpp_result_flat) != expected_total:
        print(f"❌ SIZE MISMATCH:")
        print(f"   Python Produced: {expected_rows} x {expected_cols} = {expected_total} elements")
        print(f"   C++ Produced:    {len(cpp_result_flat)} elements")
        sys.exit(1)

    # Reshape C++ flat list to match Python matrix
    cpp_result = np.array(cpp_result_flat, dtype=np.float32).reshape(expected_rows, expected_cols)
    print(f"   C++ Shape:    {cpp_result.shape} (Matched)")
    
    # 5. Numerical Validation
    # We use a tolerance (atol=1e-3) to account for slight float math differences
    # between Python (BLAS/Numpy) and C++ (std::cos/libm).
    mse = ((py_result - cpp_result) ** 2).mean()
    max_diff = np.abs(py_result - cpp_result).max()
    is_match = np.allclose(py_result, cpp_result, rtol=1e-3, atol=1e-3)
    
    print(f"\nAccuracy Metrics:")
    print(f"   Mean Squared Error: {mse:.8f}")
    print(f"   Max Difference:     {max_diff:.8f}")
    
    print(f"\nPerformance (Single Run - Overhead included):")
    print(f"   Python Time: {end_py - start_py:.6f} s")
    print(f"   C++ Time:    {end_cpp - start_cpp:.6f} s")
    
    if is_match:
        print("\n✅ PASSED: C++ implementation matches Python Gold Standard!")
    else:
        print("\n❌ FAILED: Outputs diverge.")
        print("   If the Max Difference is small (< 0.05), it might be acceptable floating-point drift.")
        print("   If it is large, check normalization or STFT windowing logic.")

if __name__ == "__main__":
    test_interoperability()