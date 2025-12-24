import sys
import os
import numpy as np
import onnx
from onnx import helper, TensorProto

# 1. Setup Path to C++ Module
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
build_dir = os.path.join(project_root, 'build')
sys.path.append(build_dir)

try:
    import audioguard_core
    print(f" SUCCESS: Imported C++ Module")
except ImportError as e:
    print(f" FAILED to import module: {e}")
    sys.exit(1)

def create_dummy_model(path):
    # Define input/output (Shape: [1, 10])
    input_info = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 10])
    output_info = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 10])

    # Create a node that does Output = Input + Input
    node = helper.make_node('Add', ['input', 'input'], ['output'])

    # Create Graph
    graph = helper.make_graph([node], 'test_graph', [input_info], [output_info])
    
    # Force IR Version 10 
    model = helper.make_model(graph, producer_name='audioguard_test', ir_version=10)
    
    # Explicitly lowering the Opset version to ensure compatibility
    del model.opset_import[:]
    opset = model.opset_import.add()
    opset.domain = ""
    opset.version = 12 

    onnx.save(model, path)
    print(f"Created dummy model (IR v10) at: {path}")

def test_inference():
    print("\n--- Testing Full C++ Pipeline ---")
    
    # A. Generate Dummy Model
    model_path = os.path.join(project_root, "dummy_model.onnx")
    create_dummy_model(model_path)

    # B. Initialize C++ Engine
    print("Initializing C++ InferenceEngine...")
    try:
        engine = audioguard_core.InferenceEngine(model_path)
    except Exception as e:
        print(f" Failed to load model: {e}")
        return

    # C. Create Fake Input (e.g., flattened spectrogram features)
    # Shape [1, 10] matches our dummy model
    input_data = [1.0] * 10 
    input_shape = [1, 10]

    print(f"Input Data: {input_data[:5]}...")

    # D. Run Prediction
    print("Running Prediction...")
    try:
        output = engine.predict(input_data, input_shape)
        print(f"Output Data: {output[:5]}...")
        
        # Check math (1.0 + 1.0 = 2.0)
        if abs(output[0] - 2.0) < 1e-5:
            print(" PASSED: Inference Engine is running correctly!")
        else:
            print(" FAILED: Output values are wrong.")
            
    except Exception as e:
        print(f" Prediction crashed: {e}")

if __name__ == "__main__":
    test_inference()