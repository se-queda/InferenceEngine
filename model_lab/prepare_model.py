import onnx
from onnx import version_converter, helper

# Path to your REAL model
input_path = "model_lab/model_quantized.onnx"
output_path = "model_lab/model_final.onnx"

print(f"--- Inspecting {input_path} ---")

try:
    # 1. Load the model
    model = onnx.load(input_path)
    
    # 2. Check Input Info (Crucial for C++)
    input_tensor = model.graph.input[0]
    input_name = input_tensor.name
    
    # Get shape
    input_shape = []
    for d in input_tensor.type.tensor_type.shape.dim:
        if d.dim_value > 0:
            input_shape.append(d.dim_value)
        else:
            input_shape.append(1) # Assume batch size 1 for unknown dims
            
    print(f"✅ Input Name:  '{input_name}'")
    print(f"✅ Input Shape: {input_shape}")

    # 3. Check Version (The "Time Travel" Fix)
    current_ir = model.ir_version
    print(f"ℹ️  Current IR Version: {current_ir}")
    
    # If it's too new (Version 13+), we must downgrade it for C++
    # We essentially "Save As..." a simplified version
    if current_ir > 10:
        print("⚠️  Model is too new for C++ Runtime. Converting...")
        
        # We manually force the IR version to 10 (Stable)
        model.ir_version = 10
        
        # We also ensure the Opset (Operator Set) is compatible (v12 is very safe)
        del model.opset_import[:]
        opset = model.opset_import.add()
        opset.domain = ""
        opset.version = 12 
        
        print(f"💾 Saving compatible model to: {output_path}")
        onnx.save(model, output_path)
        print("✅ Conversion Complete. Use 'model_final.onnx' in C++.")
    else:
        print(f"✅ Model version is safe. You can use '{input_path}'.")

except Exception as e:
    print(f"❌ Error: {e}")