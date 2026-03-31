
import os
import argparse
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
from pathlib import Path

MODELS_TO_QUANTIZE = [
    "flow_lm_main",
    "flow_lm_flow",
    "mimi_decoder",
    "mimi_encoder",
    "text_conditioner"
]

def quantize_file(input_path: Path, output_path: Path, op_types=['MatMul']):
    """
    Quantize a single ONNX file using dynamic quantization.
    """
    if not input_path.exists():
        print(f"Skipping {input_path.name} (not found)")
        return

    print(f"Quantizing {input_path.name}...")
    
    # Run shape inference to fix missing types (helps with quantization stability)

    try:
        print(f"  Running shape inference...")
        # Load model
        model = onnx.load(str(input_path))
        # Infer shapes
        model = onnx.shape_inference.infer_shapes(model)
        
        # Save to temp file for quantization input
        temp_path = output_path.with_suffix(".temp.onnx")
        onnx.save(model, str(temp_path))
        
        # Quantize
        quantize_dynamic(
            model_input=str(temp_path),
            model_output=str(output_path),
            weight_type=QuantType.QInt8,
            op_types_to_quantize=op_types,
            extra_options={'ForceQuantizeNoType': True, 'DefaultTensorType': 1}
        )
        
        # Cleanup
        if temp_path.exists():
            temp_path.unlink()
            
        # Stats
        size_orig = input_path.stat().st_size / (1024 * 1024)
        size_quant = output_path.stat().st_size / (1024 * 1024)
        reduction = (size_orig - size_quant) / size_orig * 100
        print(f"  OK Complete: {size_orig:.1f}MB -> {size_quant:.1f}MB ({reduction:.1f}% reduction)")
        
    except Exception as e:
        print(f"  FAILED Quantization failed for {input_path.name}: {e}")
        # Clean up partial output
        if output_path.exists():
            output_path.unlink()

def main():
    parser = argparse.ArgumentParser(description="Quantize PocketTTS ONNX models to INT8.")
    parser.add_argument("--input_dir", "-i", type=str, default="onnx", help="Input directory containing FP32 ONNX models")
    parser.add_argument("--output_dir", "-o", type=str, default="onnx_int8", help="Output directory for INT8 ONNX models")
    parser.add_argument("--models", "-m", nargs="+", help="Specific models to quantize (e.g. flow_lm_main)")
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    
    models_to_run = args.models if args.models else MODELS_TO_QUANTIZE
    
    print(f"Starting Quantization: {input_dir} -> {output_dir}")
    print(f"Models: {', '.join(models_to_run)}")
    
    for model_name in models_to_run:
        in_file = input_dir / f"{model_name}.onnx"
        out_file = output_dir / f"{model_name}_int8.onnx"
        # Standardize on MatMul/Gemm for best stability/speed trade-off on CPU
        quantize_file(in_file, out_file, op_types=['MatMul', 'Gemm'])

    print("\nQuantization routine finished.")

if __name__ == "__main__":
    main()
