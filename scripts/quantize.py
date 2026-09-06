import os
import argparse
import onnx
import re
from onnxruntime.quantization import quantize_dynamic, QuantType
from pathlib import Path

# Settings for selective quantization
QUANTIZE_MIDDLE_LAYERS_ONLY = False # If True, only quantize middle 4 layers (1-4) of FlowLM

MODELS_TO_QUANTIZE = [
    "text_conditioner",
    "flow_lm_main",
    "flow_lm_flow",
    "mimi_decoder",
    "mimi_encoder",
]


def total_size(path: Path) -> int:
    size = path.stat().st_size
    data_path = path.with_name(path.name + ".data")
    if data_path.exists():
        size += data_path.stat().st_size
    return size


def quantize_file(input_path: Path, output_path: Path, model_name: str, separate_data: bool = False):
    """Quantize a single ONNX file using dynamic quantization."""
    if not input_path.exists():
        print(f"Skipping {input_path.name} (not found)")
        return

    print(f"Quantizing {input_path.name} (model_name={model_name})...")
    
    # Selective node quantization logic
    nodes_to_quantize = []
    op_types_to_quantize = ["MatMul", "Gemm"]
    if model_name == "text_conditioner":
        # text_conditioner is a single embedding lookup (Gather), not MatMul/Gemm.
        op_types_to_quantize = ["Gather"]
        print("  Quantizing embedding Gather op...")
    elif model_name == "flow_lm_main":
        print(f"  Applying selective node quantization (Transformer backbone, middle_only={QUANTIZE_MIDDLE_LAYERS_ONLY})...")
        model = onnx.load(str(input_path))
        for node in model.graph.node:
            if node.op_type in ["MatMul", "Gemm"]:
                name = node.name
                # Target Transformer layers: in_proj, out_proj, linear1, linear2
                # Skip input_linear and attention score MatMuls (activation*activation)
                if "/transformer/" in name:
                    if any(x in name for x in ["/in_proj/", "/out_proj/", "/linear1/", "/linear2/"]):
                        if QUANTIZE_MIDDLE_LAYERS_ONLY:
                            # Extract layer index from name like ".../layers.0/..."
                            match = re.search(r"/layers\.(\d+)/", name)
                            if match:
                                layer_idx = int(match.group(1))
                                if 1 <= layer_idx <= 4:
                                    nodes_to_quantize.append(name)
                        else:
                            nodes_to_quantize.append(name)
        print(f"  Selected {len(nodes_to_quantize)} nodes for quantization.")
    
    temp_path = None
    try:
        print("  Running shape inference...")
        model = onnx.load(str(input_path))
        model = onnx.shape_inference.infer_shapes(model)

        temp_path = output_path.with_suffix(".temp.onnx")
        onnx.save(model, str(temp_path))

        # If nodes_to_quantize is empty and we are flow_lm_main, it means we found nothing.
        # If it's empty for other models, quantize_dynamic will use op_types_to_quantize.
        quant_args = {
            "model_input": str(temp_path),
            "model_output": str(output_path),
            "weight_type": QuantType.QInt8,
            "op_types_to_quantize": op_types_to_quantize,
            "per_channel": True,
            "extra_options": {"ForceQuantizeNoType": True, "DefaultTensorType": 1},
            "use_external_data_format": separate_data,
        }

        if nodes_to_quantize:
            quant_args["nodes_to_quantize"] = nodes_to_quantize

        quantize_dynamic(**quant_args)

        size_orig = input_path.stat().st_size / (1024 * 1024)
        size_quant = total_size(output_path) / (1024 * 1024)
        reduction = (size_orig - size_quant) / size_orig * 100
        print(f"  Complete: {size_orig:.1f}MB -> {size_quant:.1f}MB ({reduction:.1f}% reduction)")
    except Exception as e:
        print(f"  Quantization failed for {input_path.name}: {e}")
        if output_path.exists():
            output_path.unlink()
        data_path = output_path.with_name(output_path.name + ".data")
        if data_path.exists():
            data_path.unlink()
        raise
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def main():
    parser = argparse.ArgumentParser(description="Quantize PocketTTS ONNX models to INT8.")
    parser.add_argument("--input_dir", "-i", type=str, default="onnx", help="Input directory containing FP32 ONNX models")
    parser.add_argument("--output_dir", "-o", type=str, default="onnx_int8", help="Output directory for INT8 ONNX models")
    parser.add_argument("--separate_data", action="store_true", help="Save each quantized model as .onnx + external .onnx.data weights file (instead of a single embedded .onnx)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting Quantization: {input_dir} -> {output_dir}")
    print("Using dynamic MatMul/Gemm quantization (per_channel=True) for CPU compatibility.")

    for model_name in MODELS_TO_QUANTIZE:
        in_file = input_dir / f"{model_name}.onnx"
        out_file = output_dir / f"{model_name}_int8.onnx"
        quantize_file(in_file, out_file, model_name, separate_data=args.separate_data)

    print("\nQuantization routine finished.")


if __name__ == "__main__":
    main()
