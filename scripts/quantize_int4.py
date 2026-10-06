import argparse
import sys
from pathlib import Path

import onnx
from onnxruntime.quantization.matmul_nbits_quantizer import (
    MatMulNBitsQuantizer,
    DefaultWeightOnlyQuantConfig,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from onnx_export.export_utils import convert_gemm_to_matmul_add, cast_weights_to_float16

# Every model here is forced to weight-only INT4 via onnxruntime's MatMulNBitsQuantizer.
# `--separate_data` controls only the on-disk container (single embedded .onnx vs.
# .onnx + external .onnx.data); the quantization itself is identical either way.
#
# Caveats (onnxruntime's MatMulNBitsQuantizer only understands MatMul and Gather):
#   - text_conditioner is a single Gather (embedding) op -> quantized via the Gather path.
#   - flow_lm_flow is exported almost entirely as Gemm nodes (torch.onnx's legacy exporter
#     turns nn.Linear into Gemm). Gemm is NOT quantized by MatMulNBitsQuantizer, so we first
#     rewrite Gemm(A, W, bias) -> MatMul(A, W^T) + Add(bias), then quantize the MatMuls.
#   - flow_lm_main is ~100% MatMul already, quantized directly.
#   - mimi_decoder and mimi_encoder are SEANet conv stacks (+ a transformer): their MatMul
#     portion (~61% of decoder weight bytes, ~65% of encoder weight bytes) gets int4 quantized
#     directly. Their Conv/ConvTranspose weights (the rest) have no weight-only int4 path in
#     onnxruntime (MatMulNBitsQuantizer only supports MatMul/Gather), so instead we store those
#     weights as fp16-on-disk with a Cast(fp16->fp32) feeding each Conv -- every op still
#     computes in fp32 (no fp16-kernel/type-propagation risk), only the stored weight bytes are
#     halved. Numerically this is a no-op beyond fp16 rounding (cosine sim to the un-cast int4
#     decoder model measured at 0.9999999).
#
#     mimi_encoder is used for voice cloning, which is infrequent, so quantization quality risk
#     matters less here than for the always-on decoder/flow-lm path.
MODELS = ["text_conditioner", "flow_lm_flow", "flow_lm_main", "mimi_decoder", "mimi_encoder"]


# --low_mem: only these are INT4 (block 64, accuracy_level 4 = int8 compute in MatMulNBits); the rest of the
# low-mem set (text_conditioner, flow_lm_flow) is INT8 from quantize.py; mimi_encoder stays fp32.
LOW_MEM_MODELS = ["flow_lm_main", "mimi_decoder"]
LOW_MEM_BLOCK_SIZE = 64
LOW_MEM_ACCURACY_LEVEL = 4


def quantize_int4(model, op_types_to_quantize, nodes_to_include=None, block_size=128, nodes_to_exclude=None,
                  accuracy_level=None):
    config_kwargs = {}
    if accuracy_level is not None:
        config_kwargs["accuracy_level"] = accuracy_level
    quantizer = MatMulNBitsQuantizer(
        model=model,
        nodes_to_include=nodes_to_include,
        nodes_to_exclude=nodes_to_exclude,
        algo_config=DefaultWeightOnlyQuantConfig(
            block_size=block_size,
            is_symmetric=True,
            op_types_to_quantize=op_types_to_quantize,
            **config_kwargs,
        ),
    )
    quantizer.process()
    return quantizer.model.model


def total_size(path: Path) -> int:
    size = path.stat().st_size
    data_path = path.with_name(path.name + ".data")
    if data_path.exists():
        size += data_path.stat().st_size
    return size


def report_size(input_path: Path, output_path: Path):
    size_in = total_size(input_path) / (1024 * 1024)
    size_out = total_size(output_path) / (1024 * 1024)
    reduction = (size_in - size_out) / size_in * 100
    print(f"  {size_in:.1f}MB -> {size_out:.1f}MB ({reduction:.1f}% reduction)")


def main():
    parser = argparse.ArgumentParser(description="Quantize PocketTTS ONNX models to weight-only INT4.")
    parser.add_argument("--input_dir", "-i", type=str, required=True, help="Directory containing FP32 ONNX models")
    parser.add_argument("--output_dir", "-o", type=str, required=True, help="Output directory for quantized ONNX models")
    parser.add_argument("--separate_data", action="store_true", help="Save each quantized model as .onnx + external .onnx.data weights file (instead of a single embedded .onnx)")
    parser.add_argument("--low_mem", action="store_true", help="Low-memory preset: INT4 (block 64, accuracy_level 4) for flow_lm_main and mimi_decoder only")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    block_override = LOW_MEM_BLOCK_SIZE if args.low_mem else None
    accuracy_level = LOW_MEM_ACCURACY_LEVEL if args.low_mem else None

    for model_name in (LOW_MEM_MODELS if args.low_mem else MODELS):
        in_file = input_dir / f"{model_name}.onnx"
        if not in_file.exists():
            print(f"Skipping {model_name} (not found in {input_dir})")
            continue

        out_file = output_dir / f"{model_name}_int4.onnx"
        print(f"\n[{model_name}] -> INT4 ({'separated' if args.separate_data else 'embedded'})")
        model = onnx.load(str(in_file), load_external_data=True)

        if model_name == "text_conditioner":
            quantized = quantize_int4(model, op_types_to_quantize=("Gather",), block_size=32)
        elif model_name == "flow_lm_flow":
            converted = convert_gemm_to_matmul_add(model)
            print(f"  Converted {converted} Gemm nodes to MatMul+Add (so they're quantizable).")
            quantized = quantize_int4(model, op_types_to_quantize=("MatMul",))
        elif model_name == "flow_lm_main":
            quantized = quantize_int4(model, op_types_to_quantize=("MatMul",),
                                      block_size=block_override or 128, accuracy_level=accuracy_level)
        elif model_name in ("mimi_decoder", "mimi_encoder"):
            # The upsampling ConvTranspose layers are exported as MatMul + overlap-add (OPTIMIZATION.md); their MatMuls
            # (/decoder/model.N/MatMul) are convolution weights, not linears: keep them out of INT4, and store them
            # fp16-on-disk like the other conv weights (the still-fp32 MatMuls left after quantization).
            conv_matmuls = [n.name for n in model.graph.node
                            if n.op_type == "MatMul" and n.name.startswith("/decoder/model.")]
            quantized = quantize_int4(model, op_types_to_quantize=("MatMul",), nodes_to_exclude=conv_matmuls,
                                      block_size=block_override or 128, accuracy_level=accuracy_level)
            n_cast = cast_weights_to_float16(quantized, op_types=("Conv", "ConvTranspose") + (("MatMul",) if conv_matmuls else ()))
            print(f"  Excluded {len(conv_matmuls)} ConvTranspose MatMuls from INT4; cast {n_cast} Conv/ConvTranspose/MatMul weight/bias tensors to fp16 (compute stays fp32).")
        else:
            continue

        # Always round-trip through external data first (keeps the in-memory protobuf small
        # regardless of final format), then merge back to a single file unless requested.
        onnx.save_model(
            quantized,
            str(out_file),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=out_file.name + ".data",
            size_threshold=0,
            convert_attribute=False,
        )

        report_size(in_file, out_file)

        if not args.separate_data:
            merged = onnx.load(str(out_file), load_external_data=True)
            onnx.save_model(merged, str(out_file), save_as_external_data=False)
            data_file = out_file.with_name(out_file.name + ".data")
            if data_file.exists():
                data_file.unlink()

    print("\nINT4 quantization finished.")


if __name__ == "__main__":
    main()
