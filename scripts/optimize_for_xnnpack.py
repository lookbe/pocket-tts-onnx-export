import os
import argparse
import onnx
try:
    from onnxruntime.transformers.optimizer import optimize_model, OptimizerType
except ImportError:
    print("⚠️ onnxruntime.transformers is missing. Basic optimization only.")
    optimize_model = None

from pathlib import Path

def optimize_onnx_model(input_path: Path, output_path: Path):
    print(f"Optimizing {input_path.name}...")
    
    if optimize_model is None:
        import shutil
        shutil.copy(str(input_path), str(output_path))
        return

    # 1. ORT Optimization (LayerNorm, Attention fusion etc.)
    # Note: We use 'gpt2' as the optimizer type because FlowLM/Mimi-Transformer are GPT-like
    try:
        optimized_model = optimize_model(
            str(input_path),
            model_type='gpt2', 
            num_heads=0, # Auto-detect
            hidden_size=0, # Auto-detect
            opt_level=99, # Max optimization
        )
        
        # 2. XNNPACK specific: Ensure layout is NCHW and fuse certain patterns
        optimized_model.use_dynamic_axes(True)
        
        # Save
        optimized_model.save_model_to_file(str(output_path))
        print(f"  ✅ Saved optimized model to {output_path.name}")
        
    except Exception as e:
        print(f"  ⚠️ ORT Optimization failed for {input_path.name}: {e}")
        print(f"  Falling back to basic copy...")
        import shutil
        shutil.copy(str(input_path), str(output_path))

def main():
    parser = argparse.ArgumentParser(description="Optimize PocketTTS ONNX models for XNNPACK/Android.")
    parser.add_argument("--input_dir", "-i", type=str, default="onnx", help="Input directory")
    parser.add_argument("--output_dir", "-o", type=str, default="onnx_optimized", help="Output directory")
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for model_path in input_dir.glob("*.onnx"):
        if "_int8" in model_path.name:
            continue
        
        out_path = output_dir / model_path.name
        optimize_onnx_model(model_path, out_path)

if __name__ == "__main__":
    main()
