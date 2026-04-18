import os
import subprocess
import sys
from pathlib import Path

# Constants
OUTPUT_DIR = Path("onnx")
WEIGHTS_DIR = Path("weights")
SCRIPTS_DIR = Path("scripts")
REPO_ID = "kyutai/pocket-tts"
WEIGHTS_FILENAME = "tts_b6369a24.safetensors"

def install_check():
    try:
        import huggingface_hub
        print("✅ huggingface_hub is installed.")
    except ImportError:
        print("❌ huggingface_hub is missing. Please run: pip install -r requirements.txt")
        sys.exit(1)

def download_weights():
    print(f"\n--- Downloading Weights from {REPO_ID} ---")
    WEIGHTS_DIR.mkdir(exist_ok=True)
    
    from huggingface_hub import hf_hub_download
    
    try:
        local_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=WEIGHTS_FILENAME,
            local_dir=WEIGHTS_DIR,
            local_dir_use_symlinks=False
        )
        print(f"✅ Downloaded: {local_path}")
    except Exception as e:
        print(f"❌ Failed to download {WEIGHTS_FILENAME}: {e}")
        # Assuming manual intervention or pre-existing files if this fails
        # but stricter error handling might be desired.
    
    # Download tokenizer (optional but good to have)
    try:
        hf_hub_download(
            repo_id=REPO_ID,
            filename="tokenizer.model",
            local_dir=WEIGHTS_DIR,
            local_dir_use_symlinks=False
        )
        print("✅ Downloaded: tokenizer.model")
    except Exception:
        print("⚠️ tokenizer.model download failed (may be optional).")

def run_export_scripts():
    print(f"\n--- Running Export Scripts ---")
    
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Setup environment for subprocesses
    # Add current directory to PYTHONPATH so scripts can import 'pocket_tts' package
    env = os.environ.copy()
    env["PYTHONPATH"] = "." + os.pathsep + env.get("PYTHONPATH", "")
    
    # Common arguments
    weights_path = str(WEIGHTS_DIR / WEIGHTS_FILENAME)
    output_dir_str = str(OUTPUT_DIR)
    
    # 1. Export Mimi & Conditioner
    print("\n[1/2] Exporting Mimi & Text Conditioner...")
    cmd1 = [
        sys.executable, 
        str(SCRIPTS_DIR / "export_mimi_and_conditioner.py"),
        "--output_dir", output_dir_str,
        "--weights_path", weights_path
    ]
    try:
        subprocess.run(cmd1, check=True, env=env)
        print("✅ Mimi/Conditioner Export Success")
    except subprocess.CalledProcessError:
        print("❌ Mimi/Conditioner Export Failed")
        sys.exit(1)

    # 2. Export FlowLM
    print("\n[2/2] Exporting FlowLM (Split Models)...")
    cmd2 = [
        sys.executable, 
        str(SCRIPTS_DIR / "export_flow_lm.py"),
        "--output_dir", output_dir_str,
        "--weights_path", weights_path
    ]
    try:
        subprocess.run(cmd2, check=True, env=env)
        print("✅ FlowLM Export Success")
    except subprocess.CalledProcessError:
        print("❌ FlowLM Export Failed")
        sys.exit(1)

def run_quantization():
    print(f"\n--- [Optional] Running Quantization ---")
    
    # Check if input directory has models
    if not any(OUTPUT_DIR.glob("*.onnx")):
        print("⚠️ No models found in output directory to quantize.")
        return

    # Quantization Output Directory (Same as input for PocketTTS compatibility)
    QUANT_DIR = OUTPUT_DIR
    
    # Setup environment
    env = os.environ.copy()
    env["PYTHONPATH"] = "." + os.pathsep + env.get("PYTHONPATH", "")
    
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "quantize.py"),
        "--input_dir", str(OUTPUT_DIR),
        "--output_dir", str(QUANT_DIR)
    ]
    
    try:
        subprocess.run(cmd, check=True, env=env)
        print(f"✅ Quantization Success! INT8 models in: {QUANT_DIR.absolute()}")
    except subprocess.CalledProcessError:
        print("❌ Quantization Failed")

def print_summary():
    print(f"\n✅ All Done! Models are in: {OUTPUT_DIR.absolute()}")
    print("Files:")
    if OUTPUT_DIR.exists():
        for f in sorted(OUTPUT_DIR.glob("*.onnx")):
            s = f.stat().st_size / (1024*1024)
            print(f" - {f.name:<30} ({s:.1f} MB)")
    else:
        print("⚠️ Output directory missing.")

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Export Script for PocketTTS")
    parser.add_argument("--quantize", action="store_true", help="Run INT8 quantization after export")
    args = parser.parse_args()

    install_check()
    download_weights()
    run_export_scripts()
    
    if args.quantize:
        run_quantization()
        
    print_summary()
