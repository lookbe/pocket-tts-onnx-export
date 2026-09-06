import os
import sys
import subprocess
from pathlib import Path
import argparse
import yaml
import shutil
from huggingface_hub import hf_hub_download

# Constants
MODELS_DIR = Path("models")
SCRIPTS_DIR = Path("scripts")
CONFIG_DIR = Path("pocket_tts") / "config"

# Low-memory variant: int4 + external (separated) weight data + a shrunk static KV-cache/
# conv-state capacity (1000 -> 400 tokens). See agents/low_mem.md for why these three go
# together. Lives in its own directory tree, mirroring MODELS_DIR per language.
LOW_MEM_DIR = Path("models_low_mem")
LOW_MEM_SEQ_LEN = 400

def parse_hf_url(url):
    """Parses hf://repo_id/filename@revision into (repo_id, filename, revision)"""
    if not url.startswith("hf://"):
        return None
    url = url[len("hf://"):]
    
    revision = None
    if "@" in url:
        url, revision = url.split("@", 1)
        
    parts = url.split("/")
    if len(parts) < 3:
        return None
        
    # Standard format: hf://owner/repo/path/to/file
    repo_id = "/".join(parts[:2])
    filename = "/".join(parts[2:])
    
    return repo_id, filename, revision

def hf_download(url, target_dir, target_filename):
    """Downloads a file from Hugging Face and saves it to a specific local path."""
    parsed = parse_hf_url(url)
    if not parsed:
        print(f"Error: Could not parse Hugging Face URL: {url}")
        return False
        
    repo_id, filename, revision = parsed
    print(f"Downloading {filename} from {repo_id}@{revision or 'main'}...")
    
    try:
        # Ensure target_dir exists
        target_dir.mkdir(parents=True, exist_ok=True)
        
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            local_dir=target_dir,
            local_dir_use_symlinks=False
        )
        
        target_path = target_dir / target_filename
        if Path(downloaded_path) != target_path:
            # Move/rename the downloaded file to the target filename
            if target_path.exists():
                os.remove(target_path)
            shutil.move(downloaded_path, target_path)
            
            # Cleanup potentially empty subdirectories created by hf_hub_download
            parts = filename.split("/")
            if len(parts) > 1:
                subfolder = target_dir / parts[0]
                if subfolder.exists() and subfolder.is_dir():
                    shutil.rmtree(subfolder)
                    
        # Remove .cache directory if it was created in the local_dir
        cache_dir = target_dir / ".cache"
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            
        print(f"Successfully downloaded to {target_path}")
        return True
    except Exception as e:
        print(f"Failed to download: {e}")
        return False

def download_safetensors(lang_name, config_path, lang_dir):
    """Downloads model.safetensors if missing using info from config YAML"""
    print(f"Weights missing. Attempting to download for {lang_name}...")
    
    if not config_path.exists():
        print(f"Error: Config {config_path} not found. Cannot download.")
        return False
        
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            
        weights_url = config.get("weights_path")
        if not weights_url:
            print(f"Error: 'weights_path' not found in {config_path}")
            return False
            
        return hf_download(weights_url, lang_dir, "model.safetensors")
    except Exception as e:
        print(f"Error reading config for weights: {e}")
        return False

def download_tokenizer(lang_name, config_path, lang_dir):
    """Downloads tokenizer.model if missing using info from config YAML"""
    print(f"Tokenizer missing. Attempting to download for {lang_name}...")
    
    if not config_path.exists():
        print(f"Error: Config {config_path} not found. Cannot download.")
        return False
        
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            
        flow_lm = config.get("flow_lm", {})
        lookup_table = flow_lm.get("lookup_table", {})
        tokenizer_url = lookup_table.get("tokenizer_path")
        
        if not tokenizer_url:
            print(f"Error: 'flow_lm.lookup_table.tokenizer_path' not found in {config_path}")
            return False
            
        return hf_download(tokenizer_url, lang_dir, "tokenizer.model")
    except Exception as e:
        print(f"Error reading config for tokenizer: {e}")
        return False

def run_cmd(cmd, env):
    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        return False
    return True

def export_language(lang_dir: Path):
    lang_name = lang_dir.name
    weights_path = lang_dir / "model.safetensors"
    header_path = lang_dir / "header.json"
    
    # Find matching config in pocket_tts/config/
    config_path = CONFIG_DIR / f"{lang_name}.yaml"

    
    if not weights_path.exists():
        print(f"Skipping {lang_name}: {weights_path} not found.")
        return False
    
    if not config_path.exists():
        print(f"FAILED: Config for {lang_name} not found at {config_path}")
        return False

    print(f"\n{'='*60}")
    print(f"Processing Language: {lang_name} (v2/Multilingual)")
    print(f"Weights: {weights_path}")
    print(f"Config:  {config_path}")
    
    if header_path.exists():
        import json
        with open(header_path, "r") as f:
            header = json.load(f)
            # Check for v2 indicators in header (e.g. 32-dim latent bottleneck)
            is_v2 = False
            if "flow_lm.emb_mean" in header:
                shape = header["flow_lm.emb_mean"].get("shape", [])
                if shape == [32]:
                    is_v2 = True
                    print(f"Header: Validated v2 architecture (32-dim latent)")
            
            if not is_v2 and "english_v1" not in lang_name:
                print(f"Warning: {lang_name} might not be a v2 model. Proceeding anyway...")
    
    print(f"{'='*60}")


    env = os.environ.copy()
    # Ensure current directory is in PYTHONPATH for pocket_tts imports
    # Also set UTF-8 encoding to handle emojis from torch.onnx on Windows
    env["PYTHONPATH"] = "." + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONIOENCODING"] = "utf-8"

    # Check if all base ONNX files exist to skip export steps
    required_files = [
        "mimi_encoder.onnx",
        "text_conditioner.onnx",
        "mimi_decoder.onnx",
        "flow_lm_main.onnx",
        "flow_lm_flow.onnx",
        "bos_before_voice.npy"
    ]
    all_files_exist = all((lang_dir / f).exists() for f in required_files)

    if all_files_exist:
        print(f"\nSkipping Safetensors -> ONNX export for {lang_name}: All base models already exist.")
    else:
        # 1. Export Mimi & Conditioner
        print(f"\n[1/3] Exporting Mimi & Text Conditioner for {lang_name}...")
        mimi_cmd = [
            sys.executable,
            str(SCRIPTS_DIR / "export_mimi_and_conditioner.py"),
            "--output_dir", str(lang_dir),
            "--weights_path", str(weights_path),
            "--config", str(config_path)
        ]
        if not run_cmd(mimi_cmd, env): 
            print(f"FAILED: Mimi/Conditioner Export Failed for {lang_name}")
            return False

        # 2. Export FlowLM
        print(f"\n[2/3] Exporting FlowLM (Split Models) for {lang_name}...")
        flow_cmd = [
            sys.executable,
            str(SCRIPTS_DIR / "export_flow_lm.py"),
            "--output_dir", str(lang_dir),
            "--weights_path", str(weights_path),
            "--config", str(config_path)
        ]
        
        if not run_cmd(flow_cmd, env):
            print(f"FAILED: FlowLM Export Failed for {lang_name}")
            return False

    # 3. Quantize to INT8
    print(f"\n[3/4] Quantizing ONNX models to INT8 for {lang_name}...")
    quant_cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "quantize.py"),
        "--input_dir", str(lang_dir),
        "--output_dir", str(lang_dir)
    ]
    if not run_cmd(quant_cmd, env):
        print(f"FAILED: Quantization Failed for {lang_name}")
        return False

    # 4. Quantize to INT4 (single embedded .onnx per model; mimi_encoder stays fp32)
    print(f"\n[4/4] Quantizing ONNX models to INT4 for {lang_name}...")
    quant_int4_cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "quantize_int4.py"),
        "--input_dir", str(lang_dir),
        "--output_dir", str(lang_dir)
    ]
    if not run_cmd(quant_int4_cmd, env):
        print(f"FAILED: INT4 Quantization Failed for {lang_name}")
        return False

    print(f"\nSUCCESS: Successfully processed {lang_name}")
    return True


def export_language_low_mem(lang_dir: Path, weights_path: Path, config_path: Path):
    """
    Low-memory variant for the same language: int4 + separated (external) weight data +
    a shrunk static state capacity (LOW_MEM_SEQ_LEN instead of 1000). Requires its own
    fp32 export because the KV-cache/conv-state shapes are baked into the graph at export
    time -- see agents/low_mem.md.
    """
    lang_name = lang_dir.name
    low_mem_lang_dir = LOW_MEM_DIR / lang_name

    print(f"\n{'='*60}")
    print(f"Processing Language (low-mem): {lang_name}")
    print(f"{'='*60}")

    env = os.environ.copy()
    env["PYTHONPATH"] = "." + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONIOENCODING"] = "utf-8"

    required_files = [
        "mimi_encoder.onnx",
        "text_conditioner.onnx",
        "mimi_decoder.onnx",
        "flow_lm_main.onnx",
        "flow_lm_flow.onnx",
        "bos_before_voice.npy",
    ]
    all_files_exist = all((low_mem_lang_dir / f).exists() for f in required_files)

    if all_files_exist:
        print(f"Skipping Safetensors -> ONNX export for {lang_name} (low-mem): All base models already exist.")
    else:
        print(f"\n[1/3] Exporting Mimi & Text Conditioner for {lang_name} (low-mem, seq_len={LOW_MEM_SEQ_LEN})...")
        mimi_cmd = [
            sys.executable,
            str(SCRIPTS_DIR / "export_mimi_and_conditioner.py"),
            "--output_dir", str(low_mem_lang_dir),
            "--weights_path", str(weights_path),
            "--config", str(config_path),
            "--seq_len", str(LOW_MEM_SEQ_LEN),
        ]
        if not run_cmd(mimi_cmd, env):
            print(f"FAILED: Mimi/Conditioner Export Failed for {lang_name} (low-mem)")
            return False

        print(f"\n[2/3] Exporting FlowLM (Split Models) for {lang_name} (low-mem, seq_len={LOW_MEM_SEQ_LEN})...")
        flow_cmd = [
            sys.executable,
            str(SCRIPTS_DIR / "export_flow_lm.py"),
            "--output_dir", str(low_mem_lang_dir),
            "--weights_path", str(weights_path),
            "--config", str(config_path),
            "--seq_len", str(LOW_MEM_SEQ_LEN),
        ]
        if not run_cmd(flow_cmd, env):
            print(f"FAILED: FlowLM Export Failed for {lang_name} (low-mem)")
            return False

    # Carry over deployment assets that aren't regenerated by the export scripts above.
    tokenizer_src = lang_dir / "tokenizer.model"
    tokenizer_dst = low_mem_lang_dir / "tokenizer.model"
    if tokenizer_src.exists() and not tokenizer_dst.exists():
        shutil.copy2(tokenizer_src, tokenizer_dst)

    # 3. Quantize to INT8
    print(f"\n[3/4] Quantizing ONNX models to INT8 for {lang_name} (low-mem)...")
    quant_cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "quantize.py"),
        "--input_dir", str(low_mem_lang_dir),
        "--output_dir", str(low_mem_lang_dir),
    ]
    if not run_cmd(quant_cmd, env):
        print(f"FAILED: Quantization Failed for {lang_name} (low-mem)")
        return False

    # 4. Quantize to INT4 with external (separated) weight data
    print(f"\n[4/4] Quantizing ONNX models to INT4 (separated data) for {lang_name} (low-mem)...")
    quant_int4_cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "quantize_int4.py"),
        "--input_dir", str(low_mem_lang_dir),
        "--output_dir", str(low_mem_lang_dir),
        "--separate_data",
    ]
    if not run_cmd(quant_int4_cmd, env):
        print(f"FAILED: INT4 Quantization Failed for {lang_name} (low-mem)")
        return False

    print(f"\nSUCCESS: Successfully processed {lang_name} (low-mem)")
    return True

def main():
    parser = argparse.ArgumentParser(description="Multilingual Export and Quantization Script")
    parser.add_argument("--lang", type=str, help="Specific language config name to process (optional)")
    parser.add_argument("--skip_low_mem", action="store_true", help="Skip the low-mem (int4 + separated data + reduced state) variant")
    args = parser.parse_args()

    if not MODELS_DIR.exists():
        print(f"Creating models directory at {MODELS_DIR.absolute()}")
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_low_mem:
        LOW_MEM_DIR.mkdir(parents=True, exist_ok=True)

    if args.lang:
        langs_to_process = [args.lang]
    else:
        # Scan for all YAML configs in pocket_tts/config/
        print(f"Scanning for configurations in {CONFIG_DIR.absolute()}...")
        langs_to_process = [f.stem for f in CONFIG_DIR.glob("*.yaml")]
        
        if not langs_to_process:
            print("No configuration files found in pocket_tts/config/")
            return

    print(f"Found {len(langs_to_process)} languages to process.")
    
    processed_count = 0
    failed_langs = []
    
    for lang_name in langs_to_process:
        lang_dir = MODELS_DIR / lang_name
        config_path = CONFIG_DIR / f"{lang_name}.yaml"
        weights_path = lang_dir / "model.safetensors"
        
        # Auto-setup: Ensure folder, weights and tokenizer exist
        if not weights_path.exists():
            if not download_safetensors(lang_name, config_path, lang_dir):
                print(f"Skipping {lang_name} due to missing weights.")
                continue
        else:
            print(f"Weights already present for {lang_name}, skipping download.")

        tokenizer_path = lang_dir / "tokenizer.model"
        if not tokenizer_path.exists():
            if not download_tokenizer(lang_name, config_path, lang_dir):
                print(f"Warning: Could not download tokenizer for {lang_name}. Export might fail if not in cache.")
        
        if export_language(lang_dir):
            processed_count += 1
        else:
            failed_langs.append(lang_name)
            continue

        if not args.skip_low_mem:
            if not export_language_low_mem(lang_dir, weights_path, config_path):
                failed_langs.append(f"{lang_name} (low-mem)")
    
    print(f"\n{'='*60}")
    print(f"Final Summary:")
    print(f" - Total languages processed: {processed_count}")
    if failed_langs:
        print(f" - Failed languages: {', '.join(failed_langs)}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
