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
            
        parsed = parse_hf_url(weights_url)
        if not parsed:
            print(f"Error: Could not parse Hugging Face URL: {weights_url}")
            return False
            
        repo_id, filename, revision = parsed
        
        print(f"Downloading {filename} from {repo_id}@{revision or 'main'}...")
        
        # Ensure lang_dir exists
        lang_dir.mkdir(parents=True, exist_ok=True)
        
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            local_dir=lang_dir,
            local_dir_use_symlinks=False
        )
        
        target_path = lang_dir / "model.safetensors"
        if Path(downloaded_path) != target_path:
            # If the filename in HF isn't 'model.safetensors', move/rename it
            shutil.move(downloaded_path, target_path)
            
            # Cleanup potentially empty subdirectories created by hf_hub_download
            parts = filename.split("/")
            if len(parts) > 1:
                subfolder = lang_dir / parts[0]
                if subfolder.exists() and subfolder.is_dir():
                    shutil.rmtree(subfolder)
                    
        # Remove .cache directory if it was created in the local_dir
        cache_dir = lang_dir / ".cache"
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            
        print(f"Successfully downloaded weights to {target_path}")
        return True
    except Exception as e:
        print(f"Failed to download weights: {e}")
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

    # 3. Quantize
    print(f"\n[3/3] Quantizing ONNX models to INT8 for {lang_name}...")
    quant_cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "quantize.py"),
        "--input_dir", str(lang_dir),
        "--output_dir", str(lang_dir)
    ]
    if not run_cmd(quant_cmd, env):
        print(f"FAILED: Quantization Failed for {lang_name}")
        return False

    print(f"\nSUCCESS: Successfully processed {lang_name}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Multilingual Export and Quantization Script")
    parser.add_argument("--lang", type=str, help="Specific language config name to process (optional)")
    args = parser.parse_args()

    if not MODELS_DIR.exists():
        print(f"Creating models directory at {MODELS_DIR.absolute()}")
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

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
        
        # Auto-setup: Ensure folder and weights exist
        if not weights_path.exists():
            if not download_safetensors(lang_name, config_path, lang_dir):
                print(f"Skipping {lang_name} due to missing weights.")
                continue
        
        if export_language(lang_dir):
            processed_count += 1
        else:
            failed_langs.append(lang_name)
    
    print(f"\n{'='*60}")
    print(f"Final Summary:")
    print(f" - Total languages processed: {processed_count}")
    if failed_langs:
        print(f" - Failed languages: {', '.join(failed_langs)}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
