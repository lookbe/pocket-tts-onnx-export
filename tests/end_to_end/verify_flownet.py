import torch
import numpy as np
import onnxruntime as ort
import os
import argparse
import sys
from pathlib import Path

# Paths
sys.path.append(str(Path(__file__).parent.parent.parent))
from pocket_tts.models.tts_model import TTSModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--onnx_path", type=str, required=True)
    parser.add_argument("--weights_path", type=str, required=True)
    args = parser.parse_args()

    # Load PT model
    tts = TTSModel.load_model(config=args.config)
    from safetensors.torch import load_file
    state_dict = load_file(args.weights_path)
    tts.load_state_dict(state_dict, strict=True)
    tts.eval()
    
    flow_pt = tts.flow_lm.flow_net

    # Load ONNX model
    ort_sess = ort.InferenceSession(args.onnx_path)

    # Test inputs
    batch_size = 1
    torch.manual_seed(42)
    c = torch.randn(batch_size, 1024)
    s = torch.tensor([[0.3]])
    t = torch.tensor([[0.7]])
    x = torch.randn(batch_size, 32)
    
    with torch.no_grad():
        pt_out = flow_pt(c, s, t, x)
        
    onnx_inputs = {
        "c": c.numpy(),
        "s": s.numpy(),
        "t": t.numpy(),
        "x": x.numpy()
    }
    onnx_out = ort_sess.run(None, onnx_inputs)[0]
    
    print(f"PT Output (first 5): {pt_out[0, :5].numpy()}")
    print(f"ONNX Output (first 5): {onnx_out[0, :5]}")
    
    diff = np.abs(pt_out.numpy() - onnx_out)
    print(f"Max diff: {np.max(diff)}")
    print(f"Mean diff: {np.mean(diff)}")

if __name__ == "__main__":
    main()
