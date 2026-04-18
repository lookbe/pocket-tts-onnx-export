
import os
import torch
import numpy as np
import onnxruntime as ort
import torch.nn.functional as F
import soundfile as sf
from scipy.signal import resample_poly
import safetensors.torch
from pathlib import Path

# Pocket TTS imports
from pocket_tts.models.mimi import MimiModel
from pocket_tts.modules.seanet import SEANetEncoder, SEANetDecoder
from pocket_tts.modules import mimi_transformer
from pocket_tts.modules.dummy_quantizer import DummyQuantizer
from pocket_tts.utils.config import load_config
from pocket_tts.utils.weights_loading import get_mimi_state_dict

# Paths
audio_path = r"d:\tools\kyutai\default_new.wav"
config_path = r"d:\tools\kyutai\pocket-tts\pocket_tts\config\english_v2.yaml"
# Using the model.safetensors from onnx-export
weights_path = r"d:\tools\kyutai\pocket-tts-onnx-export\models\english_v2\model.safetensors"
onnx_path = r"d:\tools\kyutai\pocket-tts-onnx-export\models\english_v2\mimi_encoder.onnx"

def compare_encoder():
    print(f"Loading config from {config_path}...")
    config = load_config(config_path)
    mimi_config = config.mimi.model_dump()
    
    print(f"Initializing Mimi model components...")
    encoder = SEANetEncoder(**mimi_config["seanet"])
    decoder = SEANetDecoder(**mimi_config["seanet"])
    encoder_transformer = mimi_transformer.ProjectedTransformer(**mimi_config["transformer"])
    decoder_transformer = mimi_transformer.ProjectedTransformer(**mimi_config["transformer"])
    quantizer = DummyQuantizer(**mimi_config["quantizer"])
    
    print(f"Assembling MimiModel...")
    mimi = MimiModel(
        encoder,
        decoder,
        quantizer,
        channels=mimi_config["channels"],
        sample_rate=mimi_config["sample_rate"],
        frame_rate=mimi_config["frame_rate"],
        encoder_frame_rate=mimi_config["sample_rate"] / encoder.hop_length,
        inner_dim=mimi_config["inner_dim"],
        outer_dim=mimi_config["outer_dim"],
        encoder_transformer=encoder_transformer,
        decoder_transformer=decoder_transformer,
    )
    
    print(f"Loading Mimi weights from {weights_path}...")
    mimi_state_dict = get_mimi_state_dict(Path(weights_path))
    mimi.load_state_dict(mimi_state_dict, strict=True)
    mimi.eval()
    mimi.to(device="cpu")
    
    print(f"Loading speaker_proj_weight...")
    sd = safetensors.torch.load_file(weights_path)
    speaker_proj_weight = None
    if 'flow_lm.speaker_proj_weight' in sd:
        speaker_proj_weight = sd['flow_lm.speaker_proj_weight']
    elif 'speaker_proj_weight' in sd:
        speaker_proj_weight = sd['speaker_proj_weight']
            
    if speaker_proj_weight is None:
        print("Warning: speaker_proj_weight not found! Using identity.")
    else:
        print(f"Loaded speaker_proj_weight with shape {speaker_proj_weight.shape}")

    print(f"Loading audio from {audio_path}...")
    audio, sr = sf.read(audio_path)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1) # Mono
    
    if sr != 24000:
        print(f"Resampling from {sr} to 24000...")
        audio = resample_poly(audio, 24000, sr)
    
    # Take 2 seconds to get more frames
    if audio.shape[0] > 48000:
        audio = audio[:48000]
    
    audio_pt = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0)
    
    print("Running PyTorch reference...")
    with torch.no_grad():
        encoded = mimi.encode_to_latent(audio_pt)
        latents = encoded.transpose(-1, -2) # [B, T, 32]
        if speaker_proj_weight is not None:
            pt_out = F.linear(latents, speaker_proj_weight)
        else:
            pt_out = latents
            
    print(f"Running ONNX model from {onnx_path}...")
    ort_sess = ort.InferenceSession(onnx_path)
    onnx_out = ort_sess.run(None, {"audio": audio_pt.numpy()})[0]
    
    print("\n--- Results ---")
    print(f"PyTorch shape: {pt_out.shape}")
    print(f"ONNX shape:    {onnx_out.shape}")
    
    diff = np.abs(pt_out.numpy() - onnx_out)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"Max difference:  {max_diff:.8f}")
    print(f"Mean difference: {mean_diff:.8f}")
    
    if max_diff < 1e-3:
        print("\nSUCCESS: Mimi Encoder results match!")
    else:
        print("\nFAILURE: Significant discrepancy found!")
        print("\nSample (first 5 dims of first frame):")
        print(f"PT:   {pt_out[0, 0, :5].tolist()}")
        print(f"ONNX: {onnx_out[0, 0, :5].tolist()}")

if __name__ == "__main__":
    compare_encoder()
