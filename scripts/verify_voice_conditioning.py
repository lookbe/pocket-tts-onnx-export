import argparse
import os
import torch
import numpy as np
import onnxruntime as ort
import soundfile as sf
from scipy.signal import resample_poly

from pocket_tts.models.tts_model import TTSModel
from pocket_tts.modules.stateful_module import init_states
from onnx_export.export_utils import get_state_structure, flatten_state
from onnx_export.wrappers import MimiEncoderWrapper

def assert_allclose_with_logging(name, expected, actual, rtol=1e-3, atol=1e-3):
    try:
        np.testing.assert_allclose(expected, actual, rtol=rtol, atol=atol)
        print(f"[PASS] {name} matches! (rtol={rtol}, atol={atol})")
    except AssertionError as e:
        print(f"===========================================================")
        print(f"[FAIL] {name} mismatch!")
        diff = np.abs(expected - actual)
        print(f"Max diff: {np.max(diff)}")
        print(f"Mean diff: {np.mean(diff)}")
        print(f"===========================================================")
        return False
    return True

def cast_to_ort_type(tensor, ort_type_name):
    if ort_type_name == "FLOAT":
        return tensor.astype(np.float32)
    if ort_type_name == "INT64":
        return tensor.astype(np.int64)
    return tensor

def get_session_input_types(session):
    res = {}
    for i in session.get_inputs():
        t = i.type
        if "float" in t: res[i.name] = "FLOAT"
        elif "int64" in t: res[i.name] = "INT64"
        else: res[i.name] = "UNKNOWN"
    return res

# ==============================================================================
# MONKEYPATCHES for EXACT ONNX PARITY
# ==============================================================================
import pocket_tts.modules.transformer as transformer_module
from pocket_tts.modules.transformer import StreamingMultiheadAttention
import torch.nn.functional as F

def patched_init_state(self, batch_size: int, sequence_length: int) -> dict[str, torch.Tensor]:
    device = self.in_proj.weight.device
    dtype = self.in_proj.weight.dtype
    return dict(
        cache=torch.full(
            (2, batch_size, sequence_length, self.num_heads, self.dim_per_head),
            float("NaN"),
            device=device,
            dtype=dtype,
        ),
        current_end=torch.zeros(0, device=device, dtype=dtype),
        step=torch.zeros(batch_size, dtype=torch.long, device=device),
    )

def patched_increment_step(self, state: dict, increment: int = 1):
    state["step"] = state["step"] + increment

def patched_append_and_get(self, k, v, state):
    if state is None:
        k_attn = k.permute(0, 2, 1, 3); v_attn = v.permute(0, 2, 1, 3)
        pos_k = torch.arange(k_attn.shape[2], device=k_attn.device, dtype=torch.long).view(1, -1).expand(k_attn.shape[0], -1)
        step = torch.zeros(k_attn.shape[0], device=k_attn.device, dtype=torch.long)
        return k_attn, v_attn, pos_k, step

    cache = state["cache"]
    step = state["step"]
    off = step.view(-1)[0]
    
    new_cache = cache.clone()
    new_cache[0, :, off : off + k.shape[1]] = k
    new_cache[1, :, off : off + v.shape[1]] = v
    state["cache"] = new_cache
    
    valid_len = off + k.shape[1]
    k_attn = new_cache[0, :, :valid_len].permute(0, 2, 1, 3)
    v_attn = new_cache[1, :, :valid_len].permute(0, 2, 1, 3)
    
    MAX_POS = 4096
    pos_k = torch.arange(MAX_POS, device=k_attn.device, dtype=torch.long)[:valid_len].unsqueeze(0).expand(k_attn.shape[0], -1)
    return k_attn, v_attn, pos_k, step

def patched_sma_forward(self, query: torch.Tensor, model_state: dict | None):
    state = None if model_state is None else self.get_state(model_state)
    projected = self.in_proj(query)
    b, t, _ = projected.shape
    d = self.dim_per_head
    packed = projected.view(b, t, 3, self.num_heads, d)
    q, k, v = torch.unbind(packed, dim=2)
    
    if state is None:
        rope_offset = torch.zeros((), dtype=torch.long, device=q.device)
    else:
        rope_offset = state["step"].view(-1)[0]
        
    q, k = self.rope(q, k, offset=rope_offset)
    q = q.transpose(1, 2)
    k_attn, v_attn, pos_k, step = self._cache_backend.append_and_get(k, v, state)
    MAX_POS = 4096
    pos_q = step.view(-1, 1) + torch.arange(MAX_POS, device=q.device, dtype=torch.long)[:t].unsqueeze(0)
    from pocket_tts.modules.transformer import _build_attention_mask
    attn_mask = _build_attention_mask(pos_q, pos_k, self.context)
    x = F.scaled_dot_product_attention(q, k_attn, v_attn, attn_mask, dropout_p=0.0)
    x = x.transpose(1, 2)
    b, t, h, d = x.shape
    x = x.reshape(b, t, h * d)
    x = self.out_proj(x)
    return x

transformer_module._LinearKVCacheBackend.append_and_get = patched_append_and_get
StreamingMultiheadAttention.init_state = patched_init_state
StreamingMultiheadAttention.increment_step = patched_increment_step
StreamingMultiheadAttention.forward = patched_sma_forward

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--onnx_dir", type=str, required=True, help="Path to directory containing ONNX files")
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file for voice conditioning")
    parser.add_argument("--weights_path", type=str, help="Path to safetensors weights")
    args = parser.parse_args()

    # 1. Load PyTorch Model
    print(f"Loading PyTorch model from {args.config}...")
    tts_model = TTSModel.load_model(config=args.config)
    
    if args.weights_path and os.path.exists(args.weights_path):
        import safetensors.torch
        print(f"Loading weights from {args.weights_path}...")
        state_dict = safetensors.torch.load_file(args.weights_path)
        tts_model.load_state_dict(state_dict, strict=False)
    
    tts_model.eval()

    # 2. Load Audio
    print(f"Loading and resampling audio from {args.audio}...")
    audio, sr = sf.read(args.audio)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    if sr != 24000:
        audio = resample_poly(audio, 24000, sr)
    
    # Truncate if too long (e.g. 5 seconds for test)
    if audio.shape[0] > 24000 * 5:
        audio = audio[:24000 * 5]
    
    audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0)

    # 3. PyTorch Reference: Produce Prompt Embedding
    with torch.no_grad():
        # This matches TTSModel._encode_audio + BOS logic
        encoded = tts_model.mimi.encode_to_latent(audio_tensor)
        latents = encoded.transpose(-1, -2).to(torch.float32)
        pt_prompt = torch.nn.functional.linear(latents, tts_model.flow_lm.speaker_proj_weight)
        
        if tts_model.flow_lm.insert_bos_before_voice:
            print("Detected English v2 style: Prepending BOS before voice.")
            pt_prompt = torch.cat([tts_model.flow_lm.bos_before_voice, pt_prompt], dim=1)

    # 4. ONNX Reference: Produce Prompt Embedding
    encoder_path = os.path.join(args.onnx_dir, "mimi_encoder.onnx")
    if os.path.exists(encoder_path):
        print(f"Running ONNX Mimi Encoder: {encoder_path}")
        ort_encoder = ort.InferenceSession(encoder_path)
        onnx_prompt = ort_encoder.run(None, {"audio": audio_tensor.numpy()})[0]
        
        # In ONNX, the BOS usually needs to be handled manually or it might be in text_conditioner.
        # However, the user wants to check Flow LM conditioning.
        # If the ONNX Mimi Encoder ONLY does the audio part, we must prepend BOS here if testing main LM.
        
        # Verify Encoder Parity First
        ref_encoder_out = pt_prompt
        if tts_model.flow_lm.insert_bos_before_voice:
             # Remove BOS for pure encoder check if pt_prompt has it
             ref_encoder_out = pt_prompt[:, 1:]
             
        assert_allclose_with_logging("Mimi Encoder + Projection Parity", ref_encoder_out.numpy(), onnx_prompt)

    # 5. Compare Flow LM Main Conditioning Step
    flow_lm_path = os.path.join(args.onnx_dir, "flow_lm_main.onnx")
    if os.path.exists(flow_lm_path):
        print(f"\nComparing Flow LM Main conditioning with prompt (len={pt_prompt.shape[1]})...")
        ort_flow = ort.InferenceSession(flow_lm_path)
        
        # Initialize states - MUST MATCH ONNX FIXED SHAPE (1000)
        flow_state = init_states(tts_model.flow_lm, batch_size=1, sequence_length=1000)
        flat_flow_state = flatten_state(flow_state)
        
        # Conditioning input: empty sequence (seq_len=0)
        test_seq = torch.zeros((1, 0, tts_model.flow_lm.ldim))
        
        # PyTorch flow LM run (prompting)
        with torch.no_grad():
            import sys
            import importlib.util
            from pathlib import Path
            script_dir = Path(__file__).parent
            export_script_path = script_dir / "export_flow_lm.py"
            spec = importlib.util.spec_from_file_location("export_flow_lm", str(export_script_path))
            export_flow_lm = importlib.util.module_from_spec(spec)
            sys.modules["export_flow_lm"] = export_flow_lm
            spec.loader.exec_module(export_flow_lm)
            
            pt_wrapper = export_flow_lm.FlowLMMainWrapper(tts_model.flow_lm, get_state_structure(flow_state))
            pt_outputs = pt_wrapper(test_seq, pt_prompt, flat_flow_state)
            
        pt_c = pt_outputs[0].numpy()
        pt_states = [s.numpy() for s in pt_outputs[2:]]
        
        # ONNX flow LM run
        input_types = get_session_input_types(ort_flow)
        ort_inputs = {
            "sequence": test_seq.numpy(),
            "text_embeddings": pt_prompt.numpy() # Use PT prompt to isolate FlowLM check
        }
        for name in ort_inputs:
            if name in input_types:
                ort_inputs[name] = cast_to_ort_type(ort_inputs[name], input_types[name])
        
        for i, st in enumerate(flat_flow_state):
            name = f"state_{i}"
            val = st.numpy()
            if name in input_types:
                val = cast_to_ort_type(val, input_types[name])
            ort_inputs[name] = val
            
        onnx_outputs = ort_flow.run(None, ort_inputs)
        onnx_c = onnx_outputs[0]
        onnx_states = onnx_outputs[2:]
        
        assert_allclose_with_logging("FlowLM Conditioning (c) output", pt_c, onnx_c)
        
        all_states_pass = True
        for i, (p_s, o_s) in enumerate(zip(pt_states, onnx_states)):
            if not assert_allclose_with_logging(f"FlowLM State {i}", p_s, o_s, atol=1e-3):
                all_states_pass = False
        
        if all_states_pass:
            print("[PASS] All FlowLM States match after voice conditioning!")

if __name__ == "__main__":
    main()
