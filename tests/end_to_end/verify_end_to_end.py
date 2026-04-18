import argparse
import os
import torch
import numpy as np
import onnxruntime as ort
import soundfile as sf
from scipy.signal import resample_poly
import sys
import importlib.util
from pathlib import Path

# Paths
sys.path.append(str(Path(__file__).parent.parent.parent)) # Root for onnx_export
from pocket_tts.models.tts_model import TTSModel
from pocket_tts.modules.stateful_module import init_states
from onnx_export.export_utils import get_state_structure, flatten_state

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

def assert_allclose_with_logging(name, expected, actual, rtol=1e-3, atol=1e-3):
    try:
        # For caches, we want to ignore NaNs when comparing, but ALSO check that NaNs match
        mask_expected = np.isnan(expected)
        mask_actual = np.isnan(actual)
        
        if not np.array_equal(mask_expected, mask_actual):
            print(f"===========================================================")
            print(f"[FAIL] {name} NaN mask mismatch!")
            # Find where they differ
            diff_mask = mask_expected ^ mask_actual
            count = np.sum(diff_mask)
            print(f"Number of differing positions in NaN mask: {count}")
            # If the difference is in the USED part (where expected is NOT nan), it's a bug.
            # If it's in the UNUSED part, it might be an ONNX optimization.
            print(f"===========================================================")
            # Continue to check non-nan values if possible
        
        # Compare only non-nan values
        valid_mask = ~mask_expected
        if np.any(valid_mask):
            np.testing.assert_allclose(expected[valid_mask], actual[valid_mask], rtol=rtol, atol=atol)
            
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
    if ort_type_name == "FLOAT": return tensor.astype(np.float32)
    if ort_type_name == "INT64": return tensor.astype(np.int64)
    return tensor

def get_session_input_types(session):
    res = {}
    for i in session.get_inputs():
        t = i.type
        if "float" in t: res[i.name] = "FLOAT"
        elif "int64" in t: res[i.name] = "INT64"
        else: res[i.name] = "UNKNOWN"
    return res

def lsd_decode_onnx(ort_flow, c, x_0, num_steps=10):
    dt = 1.0 / num_steps
    x_t = x_0
    for i in range(num_steps):
        s = i / num_steps
        t = s + dt
        
        # Inputs: c(B, 1024), s(B, 1), t(B, 1), x(B, 32)
        flow_inputs = {
            "c": c,
            "s": np.array([[s]], dtype=np.float32),
            "t": np.array([[t]], dtype=np.float32),
            "x": x_t
        }
        flow_dir = ort_flow.run(None, flow_inputs)[0]
        if i == 0:
            print(f"DEBUG: ONNX First flow_dir (Step {i+1}) sample: {flow_dir[0, :5]}")
        x_t = x_t + flow_dir * dt
    return x_t

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--onnx_dir", type=str, required=True)
    parser.add_argument("--audio", type=str, required=True)
    parser.add_argument("--text", type=str, default="Hello")
    args = parser.parse_args()

    # 1. Load Models
    print(f"Loading models from {args.onnx_dir}...")
    tts_model = TTSModel.load_model(config=args.config)
    tts_model.eval()

    ort_encoder = ort.InferenceSession(os.path.join(args.onnx_dir, "mimi_encoder.onnx"))
    ort_conditioner = ort.InferenceSession(os.path.join(args.onnx_dir, "text_conditioner.onnx"))
    ort_main = ort.InferenceSession(os.path.join(args.onnx_dir, "flow_lm_main.onnx"))
    ort_flow = ort.InferenceSession(os.path.join(args.onnx_dir, "flow_lm_flow.onnx"))

    # Load export wrapper for PT reference
    export_script_path = Path(__file__).parent.parent.parent / "scripts" / "export_flow_lm.py"
    spec = importlib.util.spec_from_file_location("export_flow_lm", str(export_script_path))
    export_flow_lm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(export_flow_lm)

    # 2. Audio -> Voice Prompt
    audio, sr = sf.read(args.audio)
    if len(audio.shape) > 1: audio = audio.mean(axis=1)
    if sr != 24000: audio = resample_poly(audio, 24000, sr)
    audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0)

    # 3. Step-by-Step Validation
    print("\n--- PHASE 1: Voice Conditioning ---")
    with torch.no_grad():
        encoded = tts_model.mimi.encode_to_latent(audio_tensor)
        latents = encoded.transpose(-1, -2).to(torch.float32)
        pt_voice_latents = torch.nn.functional.linear(latents, tts_model.flow_lm.speaker_proj_weight)
        
        # In the new version, the wrapper handles BOS automatically
        pt_voice_prompt = pt_voice_latents
        
        # State init (1000 frames)
        flow_state = init_states(tts_model.flow_lm, batch_size=1, sequence_length=1000)
        pt_main_wrapper = export_flow_lm.FlowLMMainWrapper(tts_model.flow_lm, get_state_structure(flow_state))
        empty_seq = torch.zeros((1, 0, tts_model.flow_lm.ldim))
        
        # PT Voice Prompt
        pt_outputs = pt_main_wrapper(empty_seq, pt_voice_prompt, flatten_state(flow_state))
        pt_voice_c = pt_outputs[0]
        
        # Update flow_state
        idx = 2
        for module_name, module_state in flow_state.items():
             for k in sorted(module_state.keys()):
                 module_state[k] = pt_outputs[idx]
                 idx += 1

    # Auto-BOS Injection is now handled internally by the ONNX model
    onnx_voice_latents = ort_encoder.run(None, {"audio": audio_tensor.numpy()})[0]
    onnx_voice_prompt = onnx_voice_latents

    input_types = get_session_input_types(ort_main)
    ort_inputs = {"sequence": empty_seq.numpy(), "text_embeddings": onnx_voice_prompt}
    for i, st in enumerate(flatten_state(init_states(tts_model.flow_lm, 1, 1000))):
        ort_inputs[f"state_{i}"] = cast_to_ort_type(st.numpy(), input_types.get(f"state_{i}", "FLOAT"))
    
    onnx_outputs = ort_main.run(None, ort_inputs)
    assert_allclose_with_logging("Voice Prompt Parity (c)", pt_voice_c.numpy(), onnx_outputs[0], rtol=1e-5, atol=1e-5)
    
    # Check states after Voice
    for i in range(len(onnx_outputs) - 2):
        assert_allclose_with_logging(f"State {i} after Voice", pt_outputs[i+2].numpy(), onnx_outputs[i+2], rtol=1e-5, atol=1e-5)

    # Update ONNX state
    onnx_state = {f"state_{i}": onnx_outputs[i+2] for i in range(len(onnx_outputs) - 2)}

    print("\n--- PHASE 2: Text Conditioning ---")
    from pocket_tts.conditioners.base import TokenizedText
    tokens = tts_model.flow_lm.conditioner.tokenizer(args.text)
    token_ids = tokens.tokens
    
    with torch.no_grad():
        pt_text_emb = tts_model.flow_lm.conditioner(tokens)
        pt_outputs_text = pt_main_wrapper(empty_seq, pt_text_emb, flatten_state(flow_state))
        pt_text_c = pt_outputs_text[0]
        
        # Update flow_state
        idx = 2
        for module_name, module_state in flow_state.items():
             for k in sorted(module_state.keys()):
                 module_state[k] = pt_outputs_text[idx]
                 idx += 1

    onnx_text_emb = ort_conditioner.run(None, {"token_ids": token_ids.numpy().astype(np.int64)})[0]
    ort_inputs_text = {"sequence": empty_seq.numpy(), "text_embeddings": onnx_text_emb, **onnx_state}
    for name in ort_inputs_text:
        if name in input_types: ort_inputs_text[name] = cast_to_ort_type(ort_inputs_text[name], input_types[name])
    
    onnx_outputs_text = ort_main.run(None, ort_inputs_text)
    assert_allclose_with_logging("Text Prompt Parity (c)", pt_text_c.numpy(), onnx_outputs_text[0], rtol=1e-5, atol=1e-5)

    # Check states after Text
    for i in range(len(onnx_outputs_text) - 2):
        assert_allclose_with_logging(f"State {i} after Text", pt_outputs_text[i+2].numpy(), onnx_outputs_text[i+2], rtol=1e-5, atol=1e-5)

    # Update ONNX state
    onnx_state = {f"state_{i}": onnx_outputs_text[i+2] for i in range(len(onnx_outputs_text) - 2)}


    print("\n--- PHASE 3: AR Generation (5 steps) ---")
    # Shared fixed noise
    noise_fixed = torch.randn((1, 32))
    
    # Starting latent: [NaN] * 32
    pt_current_latent = torch.full((1, 1, 32), float("NaN"))
    onnx_current_latent = np.full((1, 1, 32), np.nan, dtype=np.float32)
    
    def pt_v_t(c_val, s, t, x):
        return tts_model.flow_lm.flow_net(c_val, s, t, x)

    from pocket_tts.models.flow_lm import lsd_decode as pt_lsd_decode

    for step_idx in range(5):
        print(f"\nStep {step_idx + 1}...")
        
        # 1. Flow LM Main (Conditioning)
        with torch.no_grad():
            # TTSModel uses an empty text_embeddings tensor for AR steps
            empty_text_emb_pt = torch.zeros((1, 0, 1024))
            pt_ar_outputs = pt_main_wrapper(pt_current_latent, empty_text_emb_pt, flatten_state(flow_state))
            pt_c = pt_ar_outputs[0]
            pt_eos = pt_ar_outputs[1]
            
            # 2. LSD Decode
            # Note: lsd_decode_steps=10
            v_t = lambda s, t, x: pt_v_t(pt_c, s, t, x)
            
            # Manual check for the first flow dir
            first_s = torch.tensor([[0.0]])
            first_t = torch.tensor([[0.1]]) # dt = 0.1 for 10 steps
            pt_first_v = pt_v_t(pt_c, first_s, first_t, noise_fixed)
            print(f"DEBUG: PT First flow_dir (Step 1) sample: {pt_first_v[0, :5].numpy()}")

            pt_next_latent = pt_lsd_decode(v_t, noise_fixed, num_steps=10).unsqueeze(1)
            
            # Update PT state
            idx = 2
            for module_name, module_state in flow_state.items():
                 for k in sorted(module_state.keys()):
                     module_state[k] = pt_ar_outputs[idx]
                     idx += 1

        # ONNX AR
        empty_text_emb_onnx = np.zeros((1, 0, 1024), dtype=np.float32)
        ort_inputs_ar = {"sequence": onnx_current_latent, "text_embeddings": empty_text_emb_onnx, **onnx_state}
        for name in ort_inputs_ar:
            if name in input_types: ort_inputs_ar[name] = cast_to_ort_type(ort_inputs_ar[name], input_types[name])
        
        onnx_ar_outputs = ort_main.run(None, ort_inputs_ar)
        onnx_c = onnx_ar_outputs[0]
        onnx_eos = onnx_ar_outputs[1]
        
        assert_allclose_with_logging(f"Step {step_idx+1} Conditioning (c)", pt_c.numpy(), onnx_c, rtol=1e-5, atol=1e-5)
        assert_allclose_with_logging(f"Step {step_idx+1} EOS logit", pt_eos.numpy(), onnx_eos, rtol=1e-5, atol=1e-5)
        
        # Check states after this step
        for i in range(len(onnx_ar_outputs) - 2):
            assert_allclose_with_logging(f"State {i} after AR Step {step_idx+1}", pt_ar_outputs[i+2].numpy(), onnx_ar_outputs[i+2], rtol=1e-5, atol=1e-5)

        # ONNX LSD Loop
        onnx_next_latent_flat = lsd_decode_onnx(ort_flow, onnx_c, noise_fixed.numpy(), num_steps=10)
        onnx_next_latent = onnx_next_latent_flat.reshape(1, 1, 32)
        
        assert_allclose_with_logging(f"Step {step_idx+1} Generated Latent", pt_next_latent.numpy(), onnx_next_latent)
        
        # Update ONNX state
        onnx_state = {f"state_{i}": onnx_ar_outputs[i+2] for i in range(len(onnx_ar_outputs) - 2)}
        
        # Next loop inputs
        pt_current_latent = pt_next_latent
        onnx_current_latent = onnx_next_latent

    print("\nEnd-to-End Parity Check Complete.")

if __name__ == "__main__":
    main()
