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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--onnx_dir", type=str, required=True, help="Path to directory containing ONNX files")
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file for voice conditioning")
    parser.add_argument("--text", type=str, default="Hello", help="Text for text conditioning")
    parser.add_argument("--weights_path", type=str, help="Path to safetensors weights")
    args = parser.parse_args()

    # 1. Load PyTorch Model
    print(f"Loading PyTorch model from {args.config}...")
    tts_model = TTSModel.load_model(config=args.config)
    tts_model.eval()

    # Load ONNX Models
    encoder_path = os.path.join(args.onnx_dir, "mimi_encoder.onnx")
    conditioner_path = os.path.join(args.onnx_dir, "text_conditioner.onnx")
    flow_lm_path = os.path.join(args.onnx_dir, "flow_lm_main.onnx")

    ort_encoder = ort.InferenceSession(encoder_path)
    ort_conditioner = ort.InferenceSession(conditioner_path)
    ort_flow = ort.InferenceSession(flow_lm_path)

    # Load and resample audio
    audio, sr = sf.read(args.audio)
    if len(audio.shape) > 1: audio = audio.mean(axis=1)
    if sr != 24000: audio = resample_poly(audio, 24000, sr)
    if audio.shape[0] > 24000 * 5: audio = audio[:24000 * 5]
    audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0)

    # --------------------------------------------------------------------------
    # STEP 1: VOICE CONDITIONING
    # --------------------------------------------------------------------------
    print("\n--- Phase 1: Voice Conditioning ---")
    with torch.no_grad():
        encoded = tts_model.mimi.encode_to_latent(audio_tensor)
        latents = encoded.transpose(-1, -2).to(torch.float32)
        pt_voice_latents = torch.nn.functional.linear(latents, tts_model.flow_lm.speaker_proj_weight)
        
        # PT logic for voice conditioning (BOS prepended)
        pt_voice_prompt = torch.cat([tts_model.flow_lm.bos_before_voice, pt_voice_latents], dim=1)
        
        # Init state
        flow_state = init_states(tts_model.flow_lm, batch_size=1, sequence_length=1000)
        
        # Run PT Voice Conditioning
        import sys
        import importlib.util
        from pathlib import Path
        script_dir = Path(__file__).parent
        export_script_path = script_dir / "export_flow_lm.py"
        spec = importlib.util.spec_from_file_location("export_flow_lm", str(export_script_path))
        export_flow_lm = importlib.util.module_from_spec(spec)
        sys.modules["export_flow_lm"] = export_flow_lm
        spec.loader.exec_module(export_flow_lm)
        
        pt_main_wrapper = export_flow_lm.FlowLMMainWrapper(tts_model.flow_lm, get_state_structure(flow_state))
        empty_seq = torch.zeros((1, 0, tts_model.flow_lm.ldim))
        
        # PT Reference Step
        # NOTE: We now manually pass pt_voice_prompt (which has BOS) to the wrapper
        pt_outputs = pt_main_wrapper(empty_seq, pt_voice_prompt, flatten_state(flow_state))
        pt_voice_c = pt_outputs[0]
        # PT states are updated implicitly via references in the wrapper logic if not careful,
        # but here we should manually update our flow_state if we want to continue.
        # FlowLMMainWrapper updates states internally in pt_main_wrapper since it passed model_state
        
        # Update flow_state for continuation
        idx = 2
        for module_name, module_state in flow_state.items():
             for k in sorted(module_state.keys()):
                 module_state[k] = pt_outputs[idx]
                 idx += 1

    # ONNX Voice Conditioning Parity check
    onnx_voice_latents = ort.InferenceSession(encoder_path).run(None, {"audio": audio_tensor.numpy()})[0]
    
    # Simulating the web worker: manually prepend BOS if model requires it
    if getattr(tts_model.flow_lm, "insert_bos_before_voice", False):
        print("BOS FIX: Manually prepending BOS to voice latents for ONNX...")
        bos_emb = tts_model.flow_lm.bos_before_voice.detach().cpu().numpy()
        onnx_voice_promo_input = np.concatenate([bos_emb, onnx_voice_latents], axis=1)
    else:
        onnx_voice_promo_input = onnx_voice_latents

    input_types = get_session_input_types(ort_flow)
    ort_inputs = {
        "sequence": empty_seq.numpy(),
        "text_embeddings": onnx_voice_promo_input,
    }
    for i, st in enumerate(flatten_state(init_states(tts_model.flow_lm, 1, 1000))):
        ort_inputs[f"state_{i}"] = cast_to_ort_type(st.numpy(), input_types.get(f"state_{i}", "FLOAT"))
    
    onnx_outputs = ort_flow.run(None, ort_inputs)
    assert_allclose_with_logging("Voice Conditioning (c)", pt_voice_c.numpy(), onnx_outputs[0])
    
    # Update ONNX state for phase 2
    onnx_state = {}
    for i in range(len(onnx_outputs) - 2):
        onnx_state[f"state_{i}"] = onnx_outputs[i+2]

    # --------------------------------------------------------------------------
    # STEP 2: TEXT CONDITIONING
    # --------------------------------------------------------------------------
    print(f"\n--- Phase 2: Text Conditioning ('{args.text}') ---")
    
    # Tokenize
    from pocket_tts.conditioners.base import TokenizedText
    tokens = tts_model.flow_lm.conditioner.tokenizer(args.text)
    token_ids = tokens.tokens
    
    # PT Text Embedding
    with torch.no_grad():
        pt_text_emb = tts_model.flow_lm.conditioner(tokens)
    
    # ONNX Text Embedding
    onnx_text_emb = ort_conditioner.run(None, {"token_ids": token_ids.numpy().astype(np.int64)})[0]
    assert_allclose_with_logging("Text Embedding Parity", pt_text_emb.numpy(), onnx_text_emb)

    # Run PT Flow LM Text conditioning step
    with torch.no_grad():
        # Important: audio_conditioning is empty here. text_embeddings is just the text.
        pt_outputs_text = pt_main_wrapper(empty_seq, pt_text_emb, flatten_state(flow_state))
        pt_text_c = pt_outputs_text[0]
        
    # --------------------------------------------------------------------------
    # RAW PYTORCH COMPARISON (Check for BOS discrepancy)
    # --------------------------------------------------------------------------
    print("\n--- Comparing Wrapper against Raw PT FlowLM (Internal Check) ---")
    with torch.no_grad():
        # Raw FlowLM backbone call
        # In TTSModel, text conditioning is cat'ed before backbone.
        # Here we simulate the second step (Text Step)
        raw_pt_c = tts_model.flow_lm.transformer(pt_text_emb, flow_state)
        if tts_model.flow_lm.out_norm:
            raw_pt_c = tts_model.flow_lm.out_norm(raw_pt_c)
        raw_pt_c = raw_pt_c[:, -1] # Last token
        
    print(f"Raw PT Conditioning (c) shape: {raw_pt_c.shape}")
    print(f"Wrapper PT Conditioning (c) shape: {pt_text_c.shape}")
    
    # This will likely FAIL if the wrapper prepends BOS but raw PT doesn't.
    assert_allclose_with_logging("Raw PT vs Wrapper (BOS check)", raw_pt_c.numpy(), pt_text_c.numpy())

    # Run ONNX Flow LM Text conditioning step
    ort_inputs_text = {
        "sequence": empty_seq.numpy(),
        "text_embeddings": onnx_text_emb
    }
    for i in range(len(onnx_state)):
        ort_inputs_text[f"state_{i}"] = onnx_state[f"state_{i}"]
    
    # Apply type casting
    for name in ort_inputs_text:
        if name in input_types:
            ort_inputs_text[name] = cast_to_ort_type(ort_inputs_text[name], input_types[name])

    print("Running ONNX Flow LM for Text step...")
    onnx_outputs_text = ort_flow.run(None, ort_inputs_text)
    
    # Check parity
    assert_allclose_with_logging("Text Conditioning (c) result", pt_text_c.numpy(), onnx_outputs_text[0])
    
    all_states_pass = True
    for i in range(len(onnx_outputs_text) - 2):
        if not assert_allclose_with_logging(f"State {i} after Text", pt_outputs_text[i+2].numpy(), onnx_outputs_text[i+2]):
            all_states_pass = False
            
    if all_states_pass:
        print("[PASS] Sequential conditioning parity confirmed!")
    else:
        print("[FAIL] Sequential conditioning state mismatch.")

if __name__ == "__main__":
    main()
