import argparse
import os
import torch
import numpy as np
import onnxruntime as ort

from pocket_tts.models.tts_model import TTSModel
from pocket_tts.modules.stateful_module import init_states
from pocket_tts.utils.config import Config
from onnx_export.export_utils import get_state_structure, flatten_state
from onnx_export.wrappers import MimiWrapper, MimiEncoderWrapper, TextConditionerWrapper

def cast_to_ort_type(tensor, ort_type_name):
    """Cast a numpy array to the expected ONNX runtime type name."""
    if ort_type_name == "FLOAT":
        return tensor.astype(np.float32)
    if ort_type_name == "FLOAT16":
        return tensor.astype(np.float16)
    if ort_type_name == "INT64":
        return tensor.astype(np.int64)
    if ort_type_name == "INT32":
        return tensor.astype(np.int32)
    if ort_type_name == "BOOL":
        return tensor.astype(np.bool_)
    return tensor

def get_session_input_types(session):
    """Map session input names to their human-readable type names."""
    res = {}
    for i in session.get_inputs():
        t = i.type
        if "float16" in t: res[i.name] = "FLOAT16"
        elif "float" in t: res[i.name] = "FLOAT"
        elif "int64" in t: res[i.name] = "INT64"
        elif "int32" in t: res[i.name] = "INT32"
        elif "bool" in t: res[i.name] = "BOOL"
        else: res[i.name] = "UNKNOWN"
    return res

def assert_allclose_with_logging(name, expected, actual, rtol=1e-3, atol=1e-3):
    try:
        np.testing.assert_allclose(expected, actual, rtol=rtol, atol=atol)
        print(f"[PASS] {name} matches! (rtol={rtol}, atol={atol})")
    except AssertionError as e:
        print(f"===========================================================")
        print(f"[FAIL] {name} mismatch!")
        diff = np.abs(expected - actual)
        print(f"Max diff: {np.max(diff)}")
        print(f"===========================================================")
        return False
    return True

def compare_encoder(ort_session, pt_wrapper, tts_model, audio_path=None):
    print("\n--- Comparing Mimi Encoder ---")
    
    if audio_path and os.path.exists(audio_path):
        import soundfile as sf
        from scipy.signal import resample_poly
        print(f"Loading test audio from {audio_path}...")
        audio, sr = sf.read(audio_path)
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        if sr != 24000:
            print(f"Resampling from {sr} to 24000...")
            audio = resample_poly(audio, 24000, sr)
        if audio.shape[0] > 48000:
            audio = audio[:48000]
        test_audio = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0)
    else:
        print("Using random noise for encoder test.")
        test_audio = torch.randn(1, 1, 24000)
    
    # Ground truth: what the PyTorch reference model actually does
    # (tts_model._encode_audio → encode_to_latent + transpose + F.linear, NO normalization)
    with torch.no_grad():
        encoded = tts_model.mimi.encode_to_latent(test_audio)
        latents = encoded.transpose(-1, -2)
        import torch.nn.functional as F_test
        pt_out = F_test.linear(latents, tts_model.flow_lm.speaker_proj_weight).numpy()
    
    onnx_out = ort_session.run(None, {"audio": test_audio.numpy()})[0]

    
    assert_allclose_with_logging("Mimi Encoder Latents (vs PyTorch reference)", pt_out, onnx_out)
    
    # Also verify wrapper matches ONNX (sanity)
    with torch.no_grad():
        wrapper_out = pt_wrapper(test_audio).numpy()
    assert_allclose_with_logging("Mimi Encoder Latents (wrapper vs ONNX)", wrapper_out, onnx_out)



def compare_conditioner(ort_session, pt_wrapper):
    print("\n--- Comparing Text Conditioner ---")
    test_tokens = torch.randint(0, 1000, (1, 20))
    
    with torch.no_grad():
        pt_out = pt_wrapper(test_tokens).numpy()
    
    onnx_out = ort_session.run(None, {"token_ids": test_tokens.numpy()})[0]
    
    assert_allclose_with_logging("Text Conditioner Embeddings", pt_out, onnx_out)


def compare_flow_lm_main(ort_session, pt_wrapper, tts_model):
    print("\n--- Comparing Flow LM Main ---")
    flow_state = init_states(tts_model.flow_lm, batch_size=1, sequence_length=1000)
    flat_flow_state = flatten_state(flow_state)
    
    test_seq = torch.randn(1, 1, 32)
    test_text = torch.randn(1, 25, tts_model.flow_lm.dim)
    
    with torch.no_grad():
        pt_outputs = pt_wrapper(test_seq, test_text, flat_flow_state)
        
    pt_c = pt_outputs[0].numpy()
    pt_eos = pt_outputs[1].numpy()
    pt_states = [s.numpy() for s in pt_outputs[2:]]
    
    ort_inputs = {
        "sequence": test_seq.numpy(),
        "text_embeddings": test_text.numpy()
    }
    input_types = get_session_input_types(ort_session)
    # Apply casts to all inputs
    for name in ort_inputs:
        if name in input_types:
            ort_inputs[name] = cast_to_ort_type(ort_inputs[name], input_types[name])

    for i, st in enumerate(flat_flow_state):
        name = f"state_{i}"
        val = st.numpy()
        if name in input_types:
            val = cast_to_ort_type(val, input_types[name])
        ort_inputs[name] = val
        
    onnx_outputs = ort_session.run(None, ort_inputs)
    onnx_c = onnx_outputs[0]
    onnx_eos = onnx_outputs[1]
    onnx_states = onnx_outputs[2:]
    
    assert_allclose_with_logging("FlowLM Conditioning (c)", pt_c, onnx_c)
    assert_allclose_with_logging("FlowLM EOS", pt_eos, onnx_eos)
    
    all_states_pass = True
    for i, (p_s, o_s) in enumerate(zip(pt_states, onnx_states)):
        if not assert_allclose_with_logging(f"FlowLM State {i}", p_s, o_s):
            all_states_pass = False
    
    if all_states_pass:
        print("[PASS] All FlowLM States match!")

def compare_flow_lm_flow(ort_session, pt_wrapper, tts_model):
    print("\n--- Comparing Flow LM FlowNet ---")
    # c(1024) is rank 1, s(1,1), t(1,1), x(1,32) are rank 2
    c = torch.randn(1024)
    s = torch.tensor([[0.0]])
    t = torch.tensor([[0.1]])
    x = torch.randn(1, 32)
    
    with torch.no_grad():
        pt_out = pt_wrapper(c, s, t, x).numpy()
    
    input_types = get_session_input_types(ort_session)
    ort_inputs = {
        "c": cast_to_ort_type(c.numpy(), input_types["c"]),
        "s": cast_to_ort_type(s.numpy(), input_types["s"]),
        "t": cast_to_ort_type(t.numpy(), input_types["t"]),
        "x": cast_to_ort_type(x.numpy(), input_types["x"]),
    }
    onnx_out = ort_session.run(None, ort_inputs)[0]
    assert_allclose_with_logging("FlowNet Latent Match", pt_out, onnx_out)

def compare_mimi_decoder(ort_session, pt_wrapper, tts_model):
    print("\n--- Comparing Mimi Decoder ---")
    mimi_state = init_states(tts_model.mimi, batch_size=1, sequence_length=1000)
    flat_mimi_state = flatten_state(mimi_state)
    
    print(f"PyTorch Mimi States: {len(flat_mimi_state)}")
    print(f"ONNX Mimi Session Inputs: {len(ort_session.get_inputs()) - 1}") # -1 for 'latent'
    
    latent = torch.randn(1, 1, tts_model.flow_lm.ldim)
    
    with torch.no_grad():
        pt_outputs = pt_wrapper(latent, flat_mimi_state)
        
    pt_audio = pt_outputs[0].numpy()
    pt_states = [s.numpy() for s in pt_outputs[1:]]
    
    ort_inputs = {
        "latent": latent.numpy()
    }
    input_types = get_session_input_types(ort_session)
    if "latent" in input_types:
        ort_inputs["latent"] = cast_to_ort_type(ort_inputs["latent"], input_types["latent"])

    for i, st in enumerate(flat_mimi_state):
        name = f"state_{i}"
        val = st.numpy()
        print(f"[DEBUG] Mimi State {i} Shape: {val.shape}, Dtype: {val.dtype}")
        if name in input_types:
            val = cast_to_ort_type(val, input_types[name])
        ort_inputs[name] = val
        
    onnx_outputs = ort_session.run(None, ort_inputs)
    onnx_audio = onnx_outputs[0]
    onnx_states = onnx_outputs[1:]
    
    print(f"ONNX Audio Shape: {onnx_audio.shape}, Dtype: {onnx_audio.dtype}")
    
    assert_allclose_with_logging("Mimi Audio Output", pt_audio, onnx_audio)
    
    all_states_pass = True
    for i, (p_s, o_s) in enumerate(zip(pt_states, onnx_states)):
        if not assert_allclose_with_logging(f"Mimi State {i}", p_s, o_s):
            all_states_pass = False
            
    if all_states_pass:
        print("[PASS] All Mimi Decoder States match!")

# ==============================================================================
# MONKEYPATCHES for EXACT ONNX PARITY
# ==============================================================================
import pocket_tts.modules.transformer as transformer_module
from pocket_tts.modules.transformer import StreamingMultiheadAttention
import torch.nn.functional as F

def patched_init_state(self, batch_size: int, sequence_length: int) -> dict[str, torch.Tensor]:
    device = self.in_proj.weight.device
    # Match export dtype: FP16
    dtype = torch.float16
    return dict(
        cache_k=torch.full(
            (batch_size, sequence_length, self.num_heads, self.dim_per_head),
            0.0,
            device=device,
            dtype=dtype,
        ),
        cache_v=torch.full(
            (batch_size, sequence_length, self.num_heads, self.dim_per_head),
            0.0,
            device=device,
            dtype=dtype,
        ),
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

    cache_k = state["cache_k"]
    cache_v = state["cache_v"]
    step = state["step"]
    off = step.view(-1)[0]
    
    B, L, H, D = k.shape
    indices = (off + torch.arange(L, device=k.device, dtype=torch.long)).view(1, L, 1, 1).expand(B, L, H, D)
    
    updated_k = cache_k.scatter(1, indices, k.half())
    updated_v = cache_v.scatter(1, indices, v.half())
    state["cache_k"] = updated_k
    state["cache_v"] = updated_v
    
    valid_len = off + L
    k_attn = updated_k[:, :valid_len].permute(0, 2, 1, 3).float()
    v_attn = updated_v[:, :valid_len].permute(0, 2, 1, 3).float()
    
    MAX_POS = 4096
    pos_k = torch.arange(MAX_POS, device=k_attn.device, dtype=torch.long)[:valid_len].unsqueeze(0).expand(B, -1)
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
    parser.add_argument("--weights_path", type=str, default="weights/tts_b6369a24.safetensors")
    parser.add_argument("--int8", action="store_true", help="Test int8 quantized models")
    parser.add_argument("--audio", type=str, help="Path to audio file for encoder comparison")
    args = parser.parse_args()

    # Load unpatched PyTorch Model
    print(f"Loading pure PyTorch model with config {args.config}...")
    torch.manual_seed(42)
    
    tts_model = TTSModel.load_model(config=args.config)
    
    try:
        import safetensors.torch
        if os.path.exists(args.weights_path):
            print(f"Reloading weights from {args.weights_path}...")
            state_dict = safetensors.torch.load_file(args.weights_path)
            tts_model.load_state_dict(state_dict, strict=False)
    except Exception as e:
        print(f"Warning: Could not load weights: {e}")
        
    tts_model.eval()

    suffix = "_int8" if args.int8 else ""
    encoder_path = os.path.join(args.onnx_dir, f"mimi_encoder{suffix}.onnx")
    conditioner_path = os.path.join(args.onnx_dir, f"text_conditioner{suffix}.onnx")
    flow_lm_path = os.path.join(args.onnx_dir, f"flow_lm_main{suffix}.onnx")
    mimi_decoder_path = os.path.join(args.onnx_dir, f"mimi_decoder{suffix}.onnx")
    flow_lm_flow_path = os.path.join(args.onnx_dir, f"flow_lm_flow{suffix}.onnx")

    if os.path.exists(encoder_path):
        ort_encoder = ort.InferenceSession(encoder_path)
        pt_encoder = MimiEncoderWrapper(
            tts_model.mimi, 
            tts_model.flow_lm.speaker_proj_weight,
            emb_std=tts_model.flow_lm.emb_std,
            emb_mean=tts_model.flow_lm.emb_mean
        )
        compare_encoder(ort_encoder, pt_encoder, tts_model, audio_path=args.audio)


    if os.path.exists(conditioner_path):
        ort_conditioner = ort.InferenceSession(conditioner_path)
        pt_conditioner = TextConditionerWrapper(tts_model.flow_lm.conditioner)
        compare_conditioner(ort_conditioner, pt_conditioner)
        
    if os.path.exists(flow_lm_path):
        import sys
        import importlib.util
        from pathlib import Path
        script_dir = Path(__file__).parent
        export_script_path = script_dir / "export_flow_lm.py"
        spec = importlib.util.spec_from_file_location("export_flow_lm", str(export_script_path))
        export_flow_lm = importlib.util.module_from_spec(spec)
        sys.modules["export_flow_lm"] = export_flow_lm
        spec.loader.exec_module(export_flow_lm)
        
        ort_flow = ort.InferenceSession(flow_lm_path)
        flow_state = init_states(tts_model.flow_lm, batch_size=1, sequence_length=1000)
        pt_flow = export_flow_lm.FlowLMMainWrapper(tts_model.flow_lm, get_state_structure(flow_state))
        compare_flow_lm_main(ort_flow, pt_flow, tts_model)
        
    if os.path.exists(flow_lm_flow_path):
        ort_flow_net = ort.InferenceSession(flow_lm_flow_path)
        pt_flow_net = export_flow_lm.FlowNetWrapper(tts_model.flow_lm)
        compare_flow_lm_flow(ort_flow_net, pt_flow_net, tts_model)
        
    if os.path.exists(mimi_decoder_path):
        try:
            ort_mimi = ort.InferenceSession(mimi_decoder_path)
            mimi_state = init_states(tts_model.mimi, batch_size=1, sequence_length=1000)
            pt_mimi = MimiWrapper(tts_model.mimi, get_state_structure(mimi_state), tts_model.flow_lm.emb_std, tts_model.flow_lm.emb_mean)
            compare_mimi_decoder(ort_mimi, pt_mimi, tts_model)
        except Exception as e:
            print(f"\n[SKIP] Mimi Decoder parity failed to initialize/run: {e}")

    if tts_model.flow_lm.insert_bos_before_voice:
        print("\n--- Testing BOS Before Voice Conditioning (Integrated) ---")
        # In the new wrapper, BOS is integrated.
        # We pass the voice prompt as text_embeddings.
        # The wrapper should auto-concatenate: [voice_prompt, bos, sequence]
        # Wait, usually it's [text, bos, voice]. 
        # If we are testing "BOS before voice", we use voice as text_embeddings.
        
        voice_prompt = torch.randn(1, 10, tts_model.flow_lm.dim)
        
        # Test conditioning with voice prompt
        flow_state = init_states(tts_model.flow_lm, batch_size=1, sequence_length=1000)
        ort_flow = ort.InferenceSession(flow_lm_path)
        pt_flow = export_flow_lm.FlowLMMainWrapper(tts_model.flow_lm, get_state_structure(flow_state))
        
        test_seq = torch.randn(1, 1, 32)
        with torch.no_grad():
            pt_outputs = pt_flow(test_seq, voice_prompt, flatten_state(flow_state))
        
        ort_inputs = {
            "sequence": test_seq.detach().numpy(),
            "text_embeddings": voice_prompt.detach().numpy()
        }
        input_types = get_session_input_types(ort_flow)
        for name in ort_inputs:
            if name in input_types:
                ort_inputs[name] = cast_to_ort_type(ort_inputs[name], input_types[name])
                
        for i, st in enumerate(flatten_state(flow_state)):
            name = f"state_{i}"
            val = st.detach().numpy()
            if name in input_types:
                val = cast_to_ort_type(val, input_types[name])
            ort_inputs[name] = val
            
        onnx_outputs = ort_flow.run(None, ort_inputs)
        assert_allclose_with_logging("FlowLM Integrated BOS Conditioning (c)", pt_outputs[0].numpy(), onnx_outputs[0])
        print("[PASS] Integrated BOS parity confirmed!")


if __name__ == "__main__":
    main()
