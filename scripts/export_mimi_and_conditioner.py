import sys
from unittest.mock import MagicMock

# Create a mock that wraps calls (as decorators) and returns the original function
class SilentMock(MagicMock):
    def __call__(self, *args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return self
    def __getattr__(self, name):
        if name == "__path__":
            return []
        return super().__getattr__(name)

mock_beartype = SilentMock()
# Add specific names that are imported via 'from beartype import ...'
mock_beartype.BeartypeConf = MagicMock
mock_beartype.BeartypeStrategy = MagicMock

sys.modules['beartype'] = mock_beartype
sys.modules['beartype.claw'] = SilentMock()
sys.modules['beartype.roar'] = SilentMock()

import typing
sys.modules['beartype.typing'] = typing

import argparse
import torch
import torch.nn.functional as F
import math
import onnx
# Monkeypatch trunc_normal_ to be ONNX-friendly
def patched_trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # Use normal distribution clamped to range.
    # This avoids generic aten::uniform or complex rejection sampling loops in export
    with torch.no_grad():
        if std == 0:
            return tensor.fill_(mean).clamp_(a, b)
        return tensor.normal_(mean, std).clamp_(a, b)

torch.nn.init.trunc_normal_ = patched_trunc_normal_

# Monkeypatch global increment_steps to update scalar 'step' instead of resizing tensor
import pocket_tts.modules.stateful_module as stateful_module
from pocket_tts.modules.stateful_module import StatefulModule

def patched_increment_steps(module, model_state, increment=1):
    for module_name, m in module.named_modules():
        if not isinstance(m, StatefulModule):
            continue
        # Skip if module state is not present (e.g. filtered decoder-only states)
        if module_name in model_state:
            m.increment_step(model_state[module_name], increment)

stateful_module.increment_steps = patched_increment_steps

# Monkeypatch StreamingMultiheadAttention and its backend for ONNX tracing
import pocket_tts.modules.transformer as transformer_module
from pocket_tts.modules.transformer import StreamingMultiheadAttention

def patched_init_state(self, batch_size: int, sequence_length: int) -> dict[str, torch.Tensor]:
    device = self.in_proj.weight.device
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
    
    # RoPE offset
    if state is None:
        rope_offset = torch.zeros((), dtype=torch.long, device=q.device)
    else:
        rope_offset = state["step"].view(-1)[0]
        
    q, k = self.rope(q, k, offset=rope_offset)
    q = q.transpose(1, 2)

    k_attn, v_attn, pos_k, step = self._cache_backend.append_and_get(k, v, state)
    
    # Traceable pos_q
    MAX_POS = 4096
    off = step.view(-1)[0]
    pos_q = off + torch.arange(MAX_POS, device=q.device, dtype=torch.long)[:t].unsqueeze(0)
    
    # _build_attention_mask in lib is okay but we can make it more explicit if needed
    from pocket_tts.modules.transformer import _build_attention_mask
    attn_mask = _build_attention_mask(pos_q, pos_k, self.context)
    
    x = F.scaled_dot_product_attention(q, k_attn, v_attn, attn_mask, dropout_p=0.0)
    x = x.transpose(1, 2)
    b, t, h, d = x.shape
    x = x.reshape(b, t, h * d)
    x = self.out_proj(x)

    return x

# Apply patches
transformer_module._LinearKVCacheBackend.append_and_get = patched_append_and_get
StreamingMultiheadAttention.init_state = patched_init_state
StreamingMultiheadAttention.increment_step = patched_increment_step
StreamingMultiheadAttention.forward = patched_sma_forward

# Monkeypatch MimiStreamingMultiheadAttention logic
from pocket_tts.modules.mimi_transformer import MimiStreamingMultiheadAttention, KVCacheResult

def patched_mimi_init_state(self, batch_size: int, sequence_length: int) -> dict[str, torch.Tensor]:
    dim_per_head = self.embed_dim // self.num_heads
    device = self.in_proj.weight.device
    dtype = torch.float16
    capacity = 250
    return dict(
        offset=torch.zeros(batch_size, dtype=torch.long, device=device),
        cache_k=torch.zeros((batch_size, capacity, self.num_heads, dim_per_head), dtype=dtype, device=device),
        cache_v=torch.zeros((batch_size, capacity, self.num_heads, dim_per_head), dtype=dtype, device=device),
    )

MimiStreamingMultiheadAttention.init_state = patched_mimi_init_state

def patched_mimi_increment_step(self, state: dict, increment: int = 1):
    state["offset"] = state["offset"] + increment

MimiStreamingMultiheadAttention.increment_step = patched_mimi_increment_step

def patched_mimi_complete_kv(self, k, v, model_state: dict | None):
    if model_state is None:
        return KVCacheResult.from_kv(k, v)
        
    layer_state = self.get_state(model_state)
    cache_k = layer_state["cache_k"]
    cache_v = layer_state["cache_v"]
    offset = layer_state["offset"]
    
    capacity = cache_k.shape[1]
    B, H, T, D = k.shape
    
    # k/v come in [B, H, T, D]; store cache as [B, T, H, D].
    k_store = k.permute(0, 2, 1, 3)
    v_store = v.permute(0, 2, 1, 3)
    indexes = (offset.view(-1, 1) + torch.arange(T, device=offset.device, dtype=offset.dtype)) % capacity
    this_indexes = indexes.view(B, T, 1, 1).expand(-1, T, H, D)
    
    # Update cache (out-of-place for ONNX tracing)
    updated_k = cache_k.scatter(1, this_indexes, k_store.half())
    updated_v = cache_v.scatter(1, this_indexes, v_store.half())
    
    layer_state["cache_k"] = updated_k
    layer_state["cache_v"] = updated_v
    
    indexes_r = torch.arange(capacity, device=offset.device, dtype=torch.long)
    last_offset = offset.view(-1, 1) + T - 1
    end_index = last_offset % capacity
    delta = indexes_r - end_index
    
    positions = torch.where(delta <= 0, last_offset + delta, last_offset + delta - capacity)
    invalid = indexes_r >= (offset + T).view(-1, 1)
    positions = torch.where(invalid, torch.full_like(positions, -1), positions)
    
    keys = updated_k.permute(0, 2, 1, 3).float()
    values = updated_v.permute(0, 2, 1, 3).float()
    return KVCacheResult(keys, values, positions)

MimiStreamingMultiheadAttention._complete_kv = patched_mimi_complete_kv

import os
import onnxruntime as ort
import numpy as np
from pocket_tts.models.tts_model import TTSModel
from pocket_tts.default_parameters import DEFAULT_LANGUAGE
from pocket_tts.modules.stateful_module import init_states
from onnx_export.export_utils import get_state_structure, flatten_state, unflatten_state

EXPECTED_MIMI_STATE_COUNT = 56
EXPECTED_MIMI_MAX_STATE_INDEX = 55

def filter_mimi_states(states: dict) -> dict:
    """Filter Mimi states to only include modules used during decoding."""
    decoder_top_modules = ["decoder", "decoder_transformer", "upsample"]
    return {
        k: v for k, v in states.items() 
        if k.split('.')[0] in decoder_top_modules
    }

def assert_mimi_state_contract_from_flat(flat_mimi_state):
    count = len(flat_mimi_state)
    if count != EXPECTED_MIMI_STATE_COUNT:
        raise RuntimeError(
            f"Mimi state contract mismatch before export: got {count} states, "
            f"expected {EXPECTED_MIMI_STATE_COUNT}. "
            "This would break the web/C++ runner state feeds."
        )

def assert_mimi_state_contract_from_onnx(mimi_onnx_path: str):
    model = onnx.load(mimi_onnx_path)
    state_inputs = []
    for i in model.graph.input:
        if i.name.startswith("state_"):
            try:
                idx = int(i.name.split("_", 1)[1])
            except Exception:
                continue
            state_inputs.append((idx, i))

    if len(state_inputs) != EXPECTED_MIMI_STATE_COUNT:
        raise RuntimeError(
            f"Mimi ONNX state contract mismatch: model has {len(state_inputs)} state inputs, "
            f"expected {EXPECTED_MIMI_STATE_COUNT}."
        )

    max_idx = max(idx for idx, _ in state_inputs)
    if max_idx != EXPECTED_MIMI_MAX_STATE_INDEX:
        raise RuntimeError(
            f"Mimi ONNX state index mismatch: max state index is {max_idx}, "
            f"expected {EXPECTED_MIMI_MAX_STATE_INDEX}."
        )

from pocket_tts.modules.conv import StreamingConv1d, StreamingConvTranspose1d
def patched_conv1d_forward(self, x, model_state: dict | None):
    B, C, T = x.shape
    S = self._stride
    # Removed assert for trace
    if model_state is None:
        state = self.init_state(B, 0)
    else:
        state = self.get_state(model_state)
    TP = state["previous"].shape[-1]
    if TP and self.pad_mode == "replicate":
        # assert T >= TP 
        init = x[..., :1]
        # Out-of-place update
        new_prev = torch.where(
            state["first"].view(-1, 1, 1), init, state["previous"]
        )
        state["previous"] = new_prev
    if TP:
        x = torch.cat([state["previous"], x], dim=-1)
    y = self.conv(x)
    if TP:
        # Out-of-place update
        state["previous"] = x[..., -TP:]
        if self.pad_mode == "replicate":
            state["first"] = torch.zeros_like(state["first"])
    return y

def patched_convtr_forward(self, x, mimi_state: dict):
    state_dict = self.get_state(mimi_state)
    layer_state = state_dict["partial"]
    y = self.convtr(x)
    PT = layer_state.shape[-1]
    if PT > 0:
        # Avoid inplace on y if possible, but y is local. 
        # However, layer_state is input.
        # y[..., :PT] += layer_state -> y is modified. layer_state is read. Fine.
        # BUT for safe tracing:
        y_start = y[..., :PT] + layer_state
        y_end = y[..., PT:]
        y = torch.cat([y_start, y_end], dim=-1)

        bias = self.convtr.bias
        for_partial = y[..., -PT:]
        if bias is not None:
            for_partial = for_partial - bias[:, None]
        
        # Out-of-place update
        state_dict["partial"] = for_partial
        y = y[..., :-PT]
    return y

StreamingConv1d.forward = patched_conv1d_forward
StreamingConvTranspose1d.forward = patched_convtr_forward

from onnx_export.wrappers import FlowLMWrapper, MimiWrapper, MimiEncoderWrapper, TextConditionerWrapper
from pocket_tts.modules import conv
import math
def patched_get_extra_padding(x, kernel_size, stride, padding_total=0):
    # Using integer math to avoid math.ceil which breaks dynamic shapes in ONNX
    length = x.shape[-1]
    n_frames_num = length - kernel_size + padding_total
    # ceil(n_frames) = ceil(n_frames_num / stride + 1) = (n_frames_num + stride - 1) // stride + 1
    # Actually simpler: ideal_length is the smallest L' >= length such that (L' - kernel + padding_total) % stride == 0
    # Let target_n_frames = ceil((length - kernel + padding_total) / stride) + 1
    # n_frames_num might be negative if length < kernel
    # But for mimi, length is always >= kernel (1920) for any real audio.
    
    # (a + b - 1) // b logic:
    n_frames_ceil = (n_frames_num + stride - 1) // stride + 1
    # If n_frames_num < 0 (length < kernel), n_frames_ceil should be 1.
    if isinstance(n_frames_num, int):
        if n_frames_num < 0: n_frames_ceil = 1
    else:
        # Symbolic case
        n_frames_ceil = torch.maximum(torch.tensor(1, device=x.device), n_frames_ceil)
        
    ideal_length = (n_frames_ceil - 1) * stride + (kernel_size - padding_total)
    return ideal_length - length

conv.get_extra_padding_for_conv1d = patched_get_extra_padding

def export_models(output_dir="onnx_models", weights_path="weights/tts_b6369a24.safetensors", config_path=None):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading model with config: {config_path or DEFAULT_LANGUAGE}...")
    # Load model on CPU
    if config_path:
        tts_model = TTSModel.load_model(config=config_path)
    else:
        tts_model = TTSModel.load_model(DEFAULT_LANGUAGE)

    # Reload with local voice cloning weights (HF download may have failed)
    import safetensors.torch
    if os.path.exists(weights_path):
        print(f"Reloading weights from {weights_path} (with voice cloning)...")
        state_dict = safetensors.torch.load_file(weights_path)
        try:
            tts_model.load_state_dict(state_dict, strict=True)
            tts_model.has_voice_cloning = True
        except Exception as e:
            raise RuntimeError(
                f"Failed to load explicit weights from '{weights_path}'. "
                "Please provide weights matching the selected config."
            ) from e
    else:
        print(f"Warning: Weights file {weights_path} not found. Using defaults.")

    tts_model.eval()
    
    # ---------------------------------------------------------
    # Export Mimi Encoder (audio -> latents)
    # ---------------------------------------------------------
    print("Exporting Mimi Encoder...")
    
    mimi_encoder_wrapper = MimiEncoderWrapper(
        tts_model.mimi,
        speaker_proj_weight=tts_model.flow_lm.speaker_proj_weight,
        emb_std=tts_model.flow_lm.emb_std,
        emb_mean=tts_model.flow_lm.emb_mean
    )

    
    # Dummy audio: 1 second at 24kHz
    dummy_audio = torch.randn(1, 1, 24000)
    
    encoder_onnx_path = os.path.join(output_dir, "mimi_encoder.onnx")
    
    torch.onnx.export(
        mimi_encoder_wrapper,
        (dummy_audio,),
        encoder_onnx_path,
        input_names=["audio"],
        output_names=["latents"],
        dynamic_axes={"audio": {2: "audio_len"}},
        opset_version=17,
        dynamo=False,
        external_data=False
    )
    print(f"Mimi Encoder exported to {encoder_onnx_path}")
    
    # ---------------------------------------------------------
    # Export Text Conditioner (tokens -> embeddings)
    # ---------------------------------------------------------
    print("Exporting Text Conditioner...")
    
    text_conditioner_wrapper = TextConditionerWrapper(tts_model.flow_lm.conditioner)
    
    # Dummy tokens
    dummy_tokens = torch.randint(0, 1000, (1, 20))
    
    conditioner_onnx_path = os.path.join(output_dir, "text_conditioner.onnx")
    
    torch.onnx.export(
        text_conditioner_wrapper,
        (dummy_tokens,),
        conditioner_onnx_path,
        input_names=["token_ids"],
        output_names=["embeddings"],
        dynamic_axes={"token_ids": {1: "seq_len"}},
        opset_version=17,
        dynamo=False,
        external_data=False
    )
    print(f"Text Conditioner exported to {conditioner_onnx_path}")
    
    # Initialize state with static size sufficient for expected usage
    # 1000 tokens covers ~40s audio or long text prompts
    STATIC_SEQ_LEN = 1000
    
    flow_lm_onnx_path = None
    
    # ---------------------------------------------------------
    # Export Mimi
    # ---------------------------------------------------------
    print("Exporting Mimi...")
    
    mimi_state = init_states(tts_model.mimi, batch_size=1, sequence_length=STATIC_SEQ_LEN)
    
    mimi_structure = get_state_structure(mimi_state)
    flat_mimi_state = flatten_state(mimi_state)
    assert_mimi_state_contract_from_flat(flat_mimi_state)
    
    
    mimi_wrapper = MimiWrapper(
        tts_model.mimi, 
        mimi_structure,
        emb_std=tts_model.flow_lm.emb_std,
        emb_mean=tts_model.flow_lm.emb_mean
    )
    
    dummy_latent = torch.randn(1, 1, 32)
    mimi_args = (dummy_latent, flat_mimi_state)
    
    mimi_input_names = ["latent"] + [f"state_{i}" for i in range(len(flat_mimi_state))]
    mimi_output_names = ["audio_frame"] + [f"out_state_{i}" for i in range(len(flat_mimi_state))]
    
    # Mimi dynamic axes
    mimi_dynamic_axes = {
        "latent": {1: "seq_len"},
        # decode_from_latent outputs flattened audio samples: seq_len * 1920
        "audio_frame": {0: "num_samples"},
    }
    
    mimi_onnx_path = os.path.join(output_dir, "mimi_decoder.onnx")
    
    torch.onnx.export(
        mimi_wrapper,
        mimi_args,
        mimi_onnx_path,
        input_names=mimi_input_names,
        output_names=mimi_output_names,
        dynamic_axes=mimi_dynamic_axes,
        opset_version=17,
        dynamo=False
    )
    assert_mimi_state_contract_from_onnx(mimi_onnx_path)
    print(f"Mimi exported to {mimi_onnx_path}")
    
    # Note: BOS before voice is now embedded in flow_lm_main.onnx

    
    return flow_lm_onnx_path, mimi_onnx_path, tts_model

def verify_export(flow_lm_path, mimi_path, tts_model, output_dir="onnx_models"):
    print("Verifying export...")
    
    encoder_path = os.path.join(output_dir, "mimi_encoder.onnx")
    conditioner_path = os.path.join(output_dir, "text_conditioner.onnx")
    
    if os.path.exists(encoder_path):
        # ---------------------------------------------------------
        # Verify Mimi Encoder
        # ---------------------------------------------------------
        print("Verifying Mimi Encoder...")
        ort_encoder = ort.InferenceSession(encoder_path)
        
        # Test audio input
        test_audio = torch.randn(1, 1, 24000)  # 1 second
        
        # PyTorch run
        encoder_wrapper = MimiEncoderWrapper(
            tts_model.mimi,
            speaker_proj_weight=tts_model.flow_lm.speaker_proj_weight,
            emb_std=tts_model.flow_lm.emb_std,
            emb_mean=tts_model.flow_lm.emb_mean
        )

        with torch.no_grad():
            pt_encoder_out = encoder_wrapper(test_audio)
        
        # ONNX run
        onnx_encoder_out = ort_encoder.run(None, {"audio": test_audio.numpy()})[0]
        
        np.testing.assert_allclose(
            pt_encoder_out.numpy(), onnx_encoder_out, 
            rtol=1e-3, atol=1e-3
        )
        print("Mimi Encoder output matches!")
    
    if os.path.exists(conditioner_path):
        # ---------------------------------------------------------
        # Verify Text Conditioner
        # ---------------------------------------------------------
        print("Verifying Text Conditioner...")
        ort_conditioner = ort.InferenceSession(conditioner_path)
        
        # Test token input
        test_tokens = torch.randint(0, 1000, (1, 20))
        
        # PyTorch run
        conditioner_wrapper = TextConditionerWrapper(tts_model.flow_lm.conditioner)
        with torch.no_grad():
            pt_conditioner_out = conditioner_wrapper(test_tokens)
        
        # ONNX run
        onnx_conditioner_out = ort_conditioner.run(None, {"token_ids": test_tokens.numpy()})[0]
        
        np.testing.assert_allclose(
            pt_conditioner_out.numpy(), onnx_conditioner_out, 
            rtol=1e-5, atol=1e-5
        )
        print("Text Conditioner output matches!")
    
    if mimi_path and os.path.exists(mimi_path):
        # ---------------------------------------------------------
        # Verify Mimi
        # ---------------------------------------------------------
        ort_session_mimi = ort.InferenceSession(mimi_path)
        
        mimi_state = init_states(tts_model.mimi, batch_size=1, sequence_length=1000)
        flat_mimi_state = flatten_state(mimi_state)
        
        latent = torch.randn(1, 1, tts_model.flow_lm.ldim)
        
        # PyTorch run
        mimi_wrapper = MimiWrapper(
            tts_model.mimi, 
            get_state_structure(mimi_state),
            emb_std=tts_model.flow_lm.emb_std,
            emb_mean=tts_model.flow_lm.emb_mean
        )
        with torch.no_grad():
            pt_mimi_out = mimi_wrapper(latent, flat_mimi_state)
            
        pt_audio = pt_mimi_out[0].numpy()
        pt_mimi_states = [x.numpy() for x in pt_mimi_out[1:]]
        
        # ONNX run
        ort_mimi_inputs = {
            "latent": latent.numpy()
        }
        input_info_objs = {i.name: i for i in ort_session_mimi.get_inputs()}
        for i, state_tensor in enumerate(flat_mimi_state):
            name = f"state_{i}"
            if name in input_info_objs:
                val = state_tensor.numpy()
                t = input_info_objs[name].type
                if t == 'tensor(float16)':
                    val = val.astype(np.float16)
                elif t == 'tensor(float)':
                    val = val.astype(np.float32)
                elif t == 'tensor(int64)':
                    val = val.astype(np.int64)
                elif t == 'tensor(bool)':
                    val = val.astype(bool)
                ort_mimi_inputs[name] = val
            
        ort_mimi_outs = ort_session_mimi.run(None, ort_mimi_inputs)
        
        onnx_audio = ort_mimi_outs[0]
        onnx_mimi_states = ort_mimi_outs[1:]
        
        np.testing.assert_allclose(pt_audio, onnx_audio, rtol=1e-3, atol=1e-3)
        print("Mimi audio output matches!")
        
        for i, (pt_s, onnx_s) in enumerate(zip(pt_mimi_states, onnx_mimi_states)):
            np.testing.assert_allclose(pt_s, onnx_s, rtol=1e-3, atol=1e-3)
        print("Mimi states match!")
        
        print("Verification successful!")

def main():
    torch.manual_seed(42)
    parser = argparse.ArgumentParser(description="Export Mimi and Conditioner models to ONNX.")
    parser.add_argument("--output_dir", "-o", type=str, default="onnx_models", help="Directory for output ONNX files")
    parser.add_argument("--weights_path", "-w", type=str, default="weights/tts_b6369a24.safetensors", help="Path to weights file")
    parser.add_argument("--config", "-c", type=str, default=None, help="Path to config YAML file")
    args = parser.parse_args()
    
    flow, mimi, model = export_models(output_dir=args.output_dir, weights_path=args.weights_path, config_path=args.config)
    verify_export(flow, mimi, model, output_dir=args.output_dir)

if __name__ == "__main__":
    main()
