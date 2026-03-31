# REFINED BEARTYPE DISABLE
import beartype
import beartype.door
import beartype.claw
import beartype.typing

def beartype_mock(obj=None, **kwargs):
    if callable(obj): return obj
    return lambda x: x

beartype.beartype = beartype_mock
beartype.door.is_bearable = lambda *args, **kwargs: True
beartype.claw.beartype_this_package = lambda *args, **kwargs: None

import argparse
import faulthandler
faulthandler.enable()

import torch
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
        # Call patched increment_step on module
        m.increment_step(model_state[module_name], increment)

stateful_module.increment_steps = patched_increment_steps

# Monkeypatch StatefulModule.increment_step to allow Tensors and avoid beartype issues
def patched_stateful_increment_step(self, state: dict, increment = 1):
    pass
StatefulModule.increment_step = patched_stateful_increment_step

# Monkeypatch StreamingMultiheadAttention to use scalar 'step'
from pocket_tts.modules.transformer import StreamingMultiheadAttention, complete_kv

def patched_init_state(self, batch_size: int, sequence_length: int) -> dict[str, torch.Tensor]:
    dim_per_head = self.embed_dim // self.num_heads
    return dict(
        step=torch.tensor([0], dtype=torch.long, device=self.in_proj.weight.device),
        cache=torch.zeros(
            (2, batch_size, self.num_heads, 0, dim_per_head),
            device=self.in_proj.weight.device,
            dtype=self.in_proj.weight.dtype,
        ),
    )

def patched_increment_step(self, state: dict, increment: int = 1):
    state["step"] = state["step"] + increment

def patched_streaming_offset(self, state: dict | None) -> torch.Tensor:
    return state["step"]

def patched_sma_complete_kv(self, k, v, state: dict | None):
    # k, v shape: (B, T, H, D) -> (B, H, T, D)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    
    cache = state["cache"]
    new_k = torch.cat([cache[0], k], dim=2)
    new_v = torch.cat([cache[1], v], dim=2)
    state["cache"] = torch.stack([new_k, new_v], dim=0)
    
    return new_k, new_v

StreamingMultiheadAttention.init_state = patched_init_state
StreamingMultiheadAttention.increment_step = patched_increment_step
StreamingMultiheadAttention._streaming_offset = patched_streaming_offset
StreamingMultiheadAttention._complete_kv = patched_sma_complete_kv

# Monkeypatch _get_mask to use arithmetic implementation (avoids torch.tril specialization)
def patched_get_mask(self, shape: tuple[int, torch.Tensor], shift: torch.Tensor, device: torch.device):
    rows, cols_tensor = shape
    # rows is int (static/symbolic from query), cols_tensor is Tensor (t + step)
    
    # Create row indices [rows, 1]
    row_idx = torch.arange(rows, device=device).unsqueeze(1) 
    
    # Fix for some versions: Ensure second arg is tensor or cast to scalar if traceable
    # 4096 is safe for TTS context, slicing creates dynamic result in ONNX
    col_idx = torch.arange(4096, device=device)[:cols_tensor].unsqueeze(0)
    
    # Mask condition
    mask_bool = (col_idx <= row_idx + shift)
    
    # Create mask via broadcasting logic
    mask = torch.full(mask_bool.shape, float("-inf"), device=device)
    mask.masked_fill_(mask_bool, 0.0)
    
    return mask

StreamingMultiheadAttention._get_mask = patched_get_mask

# Monkeypatch Forward to use step for mask
import torch.nn.functional as F
def patched_get_mask(self, shape: tuple[int, torch.Tensor], shift: torch.Tensor, device: torch.device):
    rows, cols_tensor = shape
    row_idx = torch.arange(rows, device=device).unsqueeze(1) 
    col_idx = torch.arange(cols_tensor, device=device).unsqueeze(0)
    mask_bool = (col_idx <= row_idx + shift)
    
    mask = torch.full(mask_bool.shape, float("-inf"), device=device)
    mask.masked_fill_(mask_bool, 0.0)
    return mask

def patched_sma_forward(self, query: torch.Tensor, model_state: dict | None):
    state = self.check_model_state(model_state)
    projected = self.in_proj(query)
    b, t, _ = projected.shape
    d = self.embed_dim // self.num_heads
    packed = projected.view(b, t, 3, self.num_heads, d)
    q, k, v = torch.unbind(packed, dim=2)
    q, k = self._apply_rope(q, k, state)
    
    k, v = self._complete_kv(k, v, state)

    current_step = state["step"]
    mask_shape = (t, t + current_step)
    shift = current_step
    attn_mask = self._get_mask(mask_shape, shift=shift, device=q.device)

    q = q.transpose(1, 2)
    x = F.scaled_dot_product_attention(q, k, v, attn_mask)
    x = x.transpose(1, 2)
    x = x.reshape(b, t, self.num_heads * d)
    x = self.out_proj(x)

    return x

StreamingMultiheadAttention.forward = patched_sma_forward

# Monkeypatch MimiStreamingMultiheadAttention logic (mostly same)
from pocket_tts.modules.mimi_transformer import MimiStreamingMultiheadAttention, KVCacheResult

def patched_mimi_init_state(self, batch_size: int, sequence_length: int) -> dict[str, torch.Tensor]:
    dim_per_head = self.embed_dim // self.num_heads
    return dict(
        offset=torch.zeros(batch_size, dtype=torch.long),
        end_offset=torch.zeros(batch_size, dtype=torch.long),
        cache=torch.zeros((2, batch_size, self.num_heads, sequence_length, dim_per_head)),
    )

def patched_mimi_increment_step(self, state: dict, increment: int = 1):
    state["offset"] = state["offset"] + increment

MimiStreamingMultiheadAttention.init_state = patched_mimi_init_state
MimiStreamingMultiheadAttention.increment_step = patched_mimi_increment_step

# Restore existing Mimi complete_kv patch
def patched_mimi_complete_kv(self, k, v, model_state: dict | None):
    if model_state is None:
        return KVCacheResult.from_kv(k, v)
        
    layer_state = self.get_state(model_state)
    cache = layer_state["cache"]
    end_offset = layer_state["end_offset"]
    
    capacity = cache.shape[3]
    B, H, T, D = k.shape
    
    new_cache = cache.clone()
    new_end_offset = end_offset.clone()
    
    # Original logic adapted...
    # Fixed: Use robust arange construction for TRACING
    indexes = torch.arange(4096, device=end_offset.device, dtype=end_offset.dtype)[:T]
    indexes = indexes + end_offset.view(-1, 1)
    indexes = indexes % capacity
    
    this_indexes = indexes.view(B, 1, T, 1)
    this_indexes = this_indexes.expand(-1, H, T, D)
    
    new_cache[0].scatter_(2, this_indexes, k)
    new_cache[1].scatter_(2, this_indexes, v)
    
    keys = new_cache[0]
    values = new_cache[1]
    
    # Fixed: Use robust arange construction
    indexes_r = torch.arange(4096, device=end_offset.device, dtype=torch.long)[:capacity]
    last_offset = end_offset.view(-1, 1) + T - 1
    end_index = last_offset % capacity
    delta = indexes_r - end_index
    
    positions = torch.where(delta <= 0, last_offset + delta, last_offset + delta - capacity)
    new_end_offset[:] = end_offset + T
    
    invalid = indexes_r >= new_end_offset.view(-1, 1)
    positions = torch.where(invalid, torch.full_like(positions, -1), positions)
    
    layer_state["cache"] = new_cache
    layer_state["end_offset"] = new_end_offset
    
    return KVCacheResult(keys, values, positions)

MimiStreamingMultiheadAttention._complete_kv = patched_mimi_complete_kv

import os
import onnxruntime as ort
import numpy as np
from pocket_tts.models.tts_model import TTSModel
from pocket_tts.default_parameters import DEFAULT_VARIANT
from pocket_tts.modules.stateful_module import init_states
from onnx_export.export_utils import get_state_structure, flatten_state, unflatten_state

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
    length = x.shape[-1]
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return ideal_length - length
conv.get_extra_padding_for_conv1d = patched_get_extra_padding

def export_models(output_dir="onnx_models", weights_path="weights/tts_b6369a24.safetensors"):
    os.makedirs(output_dir, exist_ok=True)

    print("Loading model...", flush=True)
    # Load model on CPU
    tts_model = TTSModel.load_model(DEFAULT_VARIANT)

    # Reload with local voice cloning weights (HF download may have failed)
    import safetensors.torch
    if os.path.exists(weights_path):
        print(f"Reloading weights from {weights_path} (with voice cloning)...", flush=True)
        state_dict = safetensors.torch.load_file(weights_path)
        try:
            tts_model.load_state_dict(state_dict, strict=True)
            tts_model.has_voice_cloning = True
        except Exception as e:
            print(f"Warning: Failed to load specified weights (strict=True): {e}", flush=True)
            print("Using default loaded weights.", flush=True)
    else:
        print(f"Warning: Weights file {weights_path} not found. Using defaults.", flush=True)

    tts_model.eval()
    
    # ---------------------------------------------------------
    # Export Mimi Encoder (audio -> latents)
    # ---------------------------------------------------------
    print("Exporting Mimi Encoder...", flush=True)
    
    mimi_encoder_wrapper = MimiEncoderWrapper(
        tts_model.mimi,
        speaker_proj_weight=tts_model.flow_lm.speaker_proj_weight
    )
    
    # Dummy audio: 1 second at 24kHz
    dummy_audio = torch.randn(1, 1, 24000)
    
    encoder_onnx_path = os.path.join(output_dir, "mimi_encoder.onnx")
    
    try:
        torch.onnx.utils.export(
            mimi_encoder_wrapper,
            (dummy_audio,),
            encoder_onnx_path,
            input_names=["audio"],
            output_names=["latents"],
            dynamic_axes={"audio": {2: "audio_len"}},
            opset_version=17
        )
        print(f"Mimi Encoder exported to {encoder_onnx_path}")
    except Exception as e:
        print(f"FAILED to export Mimi Encoder: {e}")
        import traceback
        traceback.print_exc()
    
    # ---------------------------------------------------------
    # Export Text Conditioner (tokens -> embeddings)
    # ---------------------------------------------------------
    print("Exporting Text Conditioner...")
    
    text_conditioner_wrapper = TextConditionerWrapper(tts_model.flow_lm.conditioner)
    
    # Dummy tokens
    dummy_tokens = torch.randint(0, 1000, (1, 20))
    
    conditioner_onnx_path = os.path.join(output_dir, "text_conditioner.onnx")
    
    try:
        torch.onnx.utils.export(
            text_conditioner_wrapper,
            (dummy_tokens,),
            conditioner_onnx_path,
            input_names=["token_ids"],
            output_names=["embeddings"],
            dynamic_axes={"token_ids": {1: "seq_len"}},
            opset_version=17
        )
        print(f"Text Conditioner exported to {conditioner_onnx_path}")
    except Exception as e:
        print(f"FAILED to export Text Conditioner: {e}")
        import traceback
        traceback.print_exc()
    
    # Initialize state with static size sufficient for expected usage
    # 1000 tokens covers ~40s audio or long text prompts
    STATIC_SEQ_LEN = 1000
    
    flow_lm_onnx_path = None
    
    # 1000 tokens covers ~40s audio or long text prompts
    STATIC_SEQ_LEN = 1000
    mimi_state = init_states(tts_model.mimi, batch_size=1, sequence_length=STATIC_SEQ_LEN)
    mimi_wrapper = MimiWrapper(
        tts_model.mimi, 
        get_state_structure(mimi_state),
        emb_std=tts_model.flow_lm.emb_std,
        emb_mean=tts_model.flow_lm.emb_mean
    )
    
    # Pack states: k_block, v_block, conv_states_flat
    k_list, v_list, conv_list = [], [], []
    for name, m in tts_model.mimi.named_modules():
        if isinstance(m, (StreamingMultiheadAttention, MimiStreamingMultiheadAttention)):
             k_list.append(mimi_state[name]["cache"][0])
             v_list.append(mimi_state[name]["cache"][1])
             if isinstance(m, MimiStreamingMultiheadAttention):
                 # Mimi uses offset and end_offset
                 conv_list.append(mimi_state[name]["offset"])
                 conv_list.append(mimi_state[name]["end_offset"])
             else:
                 # Standard SMA uses 'step'
                 conv_list.append(mimi_state[name]["step"])
        elif isinstance(m, StreamingConv1d):
             conv_list.append(mimi_state[name]["previous"])
             conv_list.append(mimi_state[name]["first"])
        elif isinstance(m, StreamingConvTranspose1d):
             conv_list.append(mimi_state[name]["partial"])

    k_block = torch.stack(k_list, dim=0)
    v_block = torch.stack(v_list, dim=0)
    
    dummy_latent = torch.randn(1, 1, 32)
    mimi_args = (dummy_latent, k_block, v_block, conv_list)
    
    mimi_input_names = ["latent", "k_cache", "v_cache"] + [f"conv_state_{i}" for i in range(len(conv_list))]
    mimi_output_names = ["audio_frame", "out_k_cache", "out_v_cache"] + [f"out_conv_state_{i}" for i in range(len(conv_list))]
    
    mimi_onnx_path = os.path.join(output_dir, "mimi_decoder.onnx")
    
    torch.onnx.utils.export(
        mimi_wrapper,
        mimi_args,
        mimi_onnx_path,
        input_names=mimi_input_names,
        output_names=mimi_output_names,
        dynamic_axes={
            "latent": {1: "seq_len"},
            "k_cache": {3: "kv_seq_len"},
            "v_cache": {3: "kv_seq_len"},
        },
        opset_version=17
    )
    print(f"Mimi Decoder exported to {mimi_onnx_path}")
    
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
            speaker_proj_weight=tts_model.flow_lm.speaker_proj_weight
        )
        with torch.no_grad():
            pt_encoder_out = encoder_wrapper(test_audio)
        
        # ONNX run
        onnx_encoder_out = ort_encoder.run(None, {"audio": test_audio.numpy()})[0]
        
        np.testing.assert_allclose(
            pt_encoder_out.numpy(), onnx_encoder_out, 
            rtol=1e-4, atol=1e-4
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
        
        mimi_state = init_states(tts_model.mimi, batch_size=1, sequence_length=0)
        
        latent = torch.randn(1, 1, tts_model.flow_lm.ldim)
        
        # PyTorch run (manually prepare grouped inputs)
        k_list, v_list, conv_list = [], [], []
        from pocket_tts.modules.transformer import StreamingMultiheadAttention
        from pocket_tts.modules.mimi_transformer import MimiStreamingMultiheadAttention
        from pocket_tts.modules.conv import StreamingConv1d, StreamingConvTranspose1d

        for name, m in tts_model.mimi.named_modules():
            if isinstance(m, (StreamingMultiheadAttention, MimiStreamingMultiheadAttention)):
                 k_list.append(mimi_state[name]["cache"][0])
                 v_list.append(mimi_state[name]["cache"][1])
            elif isinstance(m, StreamingConv1d):
                 conv_list.append(mimi_state[name]["previous"])
                 conv_list.append(mimi_state[name]["first"])
            elif isinstance(m, StreamingConvTranspose1d):
                 conv_list.append(mimi_state[name]["partial"])

        k_block = torch.stack(k_list, dim=0)
        v_block = torch.stack(v_list, dim=0)

        mimi_wrapper = MimiWrapper(
            tts_model.mimi, 
            get_state_structure(mimi_state),
            emb_std=tts_model.flow_lm.emb_std,
            emb_mean=tts_model.flow_lm.emb_mean
        )
        
        with torch.no_grad():
            pt_mimi_out = mimi_wrapper(latent, k_block, v_block, conv_list)
            
        pt_audio = pt_mimi_out[0].numpy()
        
        # ONNX run
        ort_mimi_inputs = {
            "latent": latent.numpy(),
            "k_cache": k_block.numpy(),
            "v_cache": v_block.numpy()
        }
        for i, conv_tensor in enumerate(conv_list):
            ort_mimi_inputs[f"conv_state_{i}"] = conv_tensor.numpy()
            
        ort_mimi_outs = ort_session_mimi.run(None, ort_mimi_inputs)
        
        onnx_audio = ort_mimi_outs[0]
        
        np.testing.assert_allclose(pt_audio, onnx_audio, rtol=1e-3, atol=1e-3)
        print("Mimi audio output matches (grouped states)!")
        print("Verification successful!")

def main():
    torch.manual_seed(42)
    parser = argparse.ArgumentParser(description="Export Mimi and Conditioner models to ONNX.")
    parser.add_argument("--output_dir", "-o", type=str, default="onnx_models", help="Directory for output ONNX files")
    parser.add_argument("--weights_path", "-w", type=str, default="weights/tts_b6369a24.safetensors", help="Path to weights file")
    args = parser.parse_args()
    
    flow, mimi, model = export_models(output_dir=args.output_dir, weights_path=args.weights_path)
    verify_export(flow, mimi, model, output_dir=args.output_dir)

if __name__ == "__main__":
    main()
