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
import torch.nn as nn
import torch.nn.functional as F
import os
import onnx
import numpy as np
from pocket_tts.models.tts_model import TTSModel
from pocket_tts.default_parameters import DEFAULT_LANGUAGE
from pocket_tts.modules.stateful_module import init_states, StatefulModule
from pocket_tts.modules.transformer import StreamingMultiheadAttention
from onnx_export.export_utils import get_state_structure, flatten_state

# ==============================================================================
# 1. MONKEYPATCHES
# ==============================================================================

# Monkeypatch StreamingMultiheadAttention and its backend for ONNX tracing
import pocket_tts.modules.transformer as transformer_module
from pocket_tts.modules.transformer import StreamingMultiheadAttention

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

# Apply patches
transformer_module._LinearKVCacheBackend.append_and_get = patched_append_and_get
StreamingMultiheadAttention.init_state = patched_init_state
StreamingMultiheadAttention.increment_step = patched_increment_step
StreamingMultiheadAttention.forward = patched_sma_forward

def patched_stateful_increment_step(self, state: dict, increment = 1):
    return state
StatefulModule.increment_step = patched_stateful_increment_step

# ==============================================================================
# 2. WRAPPERS
# ==============================================================================

class FlowLMMainWrapper(nn.Module):
    """
    Unified Backbone Model for both Conditioning and AR steps.
    Inputs: 
      - sequence: (B, T, 32)
      - text_embeddings: (B, Text, 1024)
      - state_*: KVCache states
    Outputs:
      - conditioning: (B, 1024) - used for Flow step
      - eos_logit: (B, 1) - used for EOS detection
      - out_state_*: Updated states
    """
    def __init__(self, flow_lm, state_structure, eos_threshold=-4.0):
        super().__init__()
        self.flow_lm = flow_lm
        self.state_structure = state_structure
        self.eos_threshold = eos_threshold
        
        # Embed BOS before voice if it exists
        self.insert_bos_before_voice = getattr(flow_lm, 'insert_bos_before_voice', False)
        if hasattr(flow_lm, "bos_before_voice"):
            # We register it as a buffer so it's embedded in the ONNX graph
            self.register_buffer("bos_before_voice", flow_lm.bos_before_voice)
        else:
            # Placeholder for models that don't use BOS
            self.register_buffer("bos_before_voice", torch.zeros((1, 1, 1024)))
            self.insert_bos_before_voice = False
        
    def forward(self, sequence, text_embeddings, state_flat):
        idx = 0
        def unflatten_recursive(struct):
            nonlocal idx
            s = {}
            for k, v in  sorted(struct.items()):
                if isinstance(v, dict): e = unflatten_recursive(v)
                else: 
                    e = state_flat[idx]
                    idx += 1
                s[k] = e
            return s
        model_state = unflatten_recursive(self.state_structure)
        
        # Auto-BOS Injection Logic
        # Detect if this is the first call (step 0) and we have conditioning text/voice
        # We find the step tensor from the first transformer layer
        first_step_tensor = model_state['transformer.layers.0.self_attn']['step']
        is_first_step = (first_step_tensor == 0).all()
        has_text = (text_embeddings.size(1) > 0)
        
        # 1 if we should prepend, 0 otherwise. 
        # Using float64/long logic to ensure it traces to a dynamic index in ONNX
        use_bos = (is_first_step & has_text & torch.tensor(self.insert_bos_before_voice, device=sequence.device)).long()
        
        # Always concatenate, then slice dynamically
        bos_expanded = self.bos_before_voice.expand(text_embeddings.size(0), -1, -1)
        combined_text = torch.cat([bos_expanded, text_embeddings], dim=1)
        
        # start_idx = 1 - use_bos. If use_bos=1, start=0 (keep BOS). If use_bos=0, start=1 (skip BOS).
        start_idx = 1 - use_bos
        text_embeddings = combined_text[:, start_idx:, :]

        # Handle BOS replacement (NaN -> bos_emb)
        sequence = torch.where(torch.isnan(sequence), self.flow_lm.bos_emb, sequence)

        input_ = self.flow_lm.input_linear(sequence)
        
        # Backbone Forward Pass
        # Backbone Forward Pass
        transformer_out = self.flow_lm.transformer(torch.cat([text_embeddings, input_], dim=1), model_state)
        if self.flow_lm.out_norm:
            transformer_out = self.flow_lm.out_norm(transformer_out)
        
        # Splicing logic: follow PyTorch precisely
        # If sequence.shape[1] is 0, -0: is equivalent to 0:, which returns everything.
        # This is CRITICAL for c extraction during prompting.
        transformer_out = transformer_out[:, -sequence.shape[1] :]
        
        increment = sequence.shape[1] + text_embeddings.shape[1]

        # Extract Conditioning
        batch_size = transformer_out.shape[0]
        dim = transformer_out.shape[2]
        dummy_out = torch.zeros((batch_size, 1, dim), device=transformer_out.device, dtype=transformer_out.dtype)
        
        # We ensure we take the LAST token of transformer_out if it exists.
        combined_for_c = torch.cat([dummy_out, transformer_out], dim=1)
        c = combined_for_c[:, -1]
        
        # Extract EOS logit
        eos_logit = self.flow_lm.out_eos(c)

        # State Increment
        # increment is already calculated above based on BOS insertion

        
        def recurse_increment(s):
            if "step" in s: s["step"] = s["step"] + increment
            if "offset" in s: s["offset"] = s["offset"] + increment
            for k, v in s.items(): 
                if isinstance(v, dict): recurse_increment(v)
        recurse_increment(model_state)
        
        from onnx_export.export_utils import flatten_state as fs
        out_state = fs(model_state)
        
        return c, eos_logit, *out_state


class FlowNetWrapper(nn.Module):
    """
    Stateless wrapper for FlowNet.
    Inputs: c (conditioning), s (timestep), t (timestep), x (latent)
    Output: flow_dir
    """
    def __init__(self, flow_lm):
        super().__init__()
        self.flow_net = flow_lm.flow_net
        
    def forward(self, c, s, t, x):
        return self.flow_net(c, s, t, x)


# ==============================================================================
# 3. EXPORT SCRIPT
# ==============================================================================

def main():
    torch.manual_seed(42)
    parser = argparse.ArgumentParser(description="Export FlowLM models to ONNX.")
    parser.add_argument("--output_dir", "-o", type=str, default="onnx_models", help="Directory for output ONNX files")
    parser.add_argument("--weights_path", "-w", type=str, default="weights/tts_b6369a24.safetensors", help="Path to weights file used to load FlowLM")
    parser.add_argument("--config", "-c", type=str, default=None, help="Path to config YAML file")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model with config: {args.config or DEFAULT_LANGUAGE}...")
    if args.config:
        tts = TTSModel.load_model(config=args.config).cpu().eval()
    else:
        tts = TTSModel.load_model(DEFAULT_LANGUAGE).cpu().eval()
    
    # Reload weights if available to match production
    if os.path.exists(args.weights_path):
        import safetensors.torch
        print(f"Reloading weights from {args.weights_path}...")
        state_dict = safetensors.torch.load_file(args.weights_path)
        # Load only common keys or strict if possible; strict might fail if keys missing in state dict
        # Assuming safe load for now or rely on load_model matching the weights
        try:
            tts.load_state_dict(state_dict, strict=False)
        except Exception as e:
            print(f"Warning: Failed to reload specified weights: {e}")
            
    # Init patched state
    STATIC_SEQ_LEN = 1000
    state = init_states(tts.flow_lm, batch_size=1, sequence_length=STATIC_SEQ_LEN)
    structure = get_state_structure(state)
    flat_state = flatten_state(state)
    
    state_input_names = [f"state_{i}" for i in range(len(flat_state))]
    state_output_names = [f"out_state_{i}" for i in range(len(flat_state))]
    
    # -------------------------------------------------------------
    # 1. Main Flow Model (Backbone)
    print("\nExporting FlowLM Main Model (Backbone)...")
    main_wrapper = FlowLMMainWrapper(tts.flow_lm, structure)
    
    # Needs to handle dynamic axes for both seq and text
    dummy_seq = torch.randn(1, 1, tts.flow_lm.ldim)
    dummy_text = torch.randn(1, 1, tts.flow_lm.dim)
    main_args = (dummy_seq, dummy_text, flat_state)
    
    main_out_path = os.path.join(args.output_dir, "flow_lm_main.onnx")
    torch.onnx.export(
        main_wrapper, main_args, main_out_path,
        input_names=["sequence", "text_embeddings"] + state_input_names,
        output_names=["conditioning", "eos_logit"] + state_output_names,
        dynamic_axes={"sequence": {1: "seq_len"}, "text_embeddings": {1: "text_len"}},
        opset_version=17, dynamo=False
    )

    print(f"Exported {main_out_path}")

    # 1.1 Export BOS embedding logic REMOVED (now embedded in flow_lm_main.onnx)
    
    # 2. Flow Net Model
    print("\nExporting Flow Net Model...")
    flow_wrapper = FlowNetWrapper(tts.flow_lm)
    # Inputs: c(B, 1024), s(B,1), t(B,1), x(B, 32)
    dummy_c = torch.randn(1, 1024)
    dummy_s = torch.tensor([[0.0]])
    dummy_t = torch.tensor([[1.0]])
    dummy_x = torch.randn(1, 32)
    
    flow_args = (dummy_c, dummy_s, dummy_t, dummy_x)
    
    flow_out_path = os.path.join(args.output_dir, "flow_lm_flow.onnx")
    torch.onnx.export(
        flow_wrapper, flow_args, flow_out_path,
        input_names=["c", "s", "t", "x"],
        output_names=["flow_dir"],
        dynamic_axes={"c": {0: "batch"}, "s": {0: "batch"}, "t": {0: "batch"}, "x": {0: "batch"}},
        opset_version=14, dynamo=False
    )
    print(f"Exported {flow_out_path}")
    
    # Internal Parity Check
    import onnxruntime as ort
    sess = ort.InferenceSession(flow_out_path)
    onnx_out = sess.run(None, {
        "c": dummy_c.numpy(),
        "s": dummy_s.numpy(),
        "t": dummy_t.numpy(),
        "x": dummy_x.numpy()
    })[0]
    pt_out = flow_wrapper(*flow_args).detach().numpy()
    np.testing.assert_allclose(pt_out, onnx_out, rtol=1e-5, atol=1e-5)
    print("Internal FlowNet parity check PASSED.")

    print("\nDone! 2-Model split optimization complete.")

if __name__ == "__main__":
    main()
