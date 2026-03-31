import argparse
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import onnx
from pocket_tts.models.tts_model import TTSModel
from pocket_tts.default_parameters import DEFAULT_VARIANT
from pocket_tts.modules.stateful_module import init_states, StatefulModule
from pocket_tts.modules.transformer import StreamingMultiheadAttention
from pocket_tts.modules.mimi_transformer import MimiStreamingMultiheadAttention, KVCacheResult
from onnx_export.export_utils import get_state_structure, flatten_state

# EXTREME AGGRESSIVE BEARTYPE DISABLE (Must be at the very top)
import sys
from unittest.mock import MagicMock
mock_beartype = MagicMock()
sys.modules["beartype"] = mock_beartype
sys.modules["beartype.claw"] = MagicMock()
sys.modules["beartype.door"] = MagicMock()
sys.modules["beartype.roar"] = MagicMock()
sys.modules["beartype.vale"] = MagicMock()
# 1. MONKEYPATCHES
# ==============================================================================

def patched_init_state(self, batch_size: int, sequence_length: int) -> dict[str, torch.Tensor]:
    dim_per_head = self.embed_dim // self.num_heads
    # Start with an EMPTY cache (0 length) for concat-based strategy
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
    
    # CONCAT-BASED CACHING: Better for XNNPACK and avoids ScatterND
    # state["cache"] shape: (2, B, H, Seq, D)
    cache = state["cache"]
    
    # Update cache via concatenation
    new_k = torch.cat([cache[0], k], dim=2)
    new_v = torch.cat([cache[1], v], dim=2)
    
    # Update state (out-of-place for ONNX sanity)
    state["cache"] = torch.stack([new_k, new_v], dim=0)
    
    return new_k, new_v

def patched_get_mask(self, shape: tuple[int, torch.Tensor], shift: torch.Tensor, device: torch.device):
    rows, cols_tensor = shape
    # rows is static/symbolic (T), cols_tensor is dynamic (T + step)
    
    # 4096 is safe for TTS context, slicing creates dynamic result in ONNX
    row_idx = torch.arange(rows, device=device).unsqueeze(1)
    col_idx = torch.arange(4096, device=device)[:cols_tensor].unsqueeze(0)
    
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
    
    # Keys and Values are now (B, H, Seq, D)
    k, v = self._complete_kv(k, v, state)

    current_step = state["step"]
    mask_shape = (t, t + current_step)
    shift = current_step
    attn_mask = self._get_mask(mask_shape, shift=shift, device=q.device)

    # Transpose q to (B, H, T, D) for standard attention
    q = q.transpose(1, 2)
    
    # ATTENTION: q(B, H, T, D), k(B, H, S, D), v(B, H, S, D)
    x = F.scaled_dot_product_attention(q, k, v, attn_mask)
    
    x = x.transpose(1, 2) # (B, T, H, D)
    x = x.reshape(b, t, self.num_heads * d)
    x = self.out_proj(x)
    return x

StreamingMultiheadAttention.init_state = patched_init_state
StreamingMultiheadAttention.increment_step = patched_increment_step
StreamingMultiheadAttention._streaming_offset = patched_streaming_offset
StreamingMultiheadAttention._complete_kv = patched_sma_complete_kv
StreamingMultiheadAttention._get_mask = patched_get_mask
StreamingMultiheadAttention.forward = patched_sma_forward

def patched_mimi_increment_step(self, state: dict, increment: int = 1):
    state["offset"] = state["offset"] + increment

MimiStreamingMultiheadAttention.increment_step = patched_mimi_increment_step

def patched_stateful_increment_step(self, state: dict, increment = 1):
    return state
StatefulModule.increment_step = patched_stateful_increment_step

# ==============================================================================
# 2. WRAPPERS
# ==============================================================================

class FlowLMMainWrapper(nn.Module):
    """
    Optimized Backbone Model.
    Inputs: 
      - sequence: (B, T, 32)
      - text_embeddings: (B, Text, 1024)
      - k_cache_block: (NumLayers, 1, H, Seq, D)
      - v_cache_block: (NumLayers, 1, H, Seq, D)
      - global_step: (1,)
    """
    def __init__(self, flow_lm, state_structure):
        super().__init__()
        self.flow_lm = flow_lm
        self.state_structure = state_structure
        
    def forward(self, sequence, text_embeddings, k_cache_block, v_cache_block, global_step):
        # Reconstruct state dictionary from grouped tensors
        # Each layer will slice its own part from the block
        model_state = {}
        layer_idx = 0
        
        # We assume a fixed structure of layers for FlowLM
        for name, m in self.flow_lm.named_modules():
            if isinstance(m, StreamingMultiheadAttention):
                model_state[name] = {
                    "step": global_step,
                    "cache": torch.stack([k_cache_block[layer_idx], v_cache_block[layer_idx]], dim=0)
                }
                layer_idx += 1
            elif isinstance(m, MimiStreamingMultiheadAttention):
                 # Handle Mimi specific states if necessary, or group them too
                 pass

        # Handle BOS replacement
        sequence = torch.where(torch.isnan(sequence), self.flow_lm.bos_emb, sequence)
        input_ = self.flow_lm.input_linear(sequence)
        
        # Backbone Forward Pass
        transformer_out = self.flow_lm.backbone(input_, text_embeddings, sequence, model_state=model_state)
        
        # Extract Conditioning (first token of last layer)
        c = transformer_out[:, 0]
        eos_logit = self.flow_lm.out_eos(c)

        # Update global step
        new_step = global_step + (sequence.shape[1] + text_embeddings.shape[1])
        
        # Group updated states back into blocks
        out_k_list = []
        out_v_list = []
        for name, m in self.flow_lm.named_modules():
            if isinstance(m, StreamingMultiheadAttention):
                s = model_state[name]
                out_k_list.append(s["cache"][0])
                out_v_list.append(s["cache"][1])
        
        out_k_cache = torch.stack(out_k_list, dim=0)
        out_v_cache = torch.stack(out_v_list, dim=0)
        
        return c, eos_logit, out_k_cache, out_v_cache, new_step


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
    parser = argparse.ArgumentParser(description="Export Optimized FlowLM models to ONNX.")
    parser.add_argument("--output_dir", "-o", type=str, default="onnx_models", help="Directory for output ONNX files")
    parser.add_argument("--weights_path", "-w", type=str, default="weights/tts_b6369a24.safetensors", help="Path to weights file")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    tts = TTSModel.load_model(DEFAULT_VARIANT).cpu().eval()
    
    if os.path.exists(args.weights_path):
        import safetensors.torch
        print(f"Reloading weights from {args.weights_path}...")
        state_dict = safetensors.torch.load_file(args.weights_path)
        tts.load_state_dict(state_dict, strict=False)
            
    # Init state with 0 length for dynamic concat
    state = init_states(tts.flow_lm, batch_size=1, sequence_length=0)
    
    # -------------------------------------------------------------
    # 1. Main Flow Model (Backbone)
    print("\nExporting Grouped FlowLM Main Model...")
    main_wrapper = FlowLMMainWrapper(tts.flow_lm, get_state_structure(state))
    
    # Group inputs: k_block, v_block, step
    k_list, v_list = [], []
    for name, m in tts.flow_lm.named_modules():
        if isinstance(m, StreamingMultiheadAttention):
            k_list.append(state[name]["cache"][0])
            v_list.append(state[name]["cache"][1])
    
    k_cache_block = torch.stack(k_list, dim=0)
    v_cache_block = torch.stack(v_list, dim=0)
    global_step = torch.tensor([0], dtype=torch.long)
    
    dummy_seq = torch.randn(1, 1, tts.flow_lm.ldim)
    dummy_text = torch.randn(1, 1, tts.flow_lm.dim)
    main_args = (dummy_seq, dummy_text, k_cache_block, v_cache_block, global_step)
    
    main_out_path = os.path.join(args.output_dir, "flow_lm_main.onnx")
    torch.onnx.utils.export(
        main_wrapper, main_args, main_out_path,
        input_names=["sequence", "text_embeddings", "k_cache", "v_cache", "step"],
        output_names=["conditioning", "eos_logit", "out_k_cache", "out_v_cache", "out_step"],
        dynamic_axes={
            "sequence": {1: "seq_len"}, 
            "text_embeddings": {1: "text_len"},
            "k_cache": {3: "kv_seq_len"},
            "v_cache": {3: "kv_seq_len"},
        },
        opset_version=17
    )
    print(f"Exported Optimized {main_out_path}")
    
    # 2. Flow Net Model
    print("\nExporting Flow Net Model...")
    flow_wrapper = FlowNetWrapper(tts.flow_lm)
    dummy_c = torch.randn(1, 1024)
    dummy_s = torch.tensor([[0.0]])
    dummy_t = torch.tensor([[1.0]])
    dummy_x = torch.randn(1, 32)
    
    flow_args = (dummy_c, dummy_s, dummy_t, dummy_x)
    flow_out_path = os.path.join(args.output_dir, "flow_lm_flow.onnx")
    torch.onnx.utils.export(
        flow_wrapper, flow_args, flow_out_path,
        input_names=["c", "s", "t", "x"],
        output_names=["flow_dir"],
        dynamic_axes={"c": {0: "batch"}, "s": {0: "batch"}, "t": {0: "batch"}, "x": {0: "batch"}},
        opset_version=17
    )
    print(f"Exported {flow_out_path}")
    print("\nDone! 2-Model split optimization complete.")

if __name__ == "__main__":
    main()
