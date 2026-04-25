# Optimization Plan for pocket-tts-onnx-export-fork

This document outlines the plan to integrate high-performance optimizations from the `PocketTTS.cpp` export approach into the `pocket-tts-onnx-export-fork` (V2) project.

## Goal
Achieve **2x-5x faster inference** by reducing memory bandwidth, eliminating O(N) tensor copies, and optimizing the ONNX computational graph for modern runtimes (C++, Web, etc.).

---

## 1. Precision & Memory Optimization: Switch to FP16 States

The current fork uses `float32` for all KV cache states. Transitioning to `float16` for states is the single most impactful optimization.

### Changes:
- **Refactor `init_state` Monkeypatch**: 
    - Initialize `cache` tensors with `dtype=torch.float16`.
    - Split the combined `(2, B, T, H, D)` cache into separate `cache_k` and `cache_v` tensors. This avoids redundant slicing on the first dimension during every step.
- **Update `append_and_get`**:
    - Cast incoming `k` and `v` tensors to `.half()` before insertion into the cache.
    - Ensure output tensors are cast back to `.float()` if needed for `SDPA` (Scaled Dot Product Attention), as some CPU-based ONNX runtimes are more optimized for FP32 math but FP16 I/O.

---

## 2. Graph Efficiency: O(1) Scatter Updates & Split States

The current fork uses a combined `[2, B, T, H, D]` cache and slice assignments (`new_cache[..., off:off+k.shape[1]] = k`). This creates a massive **O(N)** memory copy bottleneck in ONNX.

### Changes:
- **Split K/V States**: Decompose each layer's cache into separate `state_k` and `state_v` tensors. This avoids slicing the first dimension (`[0]` for K, `[1]` for V) on every inference step, which is an implicit copy in ONNX.
- **Replace Slicing with `torch.scatter`**: 
    - Use `torch.scatter` to update the circular buffer.
    - This maps to efficient `ScatterND` ops in ONNX, allowing the runtime to perform **in-place updates**.
- **Specialized Attention Implementation**: 
    - Use the manual RoPE and Attention loops from `PocketTTS.cpp` to avoid slow operators like `torch.tril` and ensure full control over the explicit state passing.

---

## 3. Preservation of V2 Features (BOS & Dimensions)

The fork introduced critical fixes for English V2 and multilingual models. These must be preserved.

### Items to Retain and Optimize:
- **Auto-BOS Injection (V2 Logic)**: 
    - Preserve the logic that prepends `bos_before_voice` when `step == 0`.
    - **Optimization**: Instead of dynamic slicing (`combined_text[:, start_idx:, :]`), use a static 3-way concatenation `[BOS, Text, Voice]` during the initial conditioning pass to keep the graph simple and fast.
- **32-Dim Latent Projections**: Maintain support for the narrower V2 bottlenecks (`speaker_proj_weight` shape `[1024, 32]`).
- **Mimi Normalization Fix**: Ensure `MimiEncoderWrapper` continues to project **raw** latents directly, as per the correct V2 reference implementation.

---

## 4. Technical Specification for New State Layout

The new state layout for `flow_lm_main.onnx` will follow this pattern (for 6 layers):

| Input/Output | Type | Shape | Description |
| :--- | :--- | :--- | :--- |
| `sequence` | `float32` | `[1, S, 32]` | Current latent sequence |
| `text_embeddings` | `float32` | `[1, T, 1024]` | Conditioning (Text/Voice/BOS) |
| `state_0` | **`float16`** | `[1, 1000, 16, 64]` | Layer 0 K-Cache |
| `state_1` | **`float16`** | `[1, 1000, 16, 64]` | Layer 0 V-Cache |
| `state_2` | `int64` | `[1]` | Layer 0 Step/Offset |
| ... | ... | ... | ... |

*Total states: 18 (6 layers × 3 states per layer).*

---

## 5. Inference Code Implications (C++ / Web)

To benefit from these optimizations, the inference engine (e.g., `PocketTTS.cpp`) must be updated:
- **FP16 Support**: The engine must be able to feed and receive `float16` tensors for the KV states.
- **IO Binding**: Use `Ort::IoBinding` to bind the same memory to `state_i` (input) and `out_state_i` (output) to enable the ONNX Runtime to perform the `ScatterND` update in-place.
- **State Layout**: The runtime must match the new split-state layout (K and V as separate inputs).

---

## 6. Implementation Steps

1.  **Preparation**: Update `requirements.txt` to ensure `onnx >= 1.14.0` for Opset 17 support.
2.  **Monkeypatching**: Refactor `scripts/export_flow_lm.py` with the new `init_state` and `append_and_get` logic.
3.  **Wrapper Update**: Modify `FlowLMMainWrapper` in `onnx_export/wrappers.py` to handle the split states and the integrated BOS logic.
4.  **Verification**: Update `scripts/compare_onnx_pytorch.py` to validate against **raw PyTorch model** outputs (not wrapper vs wrapper) to ensure zero regression in voice quality.
5.  **Quantization**: Verify that `quantize.py` correctly handles the new FP16 state inputs.

---

> [!IMPORTANT]
> This plan assumes the downstream inference engine (e.g., `PocketTTS.cpp` or a Web Worker) will be updated to handle the split FP16 states and correctly bind memory for in-place updates.
