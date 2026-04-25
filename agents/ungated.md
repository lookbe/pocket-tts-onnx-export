# PocketTTS ONNX Export Comparison

This document compares the two ONNX export approaches for PocketTTS:
1. `pocket-tts-onnx-export` (ungated version)
2. `PocketTTS.cpp/export_onnx.py` (gated version requiring authentication)

## Key Differences in ONNX Results

### Model Architecture & Structure
Both export approaches produce functionally equivalent ONNX models with the same architecture:
- **5 core models**: mimi_encoder, text_conditioner, flow_lm_main, flow_lm_flow, mimi_decoder
- **Same input/output signatures**: Identical tensor shapes and names
- **Same state handling**: Explicit KV cache passing for stateful models
- **Same opset version**: OPSET 17

### Numerical Precision & Values
The ONNX models should produce numerically equivalent outputs when given identical inputs:
- **FP32 models**: Bitwise identical within floating-point tolerance (typically < 1e-5 difference)
- **INT8 quantized models**: Similar quantization approach (dynamic quantization of MatMul ops only)
- **Validation**: Both approaches include validation against PyTorch reference implementations

### Metadata & Configuration
Both exports produce ONNX models with:
- Proper metadata embedding
- Consistent dynamic axes specifications
- Identical operator sets

## Inference Code Comparison

### PocketTTS.cpp Runtime Usage
The PocketTTS.cpp runtime expects the exact same model interface from both exports:

#### Input Requirements:
1. **text_conditioner.onnx**: 
   - Input: token_ids [1, T_text] int64
   - Output: embeddings [1, T_text, 1024] float32

2. **mimi_encoder.onnx**:
   - Input: audio [1, 1, T_samples] float32
   - Output: conditioning [1, N_frames, 1024] float32

3. **flow_lm_main.onnx** (stateful):
   - Inputs: sequence [1, S, 32], text_embeddings [1, T, 1024], KV cache states, step states
   - Outputs: conditioning [1024], eos_logit [1], updated KV cache states, updated step states

4. **flow_lm_flow.onnx** (stateless):
   - Inputs: c [1024], s [1,1], t [1,1], x [1,32] (all float32)
   - Output: flow_dir [1,32] float32

5. **mimi_decoder.onnx** (stateful):
   - Inputs: latent [1, N, 32], all decoder state tensors
   - Outputs: audio_frame [1, 1, T_samples], updated state tensors
53: 
54: ## Performance Optimization Analysis
55: 
56: While functionally equivalent, the `PocketTTS.cpp` export approach is significantly faster due to several key optimizations:
57: 
58: ### 1. Precision & Memory Bandwidth
59: - **FP16 KV Caches**: `PocketTTS.cpp` exports `flow_lm_main` with **float16** KV cache states. Since KV caches are the largest tensors moved between memory and the ONNX Runtime on every step, this reduces memory bandwidth requirements by **50%**.
60: - **Inference Code**: The `PocketTTS.cpp` runtime specifically supports `ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16` and uses a single-buffered mode for these states to encourage in-place updates.
61: 
62: ### 2. Efficient Cache Updates (Scatter vs. Slice/Concat)
63: - **Scatter-based Updates**: `PocketTTS.cpp` uses `torch.scatter` to update circular buffers. In ONNX, this translates to efficient `ScatterND` ops that can often be performed in-place.
64: - **Ungated Approach**: Uses slice assignments like `cache[..., step:step+1] = k`. In ONNX export, this frequently generates complex `Slice`, `Concat`, and `Join` chains that force the runtime to create a full copy of the entire 1000-sequence cache on every step (**O(N)** complexity).
65: 
66: ### 3. C++ Runtime Integration
67: - **Double-Buffering**: The `PocketTTS.cpp` runtime (`StateBufferIO`) manages state tensors with a double-buffering strategy. For FP16 states, it specifically binds the same memory to both input and output to allow the ONNX Runtime to perform updates truly in-place.
68: - **Zero-Copy States**: The C++ runtime avoids any tensor allocation or copying during the autoregressive loop, merely swapping pointers between "current" and "next" state buffers.
69: 
70: ### 4. Specialized Attention Implementation
71: - **Manual RoPE/Attention**: `PocketTTS.cpp` replaces the upstream `StreamingMultiheadAttention` with a custom implementation optimized for tracing. This avoids slow operators like `torch.tril` (which often gets exported as a large constant mask) and uses arithmetic-based masking which is much more efficient in ONNX.
72: 
73: ### Summary of Speed Gains
74: | Optimization | Impact | Reason |
75: | :--- | :--- | :--- |
76: | **FP16 States** | ~2x State I/O Speed | Half the data volume |
77: | **Scatter Updates** | O(1) vs O(N) cache update | Avoids full-cache copies |
78: | **In-place ORT Binding** | Reduced Memory Pressure | Direct memory reuse |
79: | **Simplified Masking** | Lower Op Overhead | Avoids large mask constants |
80: 

## Practical Differences

### Authentication Requirements
- **pocket-tts-onnx-export**: No authentication required (uses public weights)
- **PocketTTS.cpp/export_onnx.py**: Requires HuggingFace token for gated `kyutai-labs/pocket-tts` repo

### Dependencies & Setup
- **pocket-tts-onnx-export**: Self-contained with all necessary code
- **PocketTTS.cpp/export_onnx.py**: Requires installing `pocket-tts` package from HuggingFace

## Inference Code Architecture

The performance gap is also driven by how the inference engine interacts with these specific ONNX exports:

### PocketTTS.cpp (C++ Runtime)
- **State Management**: Uses a specialized `StateBufferIO` class that implements manual double-buffering.
- **FP16 Handling**: Explicitly detects `ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16` and uses `std::vector<uint16_t>` to hold cache data.
- **IO Binding**: Uses `Ort::IoBinding` to bind tensors to the session. For state tensors, it binds the output of one step as the input of the next step by swapping pointers, achieving near-zero copy overhead.
- **Streaming Implementation**: The C++ code is architected around a `StatefulRunner` that matches the flattened state layout of the `export_onnx.py` script.

### Ungated Approach (Typically Python/Generic)
- **Generic Runtime**: Often relies on standard `ort.InferenceSession.run()` which involves more overhead in copying numpy arrays to/from the ORT device context.
- **State Unpacking**: Typically involves Python-side logic to unpack/repack state dictionaries, which is slow in a tight autoregressive loop (100+ steps).
- **Memory Usage**: Without FP16 states, memory consumption is doubled, leading to more frequent cache misses and slower computation on memory-constrained devices.

### Conclusion
The `PocketTTS.cpp` approach is a **co-designed** system where the ONNX export and the C++ runtime are tightly coupled to maximize performance through data type optimization (FP16) and memory management (IO Binding + Double Buffering). The ungated version is more of a generic export for compatibility across various runtimes.


### Export Process
Both use similar methodologies:
1. Monkeypatching to bypass beartype constraints during tracing
2. Wrapper classes to expose internal states as explicit inputs/outputs
3. Dynamic axes for variable sequence lengths
4. Optional INT8 quantization targeting MatMul operations
5. Numerical validation against PyTorch references

## Conclusion
For end-users of the PocketTTS.cpp runtime, there should be **no practical difference** in inference behavior, performance, or output quality between ONNX models exported by either method. The choice between them depends solely on:
- Whether you have access to the gated `kyutai-labs/pocket-tts` repository
- Preference for a self-contained export script vs. using the official package

The ONNX model outputs will be interchangeable in the PocketTTS.cpp runtime without requiring any code changes.