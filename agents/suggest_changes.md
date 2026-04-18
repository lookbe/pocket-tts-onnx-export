# Comprehensive Technical Comparison & Roadmap: ONNX v2

This document provides a line-by-line technical comparison across all model versions (PyTorch v1/v2 and ONNX v1/v2), specifically documenting the mathematical workarounds required to port the multilingual architecture to a static graph.

---

## 1. Flow LM Main (`flow_lm_main.onnx`)
**The "Logic Brain"**: Transformer backbone and state (KV-Cache) management.

| Feature | PyTorch v1 (PT v1) | PyTorch v2 (PT v2) | ONNX v1 (Current) | ONNX v2 (Suggested) |
| :--- | :--- | :--- | :--- | :--- |
| **Speaker Input** | 512-dim | **32-dim** | 512-dim `input` | **32-dim** `input` |
| **BOS Marker** | N/A | `bos_before_voice` | N/A | **New Slot**: `bos_marker` |
| **Math Logic** | `cat([text, speak])` | `cat([text, bos, speak])` | `Concat` (2 tensors) | `Concat` (3 tensors) |
| **Attention Math** | SDPA (Native) | SDPA (Native) | **Decomposed Math**: `BatchMatMul(Q,K)` $\rightarrow$ `Softmax` $\rightarrow$ `BatchMatMul(A,V)` | **Fused SDPA** (requires Opset 17) or Improved Decomposition. |
| **State Loop** | Mutating `dict` | Mutating `dict` | **Explicit State Passing**: 100+ individual tensor inputs/outputs. | **Enhanced State Mapping**: Add state for the BOS index. |
| **Offset Logic** | `pos += input.len` | `pos += input.len` | `Sub` $\rightarrow$ `Add` nodes | `Sub` $\rightarrow$ `Add` + **Constant 1** (for BOS). |
| **Key Difference** | Monolingual sequence. | Multilingual ready. | Hardcoded 512 dims. | **Dynamic 32/512 support.** |

---

## 2. Mimi Encoder (`mimi_encoder.onnx`)
**The "Listener"**: Audio to compressed latents.

| Feature | PyTorch v1 (PT v1) | PyTorch v2 (PT v2) | ONNX v1 (Current) | ONNX v2 (Suggested) |
| :--- | :--- | :--- | :--- | :--- |
| **Bottleneck Dim** | 512 | **32** | 512 | **32** |
| **Downsample Logic** | Conv1d Stride 2 | Conv1d + Multi-proj | Fused Graph | Fused Graph + **Downlink**. |
| **Math Workaround** | `F.pad` | `F.pad` | `Pad` (Constant) | `Pad` (Constant) + `Reshape`. |
| **Projection Math** | `Linear(1024, 512)` | `Linear(1024, 32)` | `Gemm` (Large) | `Gemm` (Small) |
| **Parity Check** | High-dim fidelity. | Low-dim bottleneck. | Matches v1 Weights. | **Must be re-exported** for v2 weights. |

---

## 3. Flow LM Flow (`flow_lm_flow.onnx`)
**The "Sampler"**: Iterative latent refinement via flow matching.

| Feature | PyTorch v1 (PT v1) | PyTorch v2 (PT v2) | ONNX v1 (Current) | ONNX v2 (Suggested) |
| :--- | :--- | :--- | :--- | :--- |
| **Input Shape** | `[1, 512]` | `[1, 32]` | `[1, 512]` | `[1, 32]` |
| **AdaLN Math** | `x * (1+m) + b` | `x * (1+m) + b` | `Split` $\rightarrow$ `Mul` $\rightarrow$ `Add` | `Split` $\rightarrow$ `Mul` $\rightarrow$ `Add` |
| **Iteration Math** | `lsd_decode` Loop | `lsd_decode` Loop | **External Engine Loop**: Python calls model N times. | **External Engine Loop**: Worker calls model N times. |
| **Time Embed** | `cos(t*freq)` | `cos(t*freq)` | `Sin`/`Cos` Nodes | `Sin`/`Cos` Nodes |

---

## 4. Mimi Decoder (`mimi_decoder.onnx`)
**The "Vocoder"**: Latents to high-fidelity waveform.

| Feature | PyTorch v1 (PT v1) | PyTorch v2 (PT v2) | ONNX v1 (Current) | ONNX v2 (Suggested) |
| :--- | :--- | :--- | :--- | :--- |
| **Input Shape** | `[1, 512]` | `[1, 32]` | `[1, 512]` | `[1, 32]` |
| **Upsampling Logic** | Transpose Conv | Upsample + T-Conv | `ConvTranspose1d` | **New Expansion Layer** |
| **Math Workaround** | Sequential buffers. | Sequential buffers. | **Buffer Sliding**: `Concat(Buffer, New)` $\rightarrow$ `Slice(1, -1)`. | **Buffer Sliding**: Same, but with expansion math. |
| **Output Math** | Audio samples. | Audio samples. | `LeakyRelu` $\rightarrow$ `Tanh`. | `LeakyRelu` $\rightarrow$ `Tanh`. |

---

## Required "Mathematical Workaround" Code Changes

To align the ONNX v2 export with the PyTorch v2 "Multilingual Bridge," the following code patches must be applied:

### A. The 3-way Concatenation Workaround
Instead of the PyTorch `if` branch (which creates dynamic graphs), the export wrapper MUST use a static graph that can handle the BOS token:
```python
# onnx_export/wrappers.py -> FlowLMMainWrapper
if getattr(self.model, 'insert_bos_before_voice', False):
    # Math: Concat([T, B, S]) where T=text, B=bos, S=speaker
    combined = torch.cat([text_embeddings, self.model.bos_before_voice, audio_conditioning], dim=1)
else:
    # Math: Concat([T, S])
    combined = torch.cat([text_embeddings, audio_conditioning], dim=1)
```

### B. KV-Cache Offset Compensation
In v2, the `pos` (positional index) is shifted by 1 due to the BOS token.
**Workaround**: If `insert_bos_before_voice` is enabled, the `increment_steps` logic must be monkeypatched to ensure the graph's internal `offset` is always mathematically consistent with the number of tokens actually processed.

### C. Dimensional Bottlenecking (512 -> 32)
The Mimi models in v2 use a 32-dim latent space. 
**Workaround**: Update the `MimiWrapper` to include the `ConvDownsample1d` and `ConvTrUpsample1d` modules in the trace. These provide the necessary `MatMul` nodes to transform the data between the 512-dim transformer space and the 32-dim quantization space.
