# Calculation & Logic Diff: pocket-tts vs pocket-tts-ori

This document details the logical changes in the model "calculations" (computational graph) between the original version (`pocket-tts-ori`) and the current version (`pocket-tts`).

## 1. Speaker Projection Logic
The way speaker embeddings from the Mimi encoder are projected into the FlowLM space has changed significantly.

| Feature | `pocket-tts-ori` | `pocket-tts` (Non-Ori/v2) | Impact |
| :--- | :--- | :--- | :--- |
| **Projection Shape** | Hardcoded `(1024, 512)` | Dynamic `(d_model, inner_dim)` | v2 models (e.g., English v2) use **32** instead of 512. |
| **Mapping Logic** | `F.linear(latents, weight)` | `F.linear(latents, weight)` | The math is the same, but the **bottleneck** is much narrower. |

> [!NOTE]
> Reducing the speaker projection to 32 dimensions suggests the model has become much more efficient at "cloning" a voice with less data, or it relies more on the transformer's internal states.

## 2. Mimi Resampling (Downsample/Upsample)
The original version performed resampling while maintaining the same dimension, whereas the new version can change dimensions during resampling.

- **Ori**: `ConvDownsample1d` only took `stride`. The output dimension always matched the input (512).
- **Non-Ori (v2)**: `ConvDownsample1d` now accepts `out_dimension`.
  ```python
  # pocket-tts/pocket_tts/models/mimi.py
  self.downsample = ConvDownsample1d(int(stride), dimension=dimension, out_dimension=inner_dim)
  ```
- **Calculation Change**: Before feeding to the quantizer, the latents are now compressed from the transformer dimension (512) down to the `inner_dim` (32).

## 3. Multilingual Transition: BOS Before Voice
The most visible logical addition for multilingual support is the `bos_before_voice` parameter.

- **Ori**: No specific marker before speaker identity conditioning.
- **Non-Ori (v2)**: Adds a learned parameter `self.bos_before_voice`.
  ```python
  # Added in pocket-tts/pocket_tts/models/flow_lm.py
  if self.insert_bos_before_voice:
      self.bos_before_voice = torch.nn.Parameter(torch.randn((1, 1, self.dim), dtype=dtype))
  ```
- **Graph Impact**: This adds an extra token to the conditioning sequence, acting as a "Start of Speaker" signal. This likely helps the model switch between different speaker identities or languages more robustly.

## 4. Input Padding & EOS Handling
The current version adds logic to handle short inputs and improved End-of-Sequence (EOS) behavior.

- **Pad with Spaces**: New flag `pad_with_spaces_for_short_inputs`.
- **Recommended Frames**: `model_recommended_frames_after_eos` allows the model to "breathe" after finishing a sentence, preventing abrupt cut-offs.

## Summary of Impact on ONNX
These logical changes explains the shape mismatches seen in `new_version_diff.md`:
1. **v1 ONNX** followed the `ori` logic (512-dim projections).
2. **v2 ONNX** implements the new `inner_dim=32` logic and includes the `bos_before_voice` buffer.
3. **Inference Workers** must account for the extra `bos_before_voice` token in the KV-cache management, or generation will be misaligned.
