# Low-Memory Export Variant

`export_multilingual.py` produces two model sets per language:

| Directory | Contents | Purpose |
| :--- | :--- | :--- |
| `models/<lang>/` | fp32, then `*_int8.onnx`, then `*_int4.onnx` (single embedded file, 1000-token state) | Default / desktop / higher-RAM targets |
| `models_low_mem/<lang>/` | fp32 (400-token state) + `*_int8.onnx`/`.onnx.data` + `*_int4.onnx`/`.onnx.data` (all separated) | Low-memory targets (e.g. iOS extensions, older/low-RAM devices) |

The low-mem variant is exactly **three changes combined**, and they only pay off together:

## 1. INT4 (`MatMulNBitsQuantizer`)
Shrinks weight bytes the most out of any lever here (vs. fp32 or int8). Applied via `scripts/quantize_int4.py`
to `text_conditioner`, `flow_lm_flow`, `flow_lm_main`, `mimi_decoder`. **`mimi_encoder` is deliberately skipped
and stays fp32** — it's only used for voice cloning (infrequent, not on the hot generation path), and it isn't
worth the extra quality risk for a model that small.

## 2. Separated (external) weight data
Without this, int4 gains are partly cancelled out at *load* time. onnxruntime's `session.use_mmap=1` only maps
the raw file bytes; for an **embedded** `.onnx` (weights inline in the protobuf), initializer bytes still get
copied into a private heap buffer during protobuf parsing — mmap saves you the initial file read, not the
in-memory duplicate. With `--separate_data` (`.onnx` + `.onnx.data`), the small `.onnx` only carries
`(offset, length)` references into the sibling `.onnx.data`, and ORT maps *that* file directly — the weight
bytes are genuinely zero-copy, clean, evictable pages instead of a private heap allocation.

This is why "int4 file is 40MB but RAM is 70MB" turned out to mostly be a **separated-data** problem, not an
int4 problem — see the RAM investigation in this session's history / `agents/optimization.md` for the deeper
trace using `PocketTTSLib.cpp`'s `LogMemoryFootprintGlobal`/`LogMemoryFootprintJob` (`dirty_private_est` vs
`mmap_file_est`).

Note this still doesn't make int4 free: `MatMulNBits`'s CPU kernel (`PrePack()`) allocates one packed
SIMD-layout buffer per quantized `MatMul` regardless of separated data — expect a real, unavoidable memory
cost roughly equal to the quantized weight payload size if prepacking stays enabled (see "Prepacking" below).

## 3. Reduced state (1000 → 400 tokens)
The KV-cache (`flow_lm_main`) and conv/attention state (`mimi_decoder`) capacity is **baked into the exported
graph's tensor shapes** — it can't be changed post-export by quantization or session options, only by
re-exporting with a smaller `sequence_length`. This is why the low-mem variant needs its own fp32 export pass
(`export_mimi_and_conditioner.py --seq_len 400` and `export_flow_lm.py --seq_len 400`), not just a re-quantize
of the existing 1000-token fp32 models. These state buffers live outside the `.onnx` file entirely (allocated
by the host at runtime — `PocketTTSLib.cpp`'s `StateMap`), so this lever specifically reduces *runtime* memory
that int4/separated-data don't touch at all. 400 was chosen to match `pocket-tts-cpp`'s existing
`kKvSeqCapacity` for `TARGET_OS_IOS` (`PocketTTSLib.cpp`); bump `LOW_MEM_SEQ_LEN` in `export_multilingual.py`
if a target needs a different cap (shorter state = less RAM but a shorter max utterance/voice-prompt length
before the cache wraps/truncates).

## Why not just pick one?
- int4 alone (embedded): still pays the full deserialize-copy cost on load (no separated data).
- separated data alone (fp32/int8): correct load behavior, but the file itself is still large.
- reduced state alone: shrinks the KV-cache but the weights are still fp32/int8-sized.

Only the combination targets all three memory sinks: on-disk/initializer weight bytes (int4), load-time
duplication (separated data), and runtime state buffers (reduced seq_len).

## Prepacking tradeoff (still applies to both variants)
`session.disable_prepacking=1` removes `MatMulNBits`'s packed-buffer cost entirely but makes int4 `MatMul`
execution noticeably slower (no SIMD-packed kernel). With separated data already in place, the prepack cost is
small and roughly fixed (≈ size of the quantized weight blob) — worth keeping prepacking **on** for the
low-mem variant unless a specific device's memory budget can't absorb even that. Tune per-device via
`session.disable_prepacking` in `PocketTTSLib.cpp`'s session options rather than by dropping int4 altogether.

## Running it
```
python export_multilingual.py                 # exports models/<lang>/ AND models_low_mem/<lang>/
python export_multilingual.py --skip_low_mem   # only models/<lang>/ (fp32 + int8 + int4 embedded)
```
Per-language, low-mem output requires `models/<lang>/model.safetensors` and its config YAML to already be
resolvable (same as the normal path) — it reuses those, it does not re-download weights.
