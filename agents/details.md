# Detailed Explanation: Exporting PocketTTS to ONNX

This document explains the technical details of how the PocketTTS model (originally in PyTorch/Safetensors) is converted into a set of 5 optimized ONNX models for high-performance inference in environments like the Web or C++.

## 1. Safetensors vs. ONNX: Where is the Calculation?

### The "Data" (Safetensors)
You are correct that **Safetensors is just data**. It contains only the trained weights (tensors) of the model. It contains no code, no logic, and no instructions on how to perform a calculation. 
- **Analogy**: Safetensors is like a "Sheet of Music." It has the notes, but it cannot play itself.

### The "Logic" (PyTorch Code)
The calculation logic lives in the **Python source code** (e.g., `pocket_tts/models/tts_model.py`). This code defines the architecture: the convolutions, the transformers, and the mathematical operations.
- **Analogy**: The source code is the "Musician" who knows how to read the notes and play the instrument.

### The "Graph" (ONNX)
**ONNX (Open Neural Network Exchange)** is different. It combines **both** the weights and the calculation logic into a single file called a **Computational Graph**.
- When we "export" to ONNX, PyTorch "traces" the execution of the Python code. It records every mathematical operation (Add, Mul, MatMul, Conv) and saves them as nodes in a directed graph.
- **Analogy**: ONNX is like a "Player Piano Roll" or a "Recording." It has both the music and the instructions to play it, bundled together.

---

## 2. Why 5 ONNX Models? (Architecture Split)

The system is split into 5 models instead of 1 large file to optimize for **memory, performance, and flexibility**. 

| Model | Filename | Purpose | Why separate? |
| :--- | :--- | :--- | :--- |
| **1. Text Conditioner** | `text_conditioner.onnx` | Tokens $\rightarrow$ Embeddings | Only runs once per sentence. |
| **2. Mimi Encoder** | `mimi_encoder.onnx` | Audio $\rightarrow$ Latents | Only needed for fine-tuning or zero-shot voice cloning. |
| **3. Flow LM Main** | `flow_lm_main.onnx` | Transformer Backbone | The "Brain." Runs once per step. Maintains KV-Cache state. |
| **4. Flow LM Flow** | `flow_lm_flow.onnx` | Flow Matcher | Used for iterative sampling. Smaller and faster to run multiple times per step if needed. |
| **5. Mimi Decoder** | `mimi_decoder.onnx` | Latents $\rightarrow$ Audio | The "Vocoder." Runs at the very end to produce the final wave. |

**Why not 1 model?**
If it were 1 model, the engine would have to load a massive graph into memory even if it only needed to run a small part (like the text styling). Splitting them allows the inference engine (like ONNX Runtime) to use less RAM and optimize the execution of the "Generation Loop" (Main + Flow) separately from the one-time tasks (Conditioner/Decoder).

---

## 3. How Input/Output Formats are Decided

The input and output names/shapes are decided by **Wrapper Classes** (found in `onnx_export/wrappers.py`).

1. **Wrappers**: Since the raw PyTorch models often use complex Python dictionaries or objects for state, we create a "Wrapper" class that inherits from `nn.Module`. 
2. **Flattening**: ONNX doesn't like Python dictionaries. The wrappers "flatten" the internal state (KV Caching) into a long list of individual tensors (`state_0`, `state_1`, etc.).
3. **Tracing**: When we call `torch.onnx.export`, we provide:
   - `input_names`: e.g., `["sequence", "text_embeddings"]`
   - `output_names`: e.g., `["conditioning", "eos_logit"]`
   - `dynamic_axes`: Tells ONNX which dimensions can change (e.g., sequence length).

---

## 4. What is Monkeypatching?

**Monkeypatching** is the process of replacing a function or method at runtime with a different one. In this project, we monkeypatch several core parts of the model during export.

### Why do we do it?
PyTorch code is often "un-traceable." Certain Python operations cannot be converted into a static ONNX graph.
1. **Removing `.item()`**: PyTorch often uses `tensor.item()` to get a Python scalar. This breaks the ONNX graph because it forces the GPU to sync with the CPU. Monkeypatching replaces these with pure tensor operations.
2. **Deterministic Randomness**: During export, we replace random number generators (like `randn`) with fixed logic or inputs so the graph becomes predictable.
3. **KV Cache Management**: The original code might update the cache "in-place" (mutating a list). ONNX requires "Functional" updates (returning a new tensor). We patch the Transformer's attention mechanism to handle state passing explicitly.
4. **Stateful $\rightarrow$ Stateless**: We patch `StatefulModule.increment_step` to ensure that state updates are tracked correctly within the ONNX graph as outputs, rather than hidden internal variables.

---

## 5. Parity: How we know it's the same?

To ensure the ONNX model produces the **exact same voice** as the PyTorch original, we perform **Parity Testing**:

1. **Capture PyTorch Output**: We run the original model with a specific seed and input, saving the output tensors.
2. **Run ONNX Output**: We run the exported ONNX model with the same input.
3. **Numerical Comparison**: We use `np.testing.assert_allclose(pt_out, onnx_out, rtol=1e-5)`.
   - `rtol` (Relative Tolerance) ensures that even if there are tiny floating-point differences (due to ONNX optimizations), the result is virtually identical.
4. **State Matching**: We don't just check the final audio; we check every hidden state (KV cache) to make sure the "memory" of the model is being updated identically.

If the results match within a very small threshold (e.g., $0.00001$), we consider the export successful.

---

# Detailed Technical Comparison: PyTorch vs. ONNX

This section provides a deep-dive into the data flow, dimensions, and execution logic for each of the 5 models in the PocketTTS ONNX export.

---

## 1. Text Conditioner (`text_conditioner.onnx`)
**Purpose**: Map discrete tokens (text) to continuous vector space.

| Feature | PyTorch (Python) | ONNX (Graph) |
| :--- | :--- | :--- |
| **Input** | `TokenizedText` object containing `tensor(long)` | `token_ids`: `tensor(int64)` |
| **Input Shape** | `[Batch, SeqLen]` (e.g. `[1, 20]`) | `[Batch, SeqLen]` (Dynamic `SeqLen`) |
| **Logic** | `nn.Embedding(vocab, dim)` | `Gather` node (lookup weights) |
| **Output Dtype** | `float32` (or `bfloat16`/`float16` if cast) | `embeddings`: `float32` |
| **Output Shape** | `[Batch, SeqLen, 1024]` | `[Batch, SeqLen, 1024]` |
| **Differences** | The Python class `LUTConditioner` handles the HuggingFace download and `sentencepiece` tokenization *before* passing to the model. | The ONNX model **only** contains the embedding table and Gather logic. Tokenization must happen in JavaScript/C++. |

---

## 2. Mimi Encoder (`mimi_encoder.onnx`)
**Purpose**: Downsample raw audio into a compressed latent space ($24\text{kHz} \rightarrow 12.5\text{Hz}$).

| Feature | PyTorch (Python) | ONNX (Graph) |
| :--- | :--- | :--- |
| **Input** | `audio`: `tensor(float32)` | `audio`: `tensor(float32)` |
| **Input Shape** | `[Batch, Channels, Time]` (e.g. `[1, 1, 24000]`) | `[Batch, 1, Time]` (Dynamic `Time`) |
| **Logic** | `SEANetEncoder` (Convolutions + ResNet blocks) | `Conv1d`, `Pad`, `Add`, `LeakyRelu` nodes |
| **Output** | `latents`: `tensor(float32)` | `latents`: `tensor(float32)` |
| **Output Shape** | `[Batch, Time/1920, 32]` | `[Batch, Time/1920, 1024]` (after speaker projection) |
| **Projection** | Speaker projection happens separately in Python. | Speaker projection is **fused** into the final layer of the ONNX graph. |

---

## 3. Flow LM Main (`flow_lm_main.onnx`)
**Purpose**: The "Heart" of the system. Processes sequence history and predicts current state.

| Feature | PyTorch (Python) | ONNX (Graph) |
| :--- | :--- | :--- |
| **Main Input** | `sequence`: `tensor(float32)` | `sequence`: `tensor(float32)` |
| **Context Input** | `text_embeddings`: `tensor(float32)` | `text_embeddings`: `tensor(float32)` |
| **State Input** | **Hidden** (internal `dict` in modules) | **Explicit**: `state_0` to `state_N`: `tensor(float32)` |
| **Input Shape** | `seq: [1, 1, 32]`, `text: [1, T, 1024]` | Match PyTorch (Dynamic text/seq length) |
| **Math Logic** | Transformer Attention (KV Cache in RAM) | `MatMul`, `Softmax`, `LayerNorm`, `Gather` (for states) |
| **State Loop** | State is updated "In-Place" (mutating tensors). | Functional: Current states $\rightarrow$ Logic $\rightarrow$ New states. |
| **Output** | `conditioning`, `eos_logit`, `states` | `conditioning`, `eos_logit`, `out_state_0...N` |
| **Differences** | Python uses a `dict` for states. | ONNX uses 100+ individual tensor inputs/outputs (one for each K and V cache per layer). |

---

## 4. Flow LM Flow (`flow_lm_flow.onnx`)
**Purpose**: Iterative Flow Matching (the "diffusion-like" step).

| Feature | PyTorch (Python) | ONNX (Graph) |
| :--- | :--- | :--- |
| **Input `c`** | `conditioning`: `tensor(float32)` | `c`: `tensor(float32)` |
| **Input `s`, `t`** | Scalars (0.0 to 1.0) | `s`, `t`: `tensor(float32)` shape `[1, 1]` |
| **Input `x`** | Noise/Latent: `tensor(float32)` | `x`: `tensor(float32)` shape `[1, 32]` |
| **Logic** | `SimpleMLPAdaLN` (Multi-Layer Perceptron) | High-speed `Linear` and `AdaLN` nodes |
| **Output** | `flow_dir`: `tensor(float32)` | `flow_dir`: `tensor(float32)` |
| **Step Logic** | PyTorch loop runs `lsd_decode` (Python loop). | The loop happens **outside** ONNX (inference engine calls this model `N` times). |

---

## 5. Mimi Decoder (`mimi_decoder.onnx`)
**Purpose**: Convert latents back to high-fidelity audio ($12.5\text{Hz} \rightarrow 24\text{kHz}$).

| Feature | PyTorch (Python) | ONNX (Graph) |
| :--- | :--- | :--- |
| **Input** | `latent`: `tensor(float32)` | `latent`: `tensor(float32)` |
| **State Input** | **Hidden** (stored in `StreamingConv` objects) | **Explicit**: `state_0...N` (Conv buffers) |
| **Input Shape** | `[Batch, 1, 32]` | `[Batch, 1, 32]` |
| **Logic** | ConvTranspose1d (Upsampling) | `ConvTranspose1d` + `Add` nodes |
| **Output** | `audio_frame`: `tensor(float32)` | `audio_frame`: `tensor(float32)` |
| **Output Shape** | `[1, 1, 1920]` (samples) | `[1, 1, 1920]` |
| **Dtype Match** | PyTorch might use dynamic casting. | ONNX graph is **strictly float32** (unless quantized to `int8`). |

---

## Summary of Data Flow (ONNX Runtime)

1. **Preprocessing**: Text $\xrightarrow{Tokenize}$ `[1, T]` IDs.
2. **Step 1**: `text_conditioner.onnx` $\rightarrow$ `embeddings [1, T, 1024]`.
3. **Loop Start**:
   - **Step A**: Input `embeddings` + `prev_latent` + `States` $\rightarrow$ `flow_lm_main.onnx` $\rightarrow$ `conditioning` + `New States`.
   - **Step B**: Input `conditioning` + `Noise` $\xrightarrow{Iterate 1-5 times}$ `flow_lm_flow.onnx` $\rightarrow$ **Generated Latent**.
   - **Step C**: Input **Generated Latent** + `Mimi States` $\rightarrow$ `mimi_decoder.onnx` $\rightarrow$ **Audio Samples**.
4. **Conclusion**: Concatenate Audio Samples $\rightarrow$ **Speaker Out**.

### Key Differences in "How" they calculate:
- **PyTorch**: Flexibility through Python objects. Logic is distributed across `.py` files.
- **ONNX**: Fixed execution. Logic is baked into the model file as a static "Recipe."
- **States**: The biggest difference is how they handle memory. PyTorch "remembers" via object variables; ONNX "remembers" by having the inference engine feed the output of one run back into the input of the next.
