import os
import time
import torch
import numpy as np
import onnxruntime as ort
import argparse
from pathlib import Path
from pocket_tts.models.tts_model import TTSModel
from pocket_tts.default_parameters import DEFAULT_VARIANT

class ONNXBenchmarker:
    def __init__(self, model_dir, use_int8=False):
        self.model_dir = Path(model_dir)
        suffix = "_int8" if use_int8 else ""
        
        # Load sessions
        self.flow_main = ort.InferenceSession(str(self.model_dir / f"flow_lm_main{suffix}.onnx"))
        self.flow_net = ort.InferenceSession(str(self.model_dir / f"flow_lm_flow{suffix}.onnx"))
        self.mimi_decoder = ort.InferenceSession(str(self.model_dir / f"mimi_decoder{suffix}.onnx"))
        self.conditioner = ort.InferenceSession(str(self.model_dir / f"text_conditioner.onnx"))
        
        self.is_optimized = "k_cache" in [i.name for i in self.flow_main.get_inputs()]
        print(f"Loaded models from {model_dir} (INT8={use_int8}, Optimized={self.is_optimized})")

    def run_benchmark(self, text_tokens, num_steps=50):
        # 1. Conditioning
        tokens = torch.tensor([text_tokens], dtype=torch.long)
        text_emb = self.conditioner.run(None, {"token_ids": tokens.numpy()})[0]
        
        # 2. AR Loop (FlowLM)
        seq = np.full((1, 1, 32), 0.0, dtype=np.float32)
        
        # States Initialization
        inputs = {"sequence": seq, "text_embeddings": text_emb}
        if self.is_optimized:
            # Handle list/tuple and dynamic shapes
            k_input = self.flow_main.get_inputs()[2]
            k_shape = [s if isinstance(s, int) else (0 if i == 3 else 1) for i, s in enumerate(k_input.shape)]
            v_input = self.flow_main.get_inputs()[3]
            v_shape = [s if isinstance(s, int) else (0 if i == 3 else 1) for i, s in enumerate(v_input.shape)]
            
            inputs["k_cache"] = np.zeros(k_shape, dtype=np.float32)
            inputs["v_cache"] = np.zeros(v_shape, dtype=np.float32)
            inputs["step"] = np.array([0], dtype=np.int64)
        else:
            for i in range(2, len(self.flow_main.get_inputs())):
                name = self.flow_main.get_inputs()[i].name
                shape = [s if isinstance(s, int) else (0 if "seq" in name or "cache" in name else 1) for s in self.flow_main.get_inputs()[i].shape]
                inputs[name] = np.zeros(shape, dtype=np.float32)

        latents = []
        flow_start = time.time()
        
        for i in range(num_steps):
            main_out = self.flow_main.run(None, inputs)
            cond = main_out[0]
            
            # Update inputs for next step
            if self.is_optimized:
                inputs["k_cache"] = main_out[2]
                inputs["v_cache"] = main_out[3]
                inputs["step"] = main_out[4]
            else:
                for j in range(2, len(main_out)):
                    in_name = self.flow_main.get_inputs()[j].name
                    inputs[in_name] = main_out[j]
            
            # Flow step
            noise = np.zeros((1, 32), dtype=np.float32)
            s, t = np.array([[0.0]], dtype=np.float32), np.array([[1.0]], dtype=np.float32)
            latent = self.flow_net.run(None, {"c": cond, "s": s, "t": t, "x": noise})[0]
            latents.append(latent)
            inputs["sequence"] = latent.reshape(1, 1, 32)
            
        flow_ms = (time.time() - flow_start) / num_steps * 1000
        
        # 3. Mimi Decode
        m_k_input = self.mimi_decoder.get_inputs()[1]
        m_k_shape = [s if isinstance(s, int) else (0 if i == 3 else 1) for i, s in enumerate(m_k_input.shape)]
        m_v_input = self.mimi_decoder.get_inputs()[2]
        m_v_shape = [s if isinstance(s, int) else (0 if i == 3 else 1) for i, s in enumerate(m_v_input.shape)]
        
        m_k = np.zeros(m_k_shape, dtype=np.float32)
        m_v = np.zeros(m_v_shape, dtype=np.float32)
        
        conv_states = {}
        for i in range(3, len(self.mimi_decoder.get_inputs())):
            inp = self.mimi_decoder.get_inputs()[i]
            name = inp.name
            shape = [s if isinstance(s, int) else 1 for s in inp.shape]
            # Match data type
            dtype = np.float32
            if "bool" in inp.type:
                dtype = bool
            elif "int64" in inp.type:
                dtype = np.int64
                
            conv_states[name] = np.zeros(shape, dtype=dtype)
            
        decode_start = time.time()
        for l in latents:
            mimi_run_inputs = {"latent": l.reshape(1, 1, 32), "k_cache": m_k, "v_cache": m_v}
            mimi_run_inputs.update(conv_states)
            
            m_out = self.mimi_decoder.run(None, mimi_run_inputs)
            m_k, m_v = m_out[1:3]
            for i, name in enumerate(conv_states.keys()):
                # Cast back to correct dtype if needed
                out_val = m_out[3+i]
                if conv_states[name].dtype == bool:
                    conv_states[name] = out_val.astype(bool)
                elif conv_states[name].dtype == np.int64:
                    conv_states[name] = out_val.astype(np.int64)
                else:
                    conv_states[name] = out_val
        
        mimi_ms = (time.time() - decode_start) / num_steps * 1000 if latents else 0
        
        self.partial_results = (flow_ms, mimi_ms)
        return flow_ms, mimi_ms

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="onnx")
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    
    # Mock tokens for "Hello world"
    tokens = [1, 2, 3, 4, 5]
    
    print("--- PocketTTS ONNX Benchmark ---")
    
    # Benchmark FP32 (Optimized Graph)
    f_ms, m_ms = 0, 0
    try:
        bench_fp32 = ONNXBenchmarker(args.dir, use_int8=False)
        print(f"Benchmarking FP32 {args.dir}...")
        f_ms, m_ms = bench_fp32.run_benchmark(tokens, args.steps)
        print(f"FP32 Result:\n  FlowLM: {f_ms:.2f} ms/step\n  Mimi:   {m_ms:.2f} ms/step")
    except Exception as e:
        print(f"FP32 Benchmark failed or partially failed: {e}")

    # Benchmark INT8 (Optimized Graph + Aggressive Quant)
    f8_ms, m8_ms = 0, 0
    try:
        bench_int8 = ONNXBenchmarker(args.dir, use_int8=True)
        print(f"Benchmarking INT8 {args.dir}...")
        f8_ms, m8_ms = bench_int8.run_benchmark(tokens, args.steps)
        print(f"INT8 Result:\n  FlowLM: {f8_ms:.2f} ms/step\n  Mimi:   {m8_ms:.2f} ms/step")
    except Exception as e:
        print(f"INT8 Benchmark failed or partially failed: {e}")
        # Try to capture partial results if session was created
        if 'bench_int8' in locals() and hasattr(bench_int8, 'partial_results'):
             f8_ms, m8_ms = bench_int8.partial_results
             print(f"INT8 Partial Result (before failure):\n  FlowLM: {f8_ms:.2f} ms/step\n  Mimi:   {m8_ms:.2f} ms/step")

    if f_ms > 0 and f8_ms > 0:
        gain = (f_ms - f8_ms) / f_ms * 100
        print(f"\nOverall FlowLM Gain: {gain:.1f}%")

if __name__ == "__main__":
    main()
