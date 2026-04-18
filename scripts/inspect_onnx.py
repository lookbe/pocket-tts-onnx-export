import onnx
import sys
import os
from pathlib import Path

def get_type_name(onnx_type):
    """Converts ONNX tensor type int to human readable string."""
    from onnx import TensorProto
    type_map = {
        TensorProto.FLOAT: "FLOAT",
        TensorProto.FLOAT16: "FLOAT16",
        TensorProto.INT32: "INT32",
        TensorProto.INT64: "INT64",
        TensorProto.BOOL: "BOOL",
        TensorProto.INT8: "INT8",
        # Add more if needed
    }
    return type_map.get(onnx_type, f"TYPE({onnx_type})")

def inspect_onnx(model_path, log_dir="logs"):
    model_path = Path(model_path)
    if not model_path.exists():
        print(f"Error: File {model_path} not found.")
        return

    print(f"Inspecting {model_path}...")
    model = onnx.load(str(model_path))
    graph = model.graph

    os.makedirs(log_dir, exist_ok=True)
    log_file = Path(log_dir) / f"{model_path.stem}_inspect.txt"
    
    with open(log_file, "w", encoding="utf-8") as f:
        def log_print(msg):
            print(msg)
            f.write(msg + "\n")

        log_print(f"ONNX Model Inspection: {model_path.name}")
        log_print("=" * 60)
        
        log_print("\nINPUTS:")
        log_print("-" * 20)
        for i in graph.input:
            shape = []
            for dim in i.type.tensor_type.shape.dim:
                if dim.HasField("dim_value"):
                    shape.append(str(dim.dim_value))
                elif dim.HasField("dim_param"):
                    shape.append(dim.dim_param)
                else:
                    shape.append("?")
            
            t_type = get_type_name(i.type.tensor_type.elem_type)
            log_print(f"Name: {i.name:<25} | Shape: {str(shape):<20} | Type: {t_type}")

        log_print("\nOUTPUTS:")
        log_print("-" * 20)
        for o in graph.output:
            shape = []
            for dim in o.type.tensor_type.shape.dim:
                if dim.HasField("dim_value"):
                    shape.append(str(dim.dim_value))
                elif dim.HasField("dim_param"):
                    shape.append(dim.dim_param)
                else:
                    shape.append("?")
            
            t_type = get_type_name(o.type.tensor_type.elem_type)
            log_print(f"Name: {o.name:<25} | Shape: {str(shape):<20} | Type: {t_type}")

        log_print("\nINITIALIZERS (Buffers/Weights):")
        log_print("-" * 20)
        for init in graph.initializer:
            log_print(f"Name: {init.name}")

        log_print("=" * 60)

        log_print(f"Log saved to: {log_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/inspect_onnx.py <path_to_onnx_file>")
        sys.exit(1)
    
    inspect_onnx(sys.argv[1])
