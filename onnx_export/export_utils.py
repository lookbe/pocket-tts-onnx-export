import os
import torch


def save_onnx_separated(model_path: str):
    """
    Rewrite an ONNX file in-place so its weights are stored in a companion
    `<filename>.data` file instead of being embedded in the .onnx protobuf.
    """
    import onnx

    data_filename = os.path.basename(model_path) + ".data"
    model = onnx.load(model_path, load_external_data=True)
    onnx.save_model(
        model,
        model_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_filename,
        size_threshold=0,
        convert_attribute=False,
    )


def cast_weights_to_float16(model, op_types=("Conv", "ConvTranspose")):
    """
    Store the weight/bias initializers of the given op types as float16 on disk, each fed
    into its consuming node through an inserted `Cast(to=FLOAT)` node.

    This halves those initializers' size WITHOUT changing any node's runtime dtype: every op
    still computes in float32 exactly as before, so there is no fp16-kernel/type-propagation
    risk (unlike converting the whole model, which can break ops like Range/rope that don't
    support float16, or leave a blocked node's neighbors mismatched). The only change is a
    fp16 rounding of the affected weights themselves. Returns the number of tensors cast.
    """
    import onnx
    import onnx.numpy_helper
    import numpy as np

    init_map = {i.name: i for i in model.graph.initializer}
    new_nodes = []
    new_inits = []
    replaced_names = set()
    cast_count = 0

    for node in model.graph.node:
        if node.op_type not in op_types:
            continue
        # Conv/ConvTranspose: inputs are (X, W[, B]); cast W and, if present, B.
        for inp_idx in (1, 2):
            if inp_idx >= len(node.input):
                continue
            name = node.input[inp_idx]
            if not name or name not in init_map or name in replaced_names:
                continue
            init = init_map[name]
            arr = onnx.numpy_helper.to_array(init).astype(np.float16)
            fp16_name = name + "__fp16"
            new_inits.append(onnx.numpy_helper.from_array(arr, name=fp16_name))

            cast_out = name + "__fp32_cast"
            new_nodes.append(onnx.helper.make_node(
                "Cast", [fp16_name], [cast_out],
                name=name + "_cast_fp32", to=onnx.TensorProto.FLOAT,
            ))
            replaced_names.add(name)
            cast_count += 1

    # Rewire every consumer of a replaced initializer to read the Cast output instead.
    for node in model.graph.node:
        for i, inp in enumerate(node.input):
            if inp in replaced_names:
                node.input[i] = inp + "__fp32_cast"

    kept_inits = [i for i in model.graph.initializer if i.name not in replaced_names]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept_inits)
    model.graph.initializer.extend(new_inits)
    model.graph.node.extend(new_nodes)

    return cast_count


def convert_gemm_to_matmul_add(model):
    """
    Rewrite eligible `Gemm(A, W, bias; alpha=1, beta=1, transA=0, transB=1)` nodes into
    `MatMul(A, W^T) + Add(bias)`.

    onnxruntime's MatMulNBitsQuantizer (weight-only int4/int8) only recognizes MatMul/Gather
    nodes, not Gemm -- so a Gemm-heavy graph (e.g. torch.onnx's legacy exporter turns every
    nn.Linear into a Gemm) would pass through int4 quantization almost untouched. This performs
    the standard nn.Linear-as-Gemm -> nn.Linear-as-MatMul+Add rewrite so those weights become
    quantizable. Returns the number of nodes converted.
    """
    import onnx
    import onnx.numpy_helper

    init_map = {i.name: i for i in model.graph.initializer}
    new_nodes = []
    new_inits = []
    converted = 0

    for node in list(model.graph.node):
        if node.op_type != "Gemm":
            new_nodes.append(node)
            continue

        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
        transA = attrs.get("transA", 0)
        transB = attrs.get("transB", 0)
        alpha = attrs.get("alpha", 1.0)
        beta = attrs.get("beta", 1.0)

        eligible = (
            transA == 0
            and transB == 1
            and alpha == 1.0
            and beta == 1.0
            and len(node.input) == 3
            and node.input[1] in init_map
        )
        if not eligible:
            new_nodes.append(node)
            continue

        w_init = init_map[node.input[1]]
        w = onnx.numpy_helper.to_array(w_init).T.copy()  # [out, in] -> [in, out]
        w_t_name = w_init.name + "__T"
        new_inits.append(onnx.numpy_helper.from_array(w, name=w_t_name))

        matmul_out = node.output[0] + "__matmul"
        new_nodes.append(onnx.helper.make_node("MatMul", [node.input[0], w_t_name], [matmul_out], name=node.name + "_matmul"))
        new_nodes.append(onnx.helper.make_node("Add", [matmul_out, node.input[2]], [node.output[0]], name=node.name + "_add"))
        converted += 1

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    model.graph.initializer.extend(new_inits)

    # Drop now-unreferenced initializers (the original, non-transposed Gemm weights).
    referenced = set()
    for node in model.graph.node:
        referenced.update(node.input)
    referenced.update(o.name for o in model.graph.output)
    kept = [i for i in model.graph.initializer if i.name in referenced]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)

    return converted


def flatten_state(state):
    """
    Flattens a nested dictionary state into a list of tensors.
    """
    flat = []
    
    # Sort keys to ensure deterministic order
    for key in sorted(state.keys()):
        value = state[key]
        if isinstance(value, dict):
            flat.extend(flatten_state(value))
        elif isinstance(value, torch.Tensor):
            flat.append(value)
        else:
            # Skip non-tensor values or handle them if necessary
            pass
            
    return flat

def unflatten_state(flat_list, structure):
    """
    Reconstructs the nested dictionary state from a flat list of tensors.
    'structure' should be a dictionary reflecting the structure of the state,
    where leaf nodes can be anything (shapes, dummy tensors, etc.)
    """
    state = {}
    idx = 0
    
    for key in sorted(structure.keys()):
        value = structure[key]
        if isinstance(value, dict):
            sub_state, consumed = unflatten_state(flat_list[idx:], value)
            state[key] = sub_state
            idx += consumed
        else:
            # Assuming leaf node corresponds to a tensor in flat_list
            # Clone to ensure we don't modify the input tensor in-place, which ONNX dislikes for inputs
            state[key] = flat_list[idx].clone()
            idx += 1
            
    return state, idx

def get_state_structure(state):
    """
    Returns the structure of the state dict (useful for unflattening).
    """
    structure = {}
    for key in sorted(state.keys()):
        value = state[key]
        if isinstance(value, dict):
            structure[key] = get_state_structure(value)
        else:
            structure[key] = "tensor" # Placeholder
    return structure
