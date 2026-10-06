"""Host-side helpers for the KV-delta flow_lm_main contract (see OPTIMIZATION.md).

The exported graph takes a head-major K/V cache [1, H, capacity, D] and returns, for each K/V out_state, ONLY the
new rows [1, H, L, D]. The host must write them into its persistent cache at slots [offset, offset + L), where
`offset` is the layer's step counter *before* the call. Every Python runner/verification script uses these helpers so
they follow exactly the same rule as the C++ host (pocket-tts-cpp, UpdateStateInternal/AppendKvDeltaRows).

Works for both torch tensors and numpy arrays. A state is only ever treated as a delta when it is rank 4 and its
shape differs from the incoming cache (the legacy full-cache contract returns an equal-shaped tensor, which passes
through unchanged, so these helpers are safe for legacy models too).
"""


def is_delta(cur, out):
    return getattr(cur, "ndim", 0) == 4 and getattr(out, "ndim", 0) == 4 and tuple(cur.shape) != tuple(out.shape)


def merge(cur, out, offset):
    """Return the new persistent state: `out` itself, or `cur` with `out`'s rows written at `offset` along the
    capacity axis (axis 2 for the head-major cache)."""
    if not is_delta(cur, out):
        return out
    new = cur.clone() if hasattr(cur, "clone") else cur.copy()
    new[:, :, offset:offset + out.shape[2]] = out
    return new


def merge_onnx_state(prev, outputs, first_state_output=2):
    """prev: {"state_i": array} fed to the session this step; outputs: the session's full output list
    (conditioning, eos_logit, out_state_0, ...). Returns the next {"state_i": array} dict. The step counter that
    pairs with K/V state i is state_(3*(i//3)+2) (layout per layer: cache_k, cache_v, step)."""
    nxt = {}
    # Count states from the inputs: trailing non-state outputs (e.g. the optional `ts_logits`) are not states.
    n_states = sum(1 for k in prev if k.startswith("state_"))
    for i in range(n_states):
        name = f"state_{i}"
        out = outputs[i + first_state_output]
        cur = prev[name]
        if is_delta(cur, out):
            offset = int(prev[f"state_{3 * (i // 3) + 2}"].reshape(-1)[0])
            nxt[name] = merge(cur, out, offset)
        else:
            nxt[name] = out
    return nxt
