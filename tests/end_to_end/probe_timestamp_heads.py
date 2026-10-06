"""Check that a configured timestamp head really tracks the text on THIS checkpoint (tokenizer-agnostic).

verify_timestamps.py needs the pocket-tts-timestamped reference model, which only exists for its own checkpoints. The
ungated language checkpoints are other weight revisions with another tokenizer, so there the README head is checked
by what makes it a timestamp head: while speaking, its text attention moves monotonically through the text tokens.

For one language, with an export that has `ts_logits` (rows = the configured head(s) FIRST, then optional control
heads), generate one utterance with ONNX only and print, per row:
  spearman  rank correlation between frame index and the attention-peak token index (1.0 = perfectly monotone)
  span      fraction of the text covered by the peak positions
A real timestamp head scores ~0.95+; an arbitrary head typically does not.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from verify_timestamps import MainRunner, generate_onnx, load_voice, prepare_context  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)).astype(float), np.argsort(np.argsort(b)).astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx_dir", required=True)
    ap.add_argument("--language", required=True, help="config name in pocket_tts/config (german, spanish, ...)")
    ap.add_argument("--heads", required=True, help="LAYER:HEAD list the export was built with, in row order")
    ap.add_argument("--n_configured", type=int, default=None, help="how many leading rows are the configured heads (rest = controls)")
    ap.add_argument("--audio", default=str(ROOT.parent / "default.wav"))
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    from pocket_tts.default_parameters import DEFAULT_TEXT_FOR_LANGUAGE, MAX_TOKEN_PER_CHUNK
    from pocket_tts.models.text_chunking import prepare_text_prompt, split_into_best_sentences
    from pocket_tts.models.tts_model import TTSModel

    heads = [tuple(int(v) for v in h.split(":")) for h in args.heads.split(",")]
    n_cfg = args.n_configured or len(heads)

    cfg_path = ROOT / "pocket_tts" / "config" / f"{args.language}.yaml"
    cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
    cfg["weights_path"] = str((ROOT / "models" / args.language / "model.safetensors").resolve())
    cfg.pop("weights_path_without_voice_cloning", None)
    cfg.pop("timestamp_heads", None)
    tmp = Path(args.onnx_dir) / "_probe_config.yaml"
    yaml.safe_dump(cfg, open(tmp, "w", encoding="utf-8"), allow_unicode=True)
    torch.set_grad_enabled(False)
    tts = TTSModel.load_model(config=str(tmp)).eval()

    text = DEFAULT_TEXT_FOR_LANGUAGE.get(args.language, DEFAULT_TEXT_FOR_LANGUAGE["english"])
    chunk = split_into_best_sentences(
        tts.flow_lm.conditioner.tokenizer, text, MAX_TOKEN_PER_CHUNK, tts.pad_with_spaces_for_short_inputs,
        remove_semicolons=tts.remove_semicolons, append_terminal_punctuation=tts.append_terminal_punctuation,
        capitalize_first_letter=tts.capitalize_first_letter, replace_characters=tts.replace_characters)[0]
    prepared, guess = prepare_text_prompt(
        chunk, tts.pad_with_spaces_for_short_inputs, tts.remove_semicolons, tts.append_terminal_punctuation,
        tts.capitalize_first_letter, tts.replace_characters)
    ids = tts.flow_lm.conditioner.prepare(prepared).numpy()
    T = ids.shape[1]
    frames_after_eos = tts.model_recommended_frames_after_eos if tts.model_recommended_frames_after_eos is not None else guess + 2

    runner = MainRunner(args.onnx_dir)
    assert runner.s.get_outputs()[-1].shape[0] == len(heads)
    bos = np.load(Path(args.onnx_dir) / "bos_before_voice.npy") if tts.flow_lm.insert_bos_before_voice else None
    state, text_start = prepare_context(runner, load_voice(args.audio, 6.0), bos, ids)
    latents, rows = generate_onnx(runner, state, tts._estimate_max_gen_len(T), frames_after_eos,
                                  tts.temp, args.seed, tts.sampler_decode_steps)
    n = len(latents)
    print(f"{args.language}: {prepared[:70]!r}  tokens={T} frames={n}")
    for r, (layer, head) in enumerate(heads):
        peaks = []
        for ts in rows:
            seg = ts[r, text_start:text_start + T].astype(np.float64)
            peaks.append(int(seg.argmax()))
        rho = spearman(np.arange(n), peaks)
        span = (max(peaks) - min(peaks)) / max(T - 1, 1)
        tag = "CONFIGURED" if r < n_cfg else "control"
        print(f"  L{layer}H{head:<2} {tag:<10} spearman {rho:+.3f}  span {span:.2f}")


if __name__ == "__main__":
    main()
