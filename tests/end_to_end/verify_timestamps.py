"""Verify the word-timestamp output (`ts_logits`) of an exported flow_lm_main against pocket-tts-timestamped.

What is checked, for each --onnx_dir given (Python-exported and/or C++-exported, same weights):

  1. Contract: `ts_logits` is the LAST output of flow_lm_main, shaped [n_heads, valid_len], and every other output keeps
     its position (checked against --plain_dir, an export of the same weights WITHOUT --timestamp_heads: the
     conditioning / eos / state outputs must be identical, and the plain graph has exactly one output less).
  2. Signal: one utterance is generated with ONNX alone (voice prompt -> text prompt -> AR loop, fixed seed). The same
     latents are then teacher-forced through the PyTorch reference model with its own SelectedAttentionCapture
     (pocket_tts_timestamped), and the per-frame text-unit scores are compared with the ones derived from the ONNX
     `ts_logits` (softmax over the text-key slice, head mean, token->unit mapping).
  3. Result: the same WordAlignment state machine is run over both score streams; the word start/end times must agree.

When several --onnx_dir are given, the first one generates the latents and the others are teacher-forced with them,
so their ts_logits are also compared to each other directly.

Needs the pocket-tts-timestamped checkout (--ref_repo) and the HF cache for its `english_2026-09` config.
"""
import argparse
import copy
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
from scipy.signal import resample_poly

ort.set_default_logger_severity(3)
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REF_REPO = ROOT.parent / "pocket-tts-timestamped"
DEFAULT_WAV = ROOT.parent / "default.wav"

NP_DTYPE = {"tensor(float)": np.float32, "tensor(float16)": np.float16, "tensor(int64)": np.int64}


class MainRunner:
    """Drives flow_lm_main.onnx positionally (works for both the Python and the C++ exporter's state names)."""

    def __init__(self, onnx_dir):
        self.dir = Path(onnx_dir)
        self.s = ort.InferenceSession(str(self.dir / "flow_lm_main.onnx"), providers=["CPUExecutionProvider"])
        ins, outs = self.s.get_inputs(), self.s.get_outputs()
        self.seq_name, self.text_name = ins[0].name, ins[1].name
        self.state_inputs = ins[2:]
        self.n = len(self.state_inputs)
        assert self.n % 3 == 0, "expected (cache_k, cache_v, step) per layer"
        self.has_ts = outs[-1].name == "ts_logits"
        self.out_names = [o.name for o in outs]
        assert len(outs) == 2 + self.n + int(self.has_ts), f"unexpected output count {len(outs)}"
        self.flow = ort.InferenceSession(str(self.dir / "flow_lm_flow.onnx"), providers=["CPUExecutionProvider"])

    def init_state(self):
        return [np.zeros(i.shape, dtype=NP_DTYPE[i.type]) for i in self.state_inputs]

    def step(self, seq, text_emb, state):
        feed = {self.seq_name: seq, self.text_name: text_emb}
        for i, st in zip(self.state_inputs, state):
            feed[i.name] = st
        o = self.s.run(None, feed)
        new = []
        for j in range(self.n // 3):
            step_in = int(state[3 * j + 2].reshape(-1)[0])
            for q in (0, 1):
                cur, out = state[3 * j + q], o[2 + 3 * j + q]
                if out.shape == cur.shape:  # legacy full-cache contract
                    new.append(out)
                else:  # KV-delta contract: rows-only output, written at [step_in, step_in + L)
                    merged = cur.copy()
                    merged[:, :, step_in:step_in + out.shape[2]] = out
                    new.append(merged)
            new.append(o[2 + 3 * j + 2])
        ts = o[2 + self.n] if self.has_ts else None
        return o[0], o[1], new, ts, o

    def sample(self, c, noise, steps=1):
        x = noise
        dt = 1.0 / steps
        for i in range(steps):
            s = np.array([[i / steps]], dtype=np.float32)
            t = np.array([[i / steps + dt]], dtype=np.float32)
            x = x + self.flow.run(None, {"c": c, "s": s, "t": t, "x": x})[0] * dt
        return x.astype(np.float32)


def load_voice(path, seconds):
    audio, sr = sf.read(str(path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 24000:
        audio = resample_poly(audio, 24000, sr)
    audio = audio[: int(seconds * 24000)].astype(np.float32)
    return torch.from_numpy(audio)[None]  # [1, N]


def prepare_context(runner, audio_t, bos, token_ids):
    """voice prompt, then text prompt, through flow_lm_main. Returns state after both and text_start."""
    enc = ort.InferenceSession(str(runner.dir / "mimi_encoder.onnx"), providers=["CPUExecutionProvider"])
    cond = ort.InferenceSession(str(runner.dir / "text_conditioner.onnx"), providers=["CPUExecutionProvider"])
    voice = enc.run(None, {"audio": audio_t.numpy()[None]})[0]  # [1, T, 1024]
    if bos is not None:
        voice = np.concatenate([bos.reshape(1, 1, -1).astype(np.float32), voice], axis=1)
    empty_seq = np.zeros((1, 0, 32), np.float32)
    state = runner.init_state()
    _, _, state, _, _ = runner.step(empty_seq, voice, state)
    text_start = int(state[2].reshape(-1)[0])
    text_emb = cond.run(None, {"token_ids": token_ids.astype(np.int64)})[0]
    _, _, state, _, _ = runner.step(empty_seq, text_emb, state)
    return state, text_start


def generate_onnx(runner, state, max_gen_len, frames_after_eos, temp, seed, steps, eos_threshold=-4.0):
    rng = np.random.default_rng(seed)
    empty_text = np.zeros((1, 0, 1024), np.float32)
    seq = np.full((1, 1, 32), np.nan, np.float32)
    latents, ts_rows, eos_step = [], [], None
    for k in range(max_gen_len):
        c, eos, state, ts, _ = runner.step(seq, empty_text, state)
        if float(eos.reshape(-1)[0]) > eos_threshold and eos_step is None:
            eos_step = k
        if eos_step is not None and k >= eos_step + frames_after_eos:
            break
        noise = (rng.standard_normal((1, 32)) * math.sqrt(temp)).astype(np.float32)
        latent = runner.sample(c, noise, steps)
        latents.append(latent)
        ts_rows.append(ts)
        seq = latent.reshape(1, 1, 32)
    return latents, ts_rows


def replay_onnx(runner, state, latents):
    """Teacher-force given latents; returns the ts_logits of every frame."""
    empty_text = np.zeros((1, 0, 1024), np.float32)
    seq = np.full((1, 1, 32), np.nan, np.float32)
    rows = []
    for latent in latents:
        _, _, state, ts, _ = runner.step(seq, empty_text, state)
        rows.append(ts)
        seq = latent.reshape(1, 1, 32)
    return rows


def onnx_unit_scores(ts_rows, text_start, token_count, token_to_unit):
    out = []
    for ts in ts_rows:
        seg = ts[:, text_start:text_start + token_count].astype(np.float64)
        seg = np.exp(seg - seg.max(axis=-1, keepdims=True))
        att = (seg / seg.sum(axis=-1, keepdims=True)).mean(axis=0)  # head mean (reference: sum then / n_heads)
        out.append(att @ token_to_unit.numpy().astype(np.float64))
    return np.stack(out)


def words_from_scores(alignment_cls, units, scores, voiced, frame_seconds):
    al = alignment_cls(units)
    words = []
    for k, (sc, v) in enumerate(zip(scores, voiced)):
        for ev in al.process_frame(torch.from_numpy(np.asarray(sc, dtype=np.float32)), v, k * frame_seconds):
            if type(ev).__name__ == "WordEnd":
                words.append((ev.word, round(ev.start_time, 4), round(ev.end_time, 4)))
    for ev in al.finish(len(scores) * frame_seconds):
        words.append((ev.word, round(ev.start_time, 4), round(ev.end_time, 4)))
    return words


def check_contract(ts_dir, plain_dir, audio_t, token_ids):
    a, b = MainRunner(ts_dir), MainRunner(plain_dir)
    assert a.has_ts and not b.has_ts, "first dir must have ts_logits, --plain_dir must not"
    # The C++ exporter's un-renamed state outputs carry auto-generated node names (Cast_NNN / Reshape_NNN) whose counter
    # shifts when nodes are added; only the stable names (conditioning, eos_logit, out_state_N) must match.
    norm = lambda names: [re.sub(r"^(Cast|Reshape)_\d+$", "<auto>", n) for n in names]
    assert norm(a.out_names[:-1]) == norm(b.out_names), (
        "outputs other than ts_logits changed:\n%s\n%s" % (a.out_names, b.out_names)
    )
    assert [i.name for i in a.s.get_inputs()] == [i.name for i in b.s.get_inputs()], "inputs changed"
    bos = np.load(Path(ts_dir) / "bos_before_voice.npy")
    sa, _ = prepare_context(a, audio_t, bos, token_ids)
    sb, _ = prepare_context(b, audio_t, bos, token_ids)
    seq = np.full((1, 1, 32), np.nan, np.float32)
    empty_text = np.zeros((1, 0, 1024), np.float32)
    ca, ea, _, ts, oa = a.step(seq, empty_text, sa)
    cb, eb, _, _, ob = b.step(seq, empty_text, sb)
    worst = max(float(np.abs(x.astype(np.float64) - y.astype(np.float64)).max()) for x, y in zip(oa[:-1], ob))
    print(f"  contract [{Path(ts_dir).name} vs {Path(plain_dir).name}]: {len(oa)} vs {len(ob)} outputs, "
          f"ts_logits {ts.shape}, max |diff| over all other outputs = {worst:.3e}")
    assert worst < 1e-5, "default outputs changed when ts_logits was added"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx_dir", action="append", required=True, help="export WITH timestamp heads (repeatable)")
    ap.add_argument("--plain_dir", action="append", default=[], help="export of the same weights WITHOUT timestamps (repeatable, pairs by order)")
    ap.add_argument("--ref_repo", default=str(DEFAULT_REF_REPO))
    ap.add_argument("--ref_language", default="english_2026-09")
    ap.add_argument("--audio", default=str(DEFAULT_WAV))
    ap.add_argument("--voice_seconds", type=float, default=6.0)
    ap.add_argument("--text", default="The quick brown fox jumps over the lazy dog, and then it takes a short nap.")
    ap.add_argument("--heads", default=None, help="LAYER:HEAD[,...] the exports were built with, if not the reference config's (row order matters)")
    ap.add_argument("--dump_fixture", default=None, help="append an ALIGN case (raw ts_logits, voiced flags, reference words) for pocket-tts-cpp/tests/test_word_timestamps.cpp")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--score_atol", type=float, default=2e-2)
    ap.add_argument("--time_tol", type=float, default=0.081, help="allowed word start/end difference in seconds (1 frame = 0.08)")
    args = ap.parse_args()

    sys.path.insert(0, args.ref_repo)
    from pocket_tts_timestamped.models.text_chunking import prepare_text_prompt, split_into_best_sentences
    from pocket_tts_timestamped.models.tts_model import TTSModel
    from pocket_tts_timestamped.default_parameters import MAX_TOKEN_PER_CHUNK
    from pocket_tts_timestamped.modules.stateful_module import increment_steps, init_states
    from pocket_tts_timestamped.timestamps import SelectedAttentionCapture, WordAlignment, is_voiced
    from pocket_tts_timestamped.timestamps.text import _iter_timestamp_text_chunks

    torch.set_grad_enabled(False)
    ref = TTSModel.load_model(language=args.ref_language).eval()
    heads = [(h.layer, h.head) for h in ref.config.timestamp_heads]
    if args.heads:
        heads = [tuple(int(v) for v in item.split(":")) for item in args.heads.split(",")]
    print(f"reference: {args.ref_language}, timestamp heads {heads}, temp {ref.temp}, decode steps {ref.sampler_decode_steps}")

    chunks = split_into_best_sentences(
        ref.flow_lm.conditioner.tokenizer, args.text, MAX_TOKEN_PER_CHUNK, ref.pad_with_spaces_for_short_inputs,
        remove_semicolons=ref.remove_semicolons, append_terminal_punctuation=ref.append_terminal_punctuation)
    assert len(chunks) == 1, "use a text that fits one chunk"
    tchunk = list(_iter_timestamp_text_chunks(args.text, chunks, ref.flow_lm.conditioner.tokenizer.sp, best_effort=True))[0]
    token_ids = tchunk.prepared_tokens.numpy()
    T = token_ids.shape[1]
    _, guess = prepare_text_prompt(tchunk.text, ref.pad_with_spaces_for_short_inputs, ref.remove_semicolons, ref.append_terminal_punctuation)
    frames_after_eos = ref.model_recommended_frames_after_eos if ref.model_recommended_frames_after_eos is not None else guess + 2
    max_gen_len = ref._estimate_max_gen_len(T)
    print(f"text: {tchunk.text!r} -> {T} tokens, {len(tchunk.words)} words, frames_after_eos {frames_after_eos}")

    audio_t = load_voice(args.audio, args.voice_seconds)

    print("\n[1] contract")
    for i, d in enumerate(args.onnx_dir):
        if i < len(args.plain_dir):
            check_contract(d, args.plain_dir[i], audio_t, token_ids)
    if not args.plain_dir:
        print("  (no --plain_dir given, skipped)")

    runners = [MainRunner(d) for d in args.onnx_dir]
    for r in runners:
        assert r.has_ts, f"{r.dir} has no ts_logits output"
        assert r.s.get_outputs()[-1].shape[0] == len(heads), (r.s.get_outputs()[-1].shape, heads)

    print("\n[2] ONNX generation (+ replay on the other exports)")
    contexts = []
    for r in runners:
        bos = np.load(r.dir / "bos_before_voice.npy") if ref.flow_lm.insert_bos_before_voice else None
        contexts.append(prepare_context(r, audio_t, bos, token_ids))
    text_start = contexts[0][1]
    assert all(c[1] == text_start for c in contexts)
    latents, ts_rows = generate_onnx(runners[0], contexts[0][0], max_gen_len, frames_after_eos, ref.temp, args.seed, ref.sampler_decode_steps)
    n = len(latents)
    print(f"  {runners[0].dir.name}: {n} frames ({n * 0.08:.2f}s), text keys [{text_start}, {text_start + T}), ts_logits rows {ts_rows[0].shape} .. {ts_rows[-1].shape}")
    all_rows = [ts_rows] + [replay_onnx(r, c[0], latents) for r, c in zip(runners[1:], contexts[1:])]
    for r, rows in zip(runners[1:], all_rows[1:]):
        d = max(float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max()) for a, b in zip(ts_rows, rows))
        print(f"  ts_logits {runners[0].dir.name} vs {r.dir.name}: max |diff| = {d:.3e}")

    print("\n[3] PyTorch reference (teacher-forced on the same latents)")
    state = ref.get_state_for_audio_prompt(audio_t)
    ref_text_start = ref._flow_lm_current_end(state)
    assert ref_text_start == text_start, f"text_start differs: torch {ref_text_start} vs onnx {text_start}"
    capture = SelectedAttentionCapture(heads, text_start, text_start + T, tchunk.token_to_unit)
    ref._expand_kv_cache(state, text_start + T + max_gen_len + 2)
    ref._run_flow_lm_and_increment_step(state, text_tokens=torch.from_numpy(token_ids))
    ref_scores, seq = [], torch.full((1, 1, 32), float("nan"))
    for latent in latents:
        capture.begin_frame()
        ref._run_flow_lm_and_increment_step(state, backbone_input_latents=seq, attention_capture=capture)
        ref_scores.append(capture.finish_frame().numpy())
        seq = torch.from_numpy(latent).reshape(1, 1, 32)
    ref_scores = np.stack(ref_scores)

    # frame audio (voiced flags), reference Mimi on the same latents
    mimi_steps = int(ref.mimi.encoder_frame_rate / ref.mimi.frame_rate)
    mstate = init_states(ref.mimi, batch_size=1, sequence_length=max_gen_len * mimi_steps)
    voiced, total = [], 0
    for latent in latents:
        x = torch.from_numpy(latent).reshape(1, 1, 32) * ref.flow_lm.emb_std + ref.flow_lm.emb_mean
        frame = ref.mimi.decode_from_latent(x, mstate)
        increment_steps(ref.mimi, mstate, increment=mimi_steps)
        voiced.append(is_voiced(frame[0, 0]))
        total += frame.shape[-1]
    frame_seconds = total / n / ref.sample_rate
    print(f"  {n} frames, {sum(voiced)} voiced, {frame_seconds * 1000:.0f} ms/frame")

    ok = True
    print("\n[4] comparison")
    ref_words = words_from_scores(WordAlignment, tchunk.units, ref_scores, voiced, frame_seconds)
    for r, rows in zip(runners, all_rows):
        sc = onnx_unit_scores(rows, text_start, T, tchunk.token_to_unit)
        diff = np.abs(sc - ref_scores)
        argmax_agree = float(np.mean(sc.argmax(1) == ref_scores.argmax(1)))
        words = words_from_scores(WordAlignment, tchunk.units, sc, voiced, frame_seconds)
        print(f"  [{r.dir.name}] unit scores: max |diff| {diff.max():.3e}, mean {diff.mean():.3e}, argmax-unit agreement {argmax_agree:.1%}")
        same_words = [w[0] for w in words] == [w[0] for w in ref_words]
        worst_t = max((max(abs(a[1] - b[1]), abs(a[2] - b[2])) for a, b in zip(words, ref_words)), default=0.0)
        print(f"  [{r.dir.name}] words: {len(words)} vs reference {len(ref_words)}, same sequence {same_words}, worst start/end diff {worst_t * 1000:.0f} ms")
        ok &= bool(diff.max() < args.score_atol and same_words and len(words) == len(ref_words) and worst_t <= args.time_tol)
        if r is runners[0]:
            print("\n  word            onnx start-end (s)      torch start-end (s)")
            for a, b in zip(words, ref_words):
                print(f"  {a[0]:<15} {a[1]:6.2f} - {a[2]:6.2f}        {b[1]:6.2f} - {b[2]:6.2f}")
            print()
    if args.dump_fixture:
        from dump_ts_fixture import write_align_case
        write_align_case(args.dump_fixture, tchunk, ref.flow_lm.conditioner.tokenizer.sp, text_start, frame_seconds,
                         all_rows[0], voiced, ref_words)
        print("fixture appended to", args.dump_fixture)
    print("RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
