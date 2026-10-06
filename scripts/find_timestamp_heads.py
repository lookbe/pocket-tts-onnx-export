"""Find (and optionally enable + end-to-end verify) the word-timestamp attention head(s) of a language checkpoint.

Timestamp heads are checkpoint specific: the head that tracks the text on one weight revision is arbitrary on another.
A real timestamp head is found by what it does: while speaking, its text attention peak moves monotonically through the
text tokens. This tool searches every layer x head of a language, end to end:

  1. make sure the weights exist (downloaded from the config YAML if missing),
  2. export mimi/conditioner + flow_lm_main with `ts_logits` for ALL heads into a work dir (reused unless --fresh),
  3. generate with ONNX only and score every head per seed: score = spearman(frame, peak token) * span of the text covered
     (probe_timestamp_heads.py); a head's score is its worst over --seeds,
  4. print the ranking; --write_yaml writes the best --pick heads into pocket_tts/config/<lang>.yaml,
  5. --e2e: exports a single-head model, INT8-quantizes it and runs the pocket-tts-cpp `pocket-tts-ts-cli` on it
     (words reported in order, speech energy inside words > between words). Tokenizer: for configs that use
     `tokenizer.json` the runner needs the SentencePiece `tokenizer.model`, which sits next to it (same path, .json -> .model).

Run from the repo root (pocket-tts-onnx-export) with its venv:
    python scripts/find_timestamp_heads.py --lang german --write_yaml --e2e
"""
import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CONFIG_DIR = ROOT / "pocket_tts" / "config"
PROBE = ROOT / "tests" / "end_to_end" / "probe_timestamp_heads.py"
DEFAULT_CLI = ROOT.parent / "pocket-tts-cpp" / "build" / "Release" / "pocket-tts-ts-cli.exe"
LINE = re.compile(r"^\s+L(\d+)H(\d+)\s+\S+\s+spearman\s+([+-][\d.]+)\s+span\s+([\d.]+)")


def run(cmd, log=None):
    env_cmd = [str(c) for c in cmd]
    print("  $", " ".join(env_cmd[:4]), "..." if len(env_cmd) > 4 else "")
    import os
    env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONIOENCODING="utf-8")
    p = subprocess.run(env_cmd, cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if log:
        Path(log).write_text(p.stdout + "\n" + p.stderr, encoding="utf-8")
    if p.returncode != 0:
        sys.exit(f"command failed ({p.returncode}): {' '.join(env_cmd[:3])}\n{(p.stderr or p.stdout)[-1500:]}")
    return p.stdout


def hf_url(cfg, key):
    return cfg["flow_lm"]["lookup_table"][key]


def ensure_weights(lang, cfg_path):
    from export_multilingual import download_safetensors
    weights = ROOT / "models" / lang / "model.safetensors"
    if not weights.exists() and not download_safetensors(lang, cfg_path, weights.parent):
        sys.exit(f"no weights for {lang}")
    return weights


def ensure_sentencepiece(cfg, dest):
    """tokenizer.model for the C++ runner: the tokenizer_path with .json replaced by .model (same repo/revision)."""
    from export_multilingual import hf_download
    url = hf_url(cfg, "tokenizer_path")
    base, _, rev = url.partition("@")
    model_url = (base[:-5] + ".model" if base.endswith(".json") else base) + (f"@{rev}" if rev else "")
    if not (dest / "tokenizer.model").exists() and not hf_download(model_url, dest, "tokenizer.model"):
        sys.exit(f"could not download {model_url}")
    return dest / "tokenizer.model"


def export_all_heads(lang, cfg_path, weights, work, fresh):
    cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
    t = cfg["flow_lm"]["transformer"]
    heads = [(l, h) for l in range(t["num_layers"]) for h in range(t["num_heads"])]
    spec = ",".join(f"{l}:{h}" for l, h in heads)
    if fresh and work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    if not (work / "mimi_encoder.onnx").exists():
        print("exporting mimi + text conditioner ...")
        run([sys.executable, "scripts/export_mimi_and_conditioner.py", "--output_dir", work, "--weights_path", weights,
             "--config", cfg_path], work / "mimi.log")
    if not (work / "flow_lm_main.onnx").exists():
        print(f"exporting flow_lm with ts_logits for all {len(heads)} heads ...")
        run([sys.executable, "scripts/export_flow_lm.py", "--output_dir", work, "--weights_path", weights,
             "--config", cfg_path, "--timestamp_heads", spec], work / "flow.log")
    return heads, spec


def sweep(lang, work, spec, seeds, audio):
    scores = {}
    for seed in seeds:
        print(f"probing seed {seed} ...")
        out = run([sys.executable, PROBE, "--onnx_dir", work, "--language", lang, "--heads", spec,
                   "--seed", seed] + (["--audio", audio] if audio else []), work / f"probe_s{seed}.txt")
        for line in out.splitlines():
            m = LINE.match(line)
            if m:
                scores.setdefault((int(m[1]), int(m[2])), []).append(float(m[3]) * float(m[4]))
    # worst case over seeds, mean as tie-break (many heads tie at 1.0)
    ranked = sorted(((min(v), sum(v) / len(v), k) for k, v in scores.items()), reverse=True)
    return ranked


def write_yaml(cfg_path, heads, note):
    """Replace any timestamp_heads block and timestamp-related top-level comment blocks, insert the new block before flow_lm:."""
    lines = cfg_path.read_text(encoding="utf-8").split("\n")
    out, i = [], 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("#"):  # contiguous comment block: drop it when it is about timestamps
            j = i
            while j < len(lines) and lines[j].startswith("#"):
                j += 1
            if not any("timestamp" in c.lower() for c in lines[i:j]):
                out.extend(lines[i:j])
            i = j
        elif ln.startswith("timestamp_heads:"):
            i += 1
            while i < len(lines) and (lines[i].startswith("- ") or lines[i].startswith("  ")) and lines[i].strip():
                i += 1
        else:
            out.append(ln)
            i += 1
    block = [f"# {note}", "timestamp_heads:"] + [f"- layer: {l}\n  head: {h}" for l, h in heads] + [""]
    at = next(k for k, ln in enumerate(out) if ln.startswith("flow_lm:"))
    out[at:at] = block
    cfg_path.write_text("\n".join(out), encoding="utf-8")


def e2e(lang, cfg_path, cfg, weights, work, head, cli, audio):
    layer, h = head
    d = work / f"e2e_L{layer}H{h}"
    d.mkdir(exist_ok=True)
    fp32 = d / "fp32"
    print(f"e2e: single-head export L{layer}H{h} ...")
    fp32.mkdir(exist_ok=True)
    run([sys.executable, "scripts/export_mimi_and_conditioner.py", "--output_dir", fp32, "--weights_path", weights,
         "--config", cfg_path], d / "mimi.log")
    run([sys.executable, "scripts/export_flow_lm.py", "--output_dir", fp32, "--weights_path", weights,
         "--config", cfg_path, "--timestamp_heads", f"{layer}:{h}"], d / "flow.log")
    run([sys.executable, "scripts/quantize.py", "--input_dir", fp32, "--output_dir", d], d / "quant.log")
    for f in ("mimi_encoder.onnx", "bos_before_voice.npy"):
        if (fp32 / f).exists():
            shutil.copy(fp32 / f, d / f)
    tok = ensure_sentencepiece(cfg, d)
    from pocket_tts.default_parameters import DEFAULT_TEXT_FOR_LANGUAGE
    text = DEFAULT_TEXT_FOR_LANGUAGE.get(lang, DEFAULT_TEXT_FOR_LANGUAGE["english"])
    (d / "text.txt").write_text(text, encoding="utf-8")
    p = subprocess.run([str(cli), str(d), str(tok), str(audio), "@" + str(d / "text.txt"), str(d / "out.wav")],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(p.stdout.strip()[-1200:])
    return p.returncode == 0 and "RESULT: PASS" in p.stdout


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", required=True, help="config name in pocket_tts/config (german, spanish, ...)")
    ap.add_argument("--work_dir", default=None, help="default: timestamp_probe/<lang>")
    ap.add_argument("--seeds", default="1234,1,2")
    ap.add_argument("--top", type=int, default=8, help="rows to print")
    ap.add_argument("--pick", type=int, default=1, help="how many of the best heads to write / test")
    ap.add_argument("--min_score", type=float, default=0.95, help="refuse to write/test heads scoring below this")
    ap.add_argument("--write_yaml", action="store_true")
    ap.add_argument("--e2e", action="store_true", help="export + INT8 + run pocket-tts-ts-cli with the best head")
    ap.add_argument("--cli", default=str(DEFAULT_CLI))
    ap.add_argument("--audio", default=str(ROOT.parent / "default.wav"))
    ap.add_argument("--fresh", action="store_true", help="redo the all-heads export")
    args = ap.parse_args()

    cfg_path = CONFIG_DIR / f"{args.lang}.yaml"
    cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
    weights = ensure_weights(args.lang, cfg_path)
    work = Path(args.work_dir) if args.work_dir else ROOT / "timestamp_probe" / args.lang
    heads, spec = export_all_heads(args.lang, cfg_path, weights, work, args.fresh)
    ranked = sweep(args.lang, work, spec, [int(s) for s in args.seeds.split(",")], args.audio)

    print(f"\n{args.lang}: {len(heads)} heads, score = min over seeds of spearman*span")
    for worst, mean, (l, h) in ranked[:args.top]:
        print(f"  L{l}H{h:<2}  worst {worst:.3f}  mean {mean:.3f}")
    best = [k for worst, _, k in ranked[:args.pick] if worst >= args.min_score]
    if not best:
        sys.exit(f"no head scores >= {args.min_score}; nothing to enable")
    configured = [(h.get("layer"), h.get("head")) for h in cfg.get("timestamp_heads") or []]
    print("currently configured:", configured or "none", " best:", best)

    ok = True
    if args.e2e:
        ok = e2e(args.lang, cfg_path, cfg, weights, work, best[0], Path(args.cli), args.audio)
        print("e2e:", "PASS" if ok else "FAIL")
    if args.write_yaml:
        if not ok:
            sys.exit("e2e failed; yaml left untouched")
        note = ("Word-timestamp head(s) for these weights, found by scripts/find_timestamp_heads.py "
                f"(worst-seed score {ranked[0][0]:.3f}). Exporters emit `ts_logits` when this is set.")
        write_yaml(cfg_path, best, note)
        print(f"wrote timestamp_heads {best} to {cfg_path}")


if __name__ == "__main__":
    main()
