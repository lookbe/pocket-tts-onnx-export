"""Writes the fixture consumed by pocket-tts-cpp/tests/test_word_timestamps.cpp.

Cases ("MAP"): for texts in several languages, the SentencePiece pieces, the text units and the token->unit matrix
exactly as pocket-tts-timestamped builds them (timestamps/text.py). The C++ host must reproduce units + matrix from
(text, pieces) alone.

Case ("ALIGN"): one utterance's raw `ts_logits` per frame, the voiced flags and the reference model's word times,
so the whole host chain (softmax over the text slice -> unit scores -> WordAligner) is compared with the reference.
(Written by verify_timestamps.py --dump_fixture, which has those inputs.)

    python dump_ts_fixture.py --ref_repo ../../../pocket-tts-timestamped --out fixture.txt
"""
import argparse
import sys
from pathlib import Path

import numpy as np

HF = Path.home() / ".cache" / "huggingface" / "hub"

TEXTS = {
    "english": [
        "Hello world.",
        "The quick brown fox jumps over the lazy dog, and then it takes a short nap.",
        "I'm fast enough to run on small CPUs - isn't that well-known? Yes!",
        "Wait... what? \"Really,\" she said.",
    ],
    "german": ["Hallo Welt. Ich bin schnell genug, um auf kleinen CPUs zu laufen.", "Schöne Grüße aus Köln, Zürich und Wien!"],
    "french": ["Bonjour le monde. J'espère que vous m'aimerez.", "Où est l'été d'aujourd'hui ? Ça va très bien, merci."],
    "italian": ["Ciao mondo. Spero che ti piacerò.", "Perché l'amore è così difficile, anche oggi?"],
    "portuguese": ["Olá mundo. Espero que você goste de mim.", "Não há coração que não se emocione, até amanhã."],
    "spanish": ["Hola mundo. Soy lo suficientemente rápido.", "¿Dónde está el niño? ¡Qué día tan bonito, señor!"],
}


def tokenizer_path(lang):
    if lang == "english":
        hits = sorted(HF.glob("models--kyutai--pocket-tts-without-voice-cloning/snapshots/*/languages/english_2026-09/tokenizer.model"))
    else:
        hits = sorted(HF.glob(f"models--kyutai--pocket-tts/snapshots/*/languages/{lang}/tokenizer.model"))
    return hits[0] if hits else None


def write_units_and_map(f, text, pieces, chunk):
    f.write(f"TEXT\t{text}\n")
    f.write(f"PIECES {len(pieces)}\n")
    for p in pieces:
        f.write(p + "\n")
    f.write(f"UNITS {len(chunk.units)}\n")
    for u in chunk.units:
        f.write(f"{int(u.is_word)} {int(u.synthetic)} {u.chunk_begin} {u.chunk_end}\t{u.text}\n")
    m = chunk.token_to_unit.numpy()
    f.write(f"TOKMAP {m.shape[0]} {m.shape[1]}\n")
    for row in m:
        f.write(" ".join(f"{v:.9g}" for v in row) + "\n")


def byte_offsets(text, chunk):
    """Python's units are in the coordinate system of the tokenizer reader; for the byte-offset sentencepiece (0.2.2+)
    the matrix is built from byte spans, while unit begin/end are code points. Convert units to bytes for the file."""
    boundaries = [0]
    for ch in text:
        boundaries.append(boundaries[-1] + len(ch.encode("utf-8")))
    return boundaries


class _View:
    pass


def write_align_case(path, tchunk, sp, text_start, frame_seconds, rows, voiced, ref_words):
    """Append the ALIGN case: units/matrix as in MAP, then per frame the raw ts_logits [n_heads, valid_len]."""
    pieces = sp.encode(tchunk.text, out_type=str)
    b = byte_offsets(tchunk.text, tchunk)
    units = []
    for u in tchunk.units:
        v = _View()
        v.is_word, v.synthetic, v.text = u.is_word, u.synthetic, u.text
        v.chunk_begin, v.chunk_end = b[u.chunk_begin], b[u.chunk_end]
        units.append(v)
    view = _View()
    view.units, view.token_to_unit = units, tchunk.token_to_unit
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write("CASE ALIGN english\n")
        write_units_and_map(f, tchunk.text, pieces, view)
        f.write(f"ALIGN {text_start} {rows[0].shape[0]} {frame_seconds:.9g} {len(rows)}\n")
        for row, v in zip(rows, voiced):
            f.write(f"FRAME {int(v)} {row.shape[1]}\n")
            for h in range(row.shape[0]):
                f.write(" ".join(f"{x:.9g}" for x in row[h].astype(np.float32)) + "\n")
        f.write(f"WORDS {len(ref_words)}\n")
        for w in ref_words:
            f.write(f"{w[0]}\t{w[1]:.4f}\t{w[2]:.4f}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref_repo", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.path.insert(0, args.ref_repo)
    import sentencepiece as spm
    from pocket_tts_timestamped.timestamps.text import _iter_timestamp_text_chunks

    cases = 0
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        for lang, texts in TEXTS.items():
            path = tokenizer_path(lang)
            if path is None:
                print(f"skip {lang}: no cached tokenizer.model")
                continue
            sp = spm.SentencePieceProcessor(model_file=str(path))
            for text in texts:
                chunk = list(_iter_timestamp_text_chunks(text, [text], sp))[0]
                pieces = sp.encode(chunk.text, out_type=str)
                b = byte_offsets(chunk.text, chunk)
                # Units to byte offsets (the host works on UTF-8 bytes).
                class U:  # minimal view with byte coordinates
                    pass
                units = []
                for u in chunk.units:
                    v = U()
                    v.is_word, v.synthetic, v.text = u.is_word, u.synthetic, u.text
                    v.chunk_begin, v.chunk_end = b[u.chunk_begin], b[u.chunk_end]
                    units.append(v)
                chunk_view = U()
                chunk_view.units = units
                chunk_view.token_to_unit = chunk.token_to_unit
                f.write(f"CASE MAP {lang}\n")
                write_units_and_map(f, chunk.text, pieces, chunk_view)
                cases += 1
    print(f"wrote {cases} MAP cases to {args.out}")


if __name__ == "__main__":
    main()
