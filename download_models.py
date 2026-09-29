"""Download only model.safetensors and the tokenizer for one, several, or all languages.

Usage (run from this directory, same as export_multilingual.py):
    python download_models.py                       # every language in pocket_tts/config/
    python download_models.py --lang french         # one language
    python download_models.py --lang french german  # several (space or comma separated)
    python download_models.py --force               # re-download even if files exist

Files land in models/<lang>/model.safetensors and models/<lang>/tokenizer.json|tokenizer.model,
exactly where export_multilingual.py expects them.
"""
import argparse

from export_multilingual import (
    CONFIG_DIR,
    MODELS_DIR,
    download_safetensors,
    download_tokenizer,
    tokenizer_filename,
)


def main():
    parser = argparse.ArgumentParser(description="Download model weights and tokenizer per language.")
    parser.add_argument("--lang", "-l", nargs="+", default=None,
                        help="Language name(s) matching pocket_tts/config/<lang>.yaml. Omit to download all.")
    parser.add_argument("--force", action="store_true", help="Re-download files that already exist")
    args = parser.parse_args()

    available = sorted(p.stem for p in CONFIG_DIR.glob("*.yaml"))
    if args.lang:
        langs = [l.strip() for arg in args.lang for l in arg.split(",") if l.strip()]
        unknown = [l for l in langs if l not in available]
        if unknown:
            print(f"Unknown language(s): {', '.join(unknown)}. Available: {', '.join(available)}")
            raise SystemExit(1)
    else:
        langs = available

    failed = []
    for lang in langs:
        config_path = CONFIG_DIR / f"{lang}.yaml"
        lang_dir = MODELS_DIR / lang
        lang_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {lang} ===")

        weights = lang_dir / "model.safetensors"
        if weights.exists() and not args.force:
            print(f"Weights already present: {weights}")
        elif not download_safetensors(lang, config_path, lang_dir):
            failed.append(f"{lang} (weights)")

        tokenizer = lang_dir / tokenizer_filename(config_path)
        if tokenizer.exists() and not args.force:
            print(f"Tokenizer already present: {tokenizer}")
        elif not download_tokenizer(lang, config_path, lang_dir):
            failed.append(f"{lang} (tokenizer)")

    print(f"\nDone: {len(langs)} language(s), {len(failed)} failure(s).")
    if failed:
        print("Failed: " + ", ".join(failed))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
