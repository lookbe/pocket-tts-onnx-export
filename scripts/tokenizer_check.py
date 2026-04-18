
import os
import sentencepiece as spm
import sys

# Paths to models
pytorch_tokenizer_path = r"d:\tools\kyutai\pocket-tts\experimental\models\english_v2\tokenizer.model"
web_tokenizer_path = r"d:\tools\kyutai\pocket-tts-web\tokenizer.model"

def check_tokenizer(model_path, name, word):
    if not os.path.exists(model_path):
        print(f"Error: {name} tokenizer not found at {model_path}")
        return None
    
    sp = spm.SentencePieceProcessor(model_file=model_path)
    # The LUTConditioner doesn't seem to do anything special other than calling sp.encode
    # However, let's look at what's in the web code.
    # We want to compare what happens when we encode "hello"
    
    tokens = sp.encode(word)
    token_pieces = sp.encode(word, out_type=str)
    
    print(f"--- {name} ---")
    print(f"Path: {model_path}")
    print(f"Vocab size: {sp.vocab_size()}")
    print(f"Input: '{word}'")
    print(f"Token IDs: {tokens}")
    print("-" * (len(name) + 8))


    return tokens

if __name__ == "__main__":
    word = "hello"
    if len(sys.argv) > 1:
        word = sys.argv[1]
        
    print(f"Comparison for word: '{word}'\n")
    
    t1 = check_tokenizer(pytorch_tokenizer_path, "PyTorch english_v2", word)
    t2 = check_tokenizer(web_tokenizer_path, "Web model", word)
    
    if t1 == t2:
        print("\nSUCCESS: Tokenizer IDs match exactly!")
    else:
        print("\nFAILURE: Tokenizer IDs mismatch!")

    # Now let's test with uppercase and period to see how it matches PyTorch normalization
    word_norm = "Hello."
    print(f"\nComparing with normalization: '{word_norm}'")
    t1n = check_tokenizer(pytorch_tokenizer_path, "PyTorch english_v2", word_norm)
    t2n = check_tokenizer(web_tokenizer_path, "Web model", word_norm)
