from tokenizer.bpe import build_vocab, encode, decode
from tokenizer.io import load_tokenizer

def main():
    merges, special_tokens = load_tokenizer("tokenizer/tokenizer.json")

    vocab = build_vocab(merges)
    text = "Machine learning is amazing."

    tokens = encode(text, merges, special_tokens)
    decoded = decode(tokens, vocab, special_tokens)

    print("Original:", repr(text))
    print("Tokens:", tokens)
    print("Token count:", len(tokens))
    print("Decoded:", repr(decoded))
    print("Same:", text == decoded)

if __name__ == "__main__":
    main()
