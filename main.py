# from data.ds import load_data, get_text
# from tokenizer.tokenizer import build_vocab
from tokenizer.bpe import get_pairs, merge, text_to_bytes, train_bpe, print_vocab, encode, decode, build_vocab


def main():
    texts = [
        "hello hello hello",
        "hello help hello",
        "hello world hello",
    ]

    merges = train_bpe(
        texts,
        vocab_size=270,
    )

    vocab = build_vocab(merges)

    text = "hello world"

    tokens = encode(text, merges)
    decoded = decode(tokens, vocab)

    print("Original:", text)
    print("Encoded: ", tokens)
    print("Decoded: ", decoded)


if __name__ == "__main__":
    main()
