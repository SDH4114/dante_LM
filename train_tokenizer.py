from data.ds import load_data
from tokenizer.bpe import train_bpe, build_vocab, build_special_tokens
from tokenizer.io import save_tokenizer

def main():
    ds = load_data()
    samples = list(ds.take(100))
    texts = [sample["text"] for sample in samples]

    print("Documents loaded:", len(texts))

    merges = train_bpe(texts, vocab_size=500)

    vocab = build_vocab(merges)
    special_tokens = build_special_tokens(vocab)

    save_tokenizer("tokenizer/tokenizer.json", merges, special_tokens)

    print("Tokenizer trained")
    print("BPE tokens - ", len(vocab))
    print("Special tokens - ", special_tokens)

if __name__ == "__main__":
    main()
