from data.ds import load_data, get_text
from tokenizer.bpe import train_bpe, build_vocab, build_special_tokens
from tokenizer.io import save_tokenizer

def main():
    ds = load_data()
    texts = get_text(ds)
    merges = train_bpe(texts, vocab_size=500, max_docs=100)

    vocab = build_vocab(merges)
    special_tokens = build_special_tokens(vocab)

    save_tokenizer("tokenizer/tokenizer.json", merges, special_tokens)

    print("Tokenizer trained")
    print("BPE tokens - ", len(vocab))
    print("Special tokens - ", special_tokens)

if __name__ == "__main__":
    main()
