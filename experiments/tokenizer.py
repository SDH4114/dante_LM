from data.ds import load_data, get_text


ds = load_data()
texts = get_text(ds)

def tokenize(text):
    return text.split()

def build_vocab(texts, max_docs=1000):
    vocab = {}

    for i, text in enumerate(texts):
        if i >= max_docs:
            break

        tokens = tokenize(text)

        for token in tokens:
            if token not in vocab:
                vocab[token] = len(vocab)

    return vocab

print(build_vocab(texts))
