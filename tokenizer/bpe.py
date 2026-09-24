from collections import Counter
from tokenizer.pretokenizer import pre_tokenize


def text_to_bytes(text):
    return list(text.encode("utf-8"))

def get_pairs(tokens):
    return Counter(zip(tokens, tokens[1:]))

def get_pairs_edu(tokens):
    pairs = {}
    for a, b in zip(tokens, tokens[1:]):
        # zip() создает пары соседних токенов
        pair = (a, b)

        if pair not in pairs:
            pairs[pair] = 0

        pairs[pair]+=1

    return pairs

def merge(tokens, pair, new_id):
    # Ищет определённую пару токенов и заменяет эту пару одним новым токеном
    new_tokens = []
    i = 0

    while i < len(tokens):
        if i < len(tokens)- 1 and tokens[i] == pair[0] and tokens[i+1] == pair[1]:
            new_tokens.append(new_id)
            i +=2
        else:
            new_tokens.append(tokens[i])
            i +=1

    return new_tokens

def count_pairs(corpus):
    pairs_count = Counter()

    for tokens in corpus:
        pairs_count.update(get_pairs(tokens))

    return pairs_count


def train_bpe(texts, vocab_size=300, max_docs=100):
    corpus = []

    for i, text in enumerate(texts):
        if i >= max_docs:
            break

        pieces = pre_tokenize(text)

        for piece in pieces:
            corpus.append(text_to_bytes(piece))

    merges = {}
    next_id = 256

    while next_id < vocab_size:
        pair_count = count_pairs(corpus)
        if not pair_count:
            break

        best_pair = pair_count.most_common(1)[0][0]
        merges[best_pair] = next_id

        new_corpus = []

        for tokens in corpus:
            tokens = merge(tokens, best_pair, next_id)
            new_corpus.append(tokens)

        corpus = new_corpus
        next_id += 1

    return merges

def build_vocab(merges):
    vocab = {}
    for i in range(256):
        vocab[i] = bytes([i])

    for pair, token_id in merges.items():
        a, b = pair
        vocab[token_id] = vocab[a]+ vocab[b]

    return vocab

def print_vocab(merges):
    vocab = build_vocab(merges)

    for pair, token_id in merges.items():
        token = vocab[token_id]
        print(token_id, token.decode("utf-8", "replace"))



def build_special_tokens(vocab):
    next_id = max(vocab)+1
    # <BOS> = Beginning Of Sequence
    # <EOS> = End Of Sequence
    special_tokens = {"<BOS>": next_id, "<EOS>": next_id + 1}

    return special_tokens


def encode(text, merges, special_tokens=None):
    result = []

    if special_tokens:
        result.append(special_tokens["<BOS>"])

    pieces = pre_tokenize(text)

    for piece in pieces:
        tokens = text_to_bytes(piece)

        for pair, token_id in merges.items():
            tokens = merge(tokens, pair, token_id)

        result.extend(tokens)

    if special_tokens:
        result.append(special_tokens["<EOS>"])

    return result

def decode(tokens, vocab, special_tokens=None):
    data = b""
    special_ids = set()

    if special_tokens:
        special_ids = set(special_tokens.values())


    for token_id in tokens:
        if token_id in special_ids:
            continue

        data += vocab[token_id]

    return data.decode("utf-8", errors="replace")
