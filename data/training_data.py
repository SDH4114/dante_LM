def build_token_stream(ds, merges, special_tokens, encode, max_docs=100):
    tokens = []

    for sample in ds.take(max_docs):
        document_tokens = encode(sample["text"], merges, special_tokens)
        tokens.extend(document_tokens)

    return tokens
