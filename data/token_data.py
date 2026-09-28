def make_chunks(tokens, chunk_size):
    chunks = []

    for i in range(0, len(tokens), chunk_size):
        chunk = tokens[i:i + chunk_size]

        if len(chunk)==chunk_size:
            chunks.append(chunk)

    return chunks

def make_training_pairs(tokens, context_size):
    pairs = []
    step = context_size

    for i in range(0, len(tokens) - context_size, step):
        chunk = tokens[i:i+ context_size+1]

        if len(chunk) != context_size+1:
            continue

        x = chunk[:-1]
        y = chunk[1:]

        pairs.append((x,y))

    return pairs
