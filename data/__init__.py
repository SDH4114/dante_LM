from data.ds import load_data, get_text


ds = load_data()
texts = get_text(ds)

for i, sample in enumerate(ds.take(5), start=1):
    print(f"\n--- DOCUMENT {i} ---\n")
    print(sample["text"][:1000])

print(next(texts))
