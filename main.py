from tokenizer.bpe import build_vocab, encode, decode, show_tokens
from tokenizer.io import load_tokenizer
from data.token_data import make_training_pairs
from data.lm_dataset import LMDataset
from torch.utils.data import DataLoader
from model.dante import DanteLM
import torch.nn.functional as F
import torch
from train.train import train_step

def main():
    merges, special_tokens = load_tokenizer("tokenizer/tokenizer.json")
    vocab = build_vocab(merges)
    text = "Machine learning is amazing."

    tokens = encode(text, merges, special_tokens)

    decoded = decode(tokens, vocab, special_tokens)

    test_tokens = [
        10, 20, 30, 40, 50,
        60, 70, 80, 90, 100,
    ]

    pairs = make_training_pairs(test_tokens, context_size=4)

    print("--- TRAINING PAIRS ---")

    for x, y in pairs:
        print("X:", x)
        print("Y:", y)
        print()

    print("--- TOKENIZER ---")

    print("Original:", repr(text))
    print("Tokens:", tokens)
    print("Token count:", len(tokens))
    print("Decoded:", repr(decoded))
    print("Same:", text == decoded)

    print("\n--- TOKEN BREAKDOWN ---")

    show_tokens(text, merges, vocab, special_tokens)

    print()

    dataset = LMDataset(test_tokens, context_size=4)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    print("Dataset size:", len(dataset))

    for x, y in loader:
        print("X batch:")
        print(x)

        print("Y batch:")
        print(y)

        print("X shape:", x.shape)
        print("Y shape:", y.shape)

        break

    vocab_size = max(max(vocab), max(special_tokens.values())) + 1

    model = DanteLM(vocab_size=vocab_size, context_size=4, d_model=64, n_heads=4, n_layers=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    for x, y in loader:
        loss = train_step(model, x, y, optimizer, vocab_size)

        print("Loss:", loss)

        for epoch in range(20):
            for x, y in loader:
                loss = train_step(model, x, y, optimizer, vocab_size)

            print("Epoch:", epoch, "Loss:", loss)

        break

    num_params = sum(p.numel() for p in model.parameters())
    print("Parameters:", num_params)

if __name__ == "__main__":
    main()
