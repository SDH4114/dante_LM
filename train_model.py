import torch
from torch.utils.data import DataLoader
from data.ds import load_data
from data.lm_dataset import LMDataset
from data.training_data import build_token_stream
from tokenizer.bpe import build_vocab, encode
from tokenizer.io import load_tokenizer
from model.dante import DanteLM
from train.train import train_model


def main():
    merges, special_tokens = load_tokenizer("tokenizer/tokenizer.json")
    vocab = build_vocab(merges)
    ds = load_data()
    tokens = build_token_stream(ds, merges, special_tokens, encode, max_docs=100)

    print("Training tokens:", len(tokens))

    context_size = 64
    batch_size = 8

    dataset = LMDataset(tokens, context_size=context_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    vocab_size = max(max(vocab), max(special_tokens.values())) + 1
    model = DanteLM(vocab_size=vocab_size, context_size=context_size, d_model=64, n_heads=4, n_layers=4)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")


    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    num_params = sum(p.numel() for p in model.parameters())

    print("Device:", device)
    print("Dataset samples:", len(dataset))
    print("Parameters:", num_params)

    train_model(model, loader, optimizer, vocab_size, device, epochs=3)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "vocab_size": vocab_size,
        "context_size": context_size,
        "d_model": 64,
        "n_heads": 4,
        "n_layers": 4
    }

    torch.save(checkpoint, "checkpoints/dante_v0.pt")
    print("Model saved: checkpoints/dante_v0.pt")


if __name__ == "__main__":
    main()
