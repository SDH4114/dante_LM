import torch
import torch.nn as nn
from .attention import CausalSelfAttention
from .block import TransformerBlock

class DanteLM(nn.Module):
    def __init__(self, vocab_size, context_size, d_model=64, n_heads=4, n_layers = 4):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(context_size, d_model)

        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads) for _ in range(n_layers)])
        # self.ln1 = nn.LayerNorm(d_model)
        # self.attention = CausalSelfAttention(d_model, n_heads)
        # self.ln2 = nn.LayerNorm(d_model)

        # self.ff = nn.Sequential(nn.Linear(d_model, d_model*4), nn.GELU(), nn.Linear(d_model*4, d_model))
        # self.lm_head = nn.Linear(d_model, vocab_size)

        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        _, seq_len = x.shape
        token_vectors = self.token_embedding(x)

        positions = torch.arange(seq_len, device=x.device)
        position_vectors = self.position_embedding(positions)

        x = token_vectors + position_vectors
        # x = x + self.attention(self.ln1(x))
        # x = x + self.ff(self.ln2(x))

        for block in self.blocks:
            x = block(x)

        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits
