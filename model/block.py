import torch.nn as nn
from .attention import CausalSelfAttention

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model)
        self.attention = CausalSelfAttention(d_model, n_heads)

        self.ln2 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model))

    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        x = x + self.ff(self.ln2(x))

        return x
