import torch
import torch.nn as nn
from src.mha import MultiHeadAttention

class FeedForward(nn.Module):
    def __init__(self, d_model: int, mlp_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, d_model)
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, mlp_dim: int):
        super().__init__()
        self.attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads, causal=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, mlp_dim)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        
        y = self.ln1(x)
        y = self.attn(y, mask=mask)
        x = x + y
        z = self.ln2(x)
        z = self.ff(z)
        return x + z
