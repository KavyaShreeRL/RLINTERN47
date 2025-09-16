import torch
from src.mha import MultiHeadAttention

def test_mha_forward():
    x = torch.randn(2, 10, 32)
    mha = MultiHeadAttention(d_model=32, num_heads=4)
    out = mha(x)
    assert out.shape == (2, 10, 32)
