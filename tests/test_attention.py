import torch
from src.attention_torch import scaled_dot_product_attention

def test_attention_shapes():
    Q = torch.randn(2, 4, 8)
    K = torch.randn(2, 4, 8)
    V = torch.randn(2, 4, 8)
    out, weights = scaled_dot_product_attention(Q, K, V)
    assert out.shape == (2, 4, 8)
    assert weights.shape == (2, 4, 4)
