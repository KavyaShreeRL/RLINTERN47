import torch
from src.transformer_block import TransformerBlock

def test_transformer_block_forward():
    x = torch.randn(2, 10, 32)
    block = TransformerBlock(d_model=32, num_heads=4, mlp_dim=64)
    out = block(x)
    assert out.shape == (2, 10, 32)
