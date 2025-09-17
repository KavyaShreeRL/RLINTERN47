import torch
from src.mha import MultiHeadAttention


def test_multihead_output_shape():
    batch, seq_len, d_model = 2, 4, 8
    num_heads = 2
    x = torch.rand(batch, seq_len, d_model)
    mha = MultiHeadAttention(d_model, num_heads)
    out = mha(x)
    assert out.shape == (
        batch,
        seq_len,
        d_model,
    ), f"Expected ({batch},{seq_len},{d_model}), got {out.shape}"
