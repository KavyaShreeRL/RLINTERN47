import torch
from src.positional_encoding import (
    LearnedPositionalEncoding,
    sinusoidal_positional_encoding,
)


def test_learned_pos_encoding():
    seq_len, d_model = 10, 8
    pe_layer = LearnedPositionalEncoding(seq_len, d_model)
    positions = torch.arange(seq_len).unsqueeze(0).long()
    out = pe_layer(positions)
    assert out.shape == (1, seq_len, d_model)


def test_sinusoidal_pos_encoding():
    seq_len, d_model = 10, 8
    pe = sinusoidal_positional_encoding(seq_len, d_model)
    assert pe.shape == (
        seq_len,
        d_model,
    ), f"Expected ({seq_len},{d_model}), got {pe.shape}"
