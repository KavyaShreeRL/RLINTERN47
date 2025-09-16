import torch
from src.positional_encoding import sinusoidal_positional_encoding

def test_positional_encoding_shape():
    pe = sinusoidal_positional_encoding(50, 32)
    assert pe.shape == (50, 32)
