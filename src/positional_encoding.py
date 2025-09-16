import torch
import math
import torch.nn as nn

def sinusoidal_positional_encoding(seq_len: int, d_model: int):
    pe = torch.zeros(seq_len, d_model)
    pos = torch.arange(0, seq_len).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe  

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, seq_len: int, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(seq_len, d_model) * 0.01)

    def forward(self, t):
        
        return self.weight[t]
