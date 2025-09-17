import torch.nn as nn
from src.attention_torch import scaled_dot_product_attention


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, causal: bool = True):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.h = num_heads
        self.d_k = d_model // num_heads
        self.causal = causal

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def split_heads(self, x):

        B, T, C = x.shape
        return x.view(B, T, self.h, self.d_k).transpose(1, 2)

    def combine_heads(self, x):

        x = x.transpose(1, 2).contiguous()
        B, T, _, _ = x.shape
        return x.view(B, T, self.d_model)

    def forward(self, x, mask=None):

        Q = self.split_heads(self.W_q(x))
        K = self.split_heads(self.W_k(x))
        V = self.split_heads(self.W_v(x))

        B, h, T, d_k = Q.shape
        Q2 = Q.reshape(B * h, T, d_k)
        K2 = K.reshape(B * h, T, d_k)
        V2 = V.reshape(B * h, T, d_k)

        attn_mask = None
        if mask is not None:  # mask: (B, T, T) -> (B*h, T, T)
            attn_mask = mask.repeat_interleave(h, dim=0)

        out, attn = scaled_dot_product_attention(
            Q2, K2, V2, mask=attn_mask, causal=self.causal
        )
        out = out.view(B, h, T, d_k)
        out = self.combine_heads(out)
        return self.W_o(out)
