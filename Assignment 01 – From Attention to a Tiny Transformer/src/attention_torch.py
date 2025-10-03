import torch
import torch.nn.functional as F


def scaled_dot_product_attention(Q, K, V, mask=None, causal=False):
    """
    Q, K, V : (..., seq_len, d_k)
    mask: (..., seq_q, seq_k) with True for keep, False for mask
    causal: if True, apply causal mask
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k**0.5)

    if causal:
        seq = scores.size(-1)
        causal_mask = torch.triu(
            torch.ones(seq, seq, dtype=torch.bool, device=scores.device), diagonal=1
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))

    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    attn = F.softmax(scores, dim=-1)
    out = torch.matmul(attn, V)
    return out, attn
