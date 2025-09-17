import numpy as np


def softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


def scaled_dot_product_attention(Q, K, V, mask=None, causal=False):
    """
    Q, K, V: shape (..., seq_len, d_k)
    mask: broadcastable mask where 0 means masked (False), 1 means keep (True)
    causal: if True, apply causal mask so each position can only attend to previous positions
    returns: (out, attn_weights)
    """
    d_k = Q.shape[-1]
    scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(d_k)

    if causal:
        L = scores.shape[-2]
        causal_mask = np.triu(np.ones((L, L), dtype=bool), k=1)
        scores = np.where(causal_mask[None, :, :], -1e9, scores)

    if mask is not None:

        scores = np.where(mask == 0, -1e9, scores)

    weights = softmax(scores, axis=-1)
    out = np.matmul(weights, V)
    return out, weights
