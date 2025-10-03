import numpy as np


def scaled_dot_product_attention(q, k, v, mask=None):
    """
    q, k, v: (batch, seq_len, d_model)
    mask: (batch, seq_len, seq_len)
    """
    dk = q.shape[-1]

    scores = np.matmul(q, k.transpose(0, 2, 1)) / np.sqrt(dk)
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
    weights = softmax(scores, axis=-1)
    output = np.matmul(weights, v)
    return output


def softmax(x, axis=None):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
