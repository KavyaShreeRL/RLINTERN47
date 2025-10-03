import torch
from src.transformer_block import TransformerBlock


def test_transformer_block_shape():
    batch, seq_len, d_model = 2, 4, 8
    mlp_dim = 16  # hidden dimension for the feed-forward layer
    x = torch.rand(batch, seq_len, d_model)

    # Pass mlp_dim along with num_heads
    block = TransformerBlock(d_model, num_heads=2, mlp_dim=mlp_dim)

    out = block(x)
    assert out.shape == (
        batch,
        seq_len,
        d_model,
    ), f"Expected ({batch},{seq_len},{d_model}), got {out.shape}"
