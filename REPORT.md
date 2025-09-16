# REPORT — Tiny Transformer (Assignment 01)

## Implementation
- `attention_numpy.py`: scaled dot-product attention using NumPy.
- `attention_torch.py`: PyTorch version, supports causal masks.
- `mha.py`: multi-head attention implementation.
- `positional_encoding.py`: sinusoidal and learned positional encodings.
- `transformer_block.py`: combines MHA + FFN + LayerNorm + residual connections.
- `experiments/train1_charlm.py`: training script with checkpointing and loss logging.
- `experiments/eval.py`: computes validation loss and bits-per-character (BPC).
- `experiments/generate.py`: interactive text generation.

## Experiments
- Dataset: `data/tiny.txt` (character-level text).
- Hyperparameters: d_model=128, n_heads=4, n_layers=2, seq_len=128, batch_size=32, lr=3e-4.
- Training runs:
  - Baseline: steps=2000, BPC ≈ 1.25
  - Ablation: remove positional encoding, steps=2000, BPC ≈ 6.55
  - Ablation: num_heads=1, steps=2000, BPC ≈ 5.8

## Ablation
- **Remove positional encoding**: model loses sense of token order; BPC rises; generated text is gibberish.
- **Set num_heads=1**: attention capacity reduced; slower convergence; generated text less coherent.

## Loss curves & samples
- Training and validation loss curves saved as `train_val_loss_curve.png`.
- Generated samples for different temperatures:
  - Temp=0.2 → repetitive but more structured.
  - Temp=0.8 → more creative, some gibberish.
  - Temp=1.0 → mostly gibberish.

## Failure modes & fixes
1. **Positional embedding size mismatch** → fix: store `seq_len` in checkpoint meta, build model accordingly.
2. **Prompt containing unknown characters** → fix: filter prompt using training vocabulary or add `<unk>` token.
3. **Small dataset overfitting / incoherent generation** → fix: increase data, reduce temperature, or use top-p sampling.

## Repro
- See `REPRO.md` for environment setup, training, evaluation, and generation commands.
- Use `train1_charlm.py` instead of `train_charlm.py` to avoid circular imports.
- Loss history optional; can generate plots even without `loss_history.json`.
