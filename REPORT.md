```markdown
# REPORT – Tiny Transformer Character LM

## 1. Implementation Overview
- **Scaled Dot-Product Attention** implemented in NumPy & PyTorch.
- **Multi-Head Attention** with configurable heads.
- **Positional Encodings**: sinusoidal (default) and learned.
- **Transformer Block**: MHA + Feedforward + LayerNorm + Residuals.
- **Character LM**: predicts one character at a time.
- **Training Loop**: random batches, AdamW, checkpoints every 1000 steps, validation every 500 steps.

---

## 2. Experiments

### Baseline
- 4 attention heads
- Sinusoidal positional encoding
- Seq_len=128, d_model=256, 2 layers
- Steps=5000

### Ablation
- 1 attention head
- No positional encoding
- Same seq_len, d_model, layers, steps

---

## 3. Results

### Loss & BPC Curves
- Saved as `checkpoints/loss_curve_baseline.png` and `checkpoints/loss_curve_nopos.png`
- BPC comparison plotted from eval.py

### Observations
- Ablation (no positional encoding, 1 head) → slower convergence, higher BPC
- Baseline → smoother loss curve, better generated text coherence

---

## 4. Generated Examples

### Baseline Prompt: "tokenization is"

Tokenization is the process of breaking text into smaller pieces before feeding to the model....


### Ablation Prompt: "tokenization is"

tokenization is the process of breaking text into smaller pieces before feeding to the model.
Self-attention lets every token look at every other ther sssghequen mence.
Pokel psititional ence tiodional us e.
sincode

---

## 5. Failure Modes & Fixes
1. **Repeating characters during generation** → addressed by increasing training steps.  
2. **Loss spikes** → fixed using gradient clipping and smaller learning rate.  
3. **Checkpoint not loading** → ensured correct meta info saved with each checkpoint.

---

## 6. Ablation Analysis
| Setting               | Heads | Pos Encoding | Steps | Avg BPC |
|-----------------------|-------|--------------|-------|---------|
| Baseline              | 4     | Yes          | 5000  | ~0.07   |
| Ablation (no-pos)     | 1     | No           | 5000  | ~9.45   |

> Clear impact of positional encoding and attention heads on text modeling.



