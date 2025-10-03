# Assignment 01 – From Attention to a Tiny Transformer

##  Objective
Build a mini Transformer from scratch and train it as a character-level language model.  
Understand how 'Attention is All You Need' works under the hood and practice engineering hygiene.

###  Goals
- Implement scaled dot-product attention and multi-head attention.
- Add positional encodings (sinusoidal and learned).
- Build a Transformer block and character-level LM.
- Train, evaluate, and generate text.
- Compare baseline vs ablation (remove positional encoding, reduce heads).
- Maintain clean code, testing, and reproducibility.

---

##  Project Structure

Attention-assignment - 01/
├─ src/
│ ├─ attention_numpy.py
│ ├─ attention_torch.py
│ ├─ mha.py
│ ├─ positional_encoding.py
│ ├─ transformer_block.py
├─ experiments/
│ ├─ train_charlm.py
│ ├─ generate.py
│ ├─ eval.py
├─ data/
│ └─ tiny.txt
├─ tests/
│ ├─ test_attention.py
│ ├─ test_mha.py
│ ├─ test_positional.py
│ ├─ test_block.py
├─ checkpoints/
├─ REPORT.md
├─ REPRO.md
├─ requirements.txt









