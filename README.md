# Attention Assignment - Character-Level Transformer

## Overview
This project implements a **character-level language model (CharLM)** using a **Transformer architecture**. It demonstrates:

- Token-level text generation.
- Multi-head self-attention.
- Positional encoding (and ablation study without it).
- Bits-per-character (BPC) evaluation.

---

## Folder Structure

attention-assignments/
├─ data/ 
│ └─ tiny.txt
├─ experiments/
│ ├─ train1_charlm.py 
│ ├─ generate.py 
│ ├─ eval.py 
├─ src/
│ ├─ transformer_block.py
│ ├─ positional_encoding.py
├─ checkpoints/ # Saved model checkpoints
├─ train_val_loss_curve.png
├─ loss_history.json
├─ README.md
├─ REPORT.md
└─ repro.md