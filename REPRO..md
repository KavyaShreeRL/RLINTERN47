# Reproducibility Instructions – Tiny Transformer

## 1. Setup
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate.ps1
pip install -r requirements.txt


## 2. Run tests

$env:PYTHONPATH="${PWD}"
pytest -q
ruff check .
black --check .
mypy src


## 3. TRAIN

baseline - 

python -m experiments.train_charlm --data data/tiny.txt --seq_len 128 --d_model 256 --n_heads 4 --n_layers 2 --steps 5000

Ablation - 

python -m experiments.train_charlm --data data/tiny.txt --seq_len 128 --d_model 256 --n_heads 1 --n_layers 2 --steps 5000 --no_pos_encoding



## 4. EVALUATE AND GENERATE

Baseline (4 heads + positional encoding): (train1_charlm.py)

python -m experiments.eval --ckpt checkpoints/baseline_pos4heads.pt --data data/tiny.txt --seq_len 128 --d_model 128 --n_heads 4 --n_layers 2 --plot_loss

python -m experiments.generate --ckpt checkpoints/baseline_pos4heads.pt --data data/tiny.txt --seq_len 128 --d_model 128 --n_heads 4 --n_layers 2 --max_new_tokens 200



Ablation: No positional encoding, 1 head (train_charlm.py)

python -m experiments.eval --ckpt checkpoints/ablation_no_pos_1head.pt --data data/tiny.txt --seq_len 128 --d_model 128 --n_heads 1 --n_layers 2 --plot_loss

python -m experiments.generate --ckpt checkpoints/ablation_no_pos_1head.pt --data data/tiny.txt --seq_len 128 --d_model 128 --n_heads 1 --n_layers 2 --max_new_tokens 200








