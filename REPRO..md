# Reproducibility Instructions

## Environment
- Python 3.10+
- PyTorch >= 2.0
- matplotlib

```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch matplotlib


python -m experiments.train1_charlm --data data/tiny.txt --seq_len 128 --d_model 128 --n_heads 4 --n_layers 2 --batch_size 32 --max_steps 2000

python -m experiments.eval --ckpt checkpoints/last.pt --data data/tiny.txt --seq_len 128 --d_model 128 --n_heads 4 --n_layers 2 --plot_loss

python -m experiments.generate --ckpt checkpoints/last.pt --data data/tiny.txt --seq_len 128 --d_model 128 --n_heads 4 --n_layers 2 --max_new_tokens 200


Baseline (4 heads + positional encoding): (train1_charlm.py)

python -m experiments.eval --ckpt checkpoints/baseline_pos4heads.pt --data data/tiny.txt --seq_len 128 --d_model 128 --n_heads 4 --n_layers 2 --plot_loss

python -m experiments.generate --ckpt checkpoints/baseline_pos4heads.pt --data data/tiny.txt --seq_len 128 --d_model 128 --n_heads 4 --n_layers 2 --max_new_tokens 200



Ablation: No positional encoding, 1 head (train_charlm.py)

python -m experiments.eval --ckpt checkpoints/ablation_no_pos_1head.pt --data data/tiny.txt --seq_len 128 --d_model 128 --n_heads 1 --n_layers 2 --plot_loss

python -m experiments.generate --ckpt checkpoints/ablation_no_pos_1head.pt --data data/tiny.txt --seq_len 128 --d_model 128 --n_heads 1 --n_layers 2 --max_new_tokens 200