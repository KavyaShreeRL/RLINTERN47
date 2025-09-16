
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.transformer_block import TransformerBlock
from src.positional_encoding import LearnedPositionalEncoding, sinusoidal_positional_encoding
import matplotlib.pyplot as plt
import json

def load_text(data_path):
    text = open(data_path, "r", encoding="utf-8").read()
    chars = sorted(list(set(text)))
    stoi = {c:i for i,c in enumerate(chars)}
    itos = {i:c for c,i in stoi.items()}
    encoded = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    return text, encoded, stoi, itos, len(chars)

class CharLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, seq_len, mlp_mult=4, learned_pos=False):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        if learned_pos:
            self.pos_embedding = LearnedPositionalEncoding(seq_len, d_model)
        else:
            pe = sinusoidal_positional_encoding(seq_len, d_model)
            self.pos_embedding = nn.Embedding(seq_len, d_model)
            with torch.no_grad():
                self.pos_embedding.weight.copy_(pe)
        mlp_dim = d_model * mlp_mult
        self.layers = nn.ModuleList([TransformerBlock(d_model, n_heads, mlp_dim) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, idx):
        b, t = idx.shape
        positions = torch.arange(0, t, device=idx.device).unsqueeze(0)
        x = self.token_embedding(idx) + self.pos_embedding(positions)
        for layer in self.layers:
            x = layer(x)
        x = self.ln(x)
        return self.fc(x)

def save_checkpoint(path, model, optimizer, meta):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "meta": meta
    }, path)

def get_batch(encoded, seq_len, batch_size, device):
    max_start = len(encoded) - seq_len - 1
    starts = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([encoded[s:s+seq_len] for s in starts]).to(device)
    y = torch.stack([encoded[s+1:s+seq_len+1] for s in starts]).to(device)
    return x, y

@torch.no_grad()
def generate_text(model, start_text, length, stoi, itos, device):
    model.eval()
    idx = torch.tensor([stoi[c] for c in start_text], dtype=torch.long, device=device).unsqueeze(0)
    generated = start_text
    for _ in range(length):
        logits = model(idx[:, -model.seq_len:])
        next_idx = torch.argmax(logits[:, -1, :], dim=-1)
        next_char = itos[next_idx.item()]
        generated += next_char
        idx = torch.cat([idx, next_idx.unsqueeze(0)], dim=1)
    return generated

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--d_model", type=int, default=192)
    parser.add_argument("--n_heads", type=int, default=3)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--learned_pos", action="store_true")
    args = parser.parse_args()

    text, encoded, stoi, itos, vocab_size = load_text(args.data)
    device = args.device

    model = CharLM(vocab_size, args.d_model, args.n_heads, args.n_layers, args.seq_len, learned_pos=args.learned_pos).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    train_losses = []
    val_losses = []
    val_interval = 200
    os.makedirs("checkpoints", exist_ok=True)

    
    from experiments.eval import evaluate, load_data

    for step in range(args.max_steps):
        x, y = get_batch(encoded, args.seq_len, args.batch_size, device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        if step % val_interval == 0 and step > 0:
            encoded_val, stoi_val, itos_val, vocab_size_eval = load_data(args.data)
            val_loss, _ = evaluate(model, encoded_val, args.seq_len, vocab_size_eval, device)
            val_losses.append(val_loss)
            print(f"Step {step} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f}")

            meta = {"seq_len": args.seq_len, "d_model": args.d_model, "n_heads": args.n_heads, "n_layers": args.n_layers}
            save_checkpoint("checkpoints/last.pt", model, optimizer, meta)

    
    meta = {"seq_len": args.seq_len, "d_model": args.d_model, "n_heads": args.n_heads, "n_layers": args.n_layers}
    save_checkpoint("checkpoints/last.pt", model, optimizer, meta)
    print("Training complete. Model saved to checkpoints/last.pt")

    
    with open("loss_history.json", "w") as f:
        json.dump({"train": train_losses, "val": val_losses}, f)

    
    plt.figure(figsize=(10,5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot([i*val_interval for i in range(len(val_losses))], val_losses, label="Val Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("train_val_loss_curve.png")
    print("Saved training/validation loss curve to train_val_loss_curve.png")

    
    sample_text = generate_text(model, start_text="The ", length=200, stoi=stoi, itos=itos, device=device)
    print("\nGenerated sample:\n", sample_text)

if __name__ == "__main__":
    main()
