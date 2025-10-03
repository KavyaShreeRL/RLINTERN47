import argparse
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from experiments.train1_charlm import CharLM


def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encoded = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
    return encoded, stoi, itos, len(chars)


def evaluate(model, data, seq_len, vocab_size, device, plot_name):
    """Compute BPC for dataset chunks and plot per-batch BPC."""
    model.eval()
    bpcs = []
    with torch.no_grad():
        for i in range(0, len(data) - seq_len - 1, seq_len):
            x = data[i : i + seq_len].unsqueeze(0).to(device)
            y = data[i + 1 : i + seq_len + 1].unsqueeze(0).to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
            bpc = loss / torch.log(torch.tensor(2.0))
            bpcs.append(bpc.item())

    plt.figure(figsize=(8, 5))
    plt.plot(range(len(bpcs)), bpcs, marker="o", markersize=2, label="BPC per batch")
    plt.xlabel("Batch Index")
    plt.ylabel("Bits per Character (BPC)")
    plt.title(f"Validation BPC Curve (Dataset Chunks) - {plot_name}")
    plt.legend()
    plt.grid(True)
    save_path = f"bpc_curve_{plot_name}.png"
    plt.savefig(save_path)
    plt.show()
    print(f"Saved BPC curve to {save_path}")

    avg_bpc = sum(bpcs) / len(bpcs)
    return avg_bpc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt", type=str, required=True, help="Path to checkpoint file"
    )
    parser.add_argument(
        "--data", type=str, required=True, help="Path to dataset (same as training)"
    )
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--d_model", type=int, default=192)
    parser.add_argument("--n_heads", type=int, default=3)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )

    parser.add_argument(
        "--plot_loss", action="store_true", help="Ignored in this version"
    )
    args = parser.parse_args()

    encoded, stoi, itos, vocab_size = load_data(args.data)
    device = args.device

    ckpt_name = Path(args.ckpt).stem

    model = CharLM(
        vocab_size, args.d_model, args.n_heads, args.n_layers, args.seq_len
    ).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)

    if "model_state" in ckpt:
        state_dict = ckpt["model_state"]
    else:
        state_dict = ckpt

    model_dict = model.state_dict()
    filtered_dict = {
        k: v
        for k, v in state_dict.items()
        if k in model_dict and v.size() == model_dict[k].size()
    }
    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict)
    print(f"Loaded {len(filtered_dict)}/{len(model_dict)} layers from checkpoint.")

    avg_bpc = evaluate(model, encoded, args.seq_len, vocab_size, device, ckpt_name)
    print(f"Average Validation BPC: {avg_bpc:.4f}")


if __name__ == "__main__":
    main()
