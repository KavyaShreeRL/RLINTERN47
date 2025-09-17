import torch
import argparse
from experiments.train_charlm import CharLM, load_text


@torch.no_grad()
def generate_greedy(model, start_text, length, stoi, itos, device):
    model.eval()
    idx = torch.tensor(
        [stoi[c] for c in start_text if c in stoi], dtype=torch.long, device=device
    ).unsqueeze(0)
    if idx.size(1) == 0:
        idx = torch.tensor(
            [[0]], dtype=torch.long, device=device
        )  # fallback to first char
    generated = start_text

    for _ in range(length):
        logits = model(idx[:, -model.seq_len :])
        next_idx = torch.argmax(logits[:, -1, :], dim=-1)
        next_char = itos[next_idx.item()]
        generated += next_char
        idx = torch.cat([idx, next_idx.unsqueeze(0)], dim=1)

    return generated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt", type=str, required=True, help="Path to checkpoint file"
    )
    parser.add_argument("--data", type=str, required=True, help="Path to dataset")
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--start_text", type=str, default="The ")
    args = parser.parse_args()

    device = args.device
    text, encoded, stoi, itos, vocab_size = load_text(args.data)

    # Load model
    ckpt = torch.load(args.ckpt, map_location=device)
    meta = ckpt.get("meta", {})
    model = CharLM(
        vocab_size=vocab_size,
        d_model=meta.get("d_model", args.d_model),
        n_heads=meta.get("n_heads", args.n_heads),
        n_layers=meta.get("n_layers", args.n_layers),
        seq_len=meta.get("seq_len", args.seq_len),
        learned_pos=meta.get("learned_pos", False),
        use_pos_encoding=meta.get("use_pos_encoding", True),
    ).to(device)

    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    print("Model loaded. Type a prompt and press Enter to generate text.")
    print("Type 'exit' to quit.\n")

    while True:
        start_seq = input("Enter prompt: ").strip()
        if start_seq.lower() == "exit":
            break
        generated_text = generate_greedy(
            model, start_seq, args.max_new_tokens, stoi, itos, device
        )
        print("\nGenerated Text:\n")
        print(generated_text)
        print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    main()
