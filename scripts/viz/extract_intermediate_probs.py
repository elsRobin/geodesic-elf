"""
Extract LM Head probability distributions at intermediate ODE time steps
for the running example (GSM8K duck-egg problem).

Usage (on AutoDL with GPU):
    python scripts/extract_probs_running_example.py \
        --checkpoint checkpoints/gsm8k_run1_step100000.pt \
        --output figs/probs_run1.json \
        --text "Janet's ducks lay 16 eggs per day..."

Output JSON structure per model:
{
  "text": "...",
  "R": 1.94,
  "snapshots": {
    "0.3": {"top_tokens": [["90", 0.42], ["$", 0.15], ...], "top_k": 10},
    "0.6": {...},
    "0.9": {...}
  }
}

Then visualize with scripts/plot_running_example_probs.py
"""

import json
import argparse
import os
import torch
import torch.nn.functional as F
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from elf.model import ELFModel
from elf.config import ModelConfig
from elf.data.tokenizer import BPETokenizer


@torch.no_grad()
def extract_prob_distributions(
    model: ELFModel,
    tokenizer: BPETokenizer,
    text: str,
    snapshot_times: list[float],
    num_ode_steps: int = 25,
    noise_scale: float = 2.0,
    top_k: int = 15,
    device: torch.device = None,
) -> dict:
    """
    Run ODE integration and extract LM Head softmax probabilities
    at specified snapshot times.
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    # Tokenize input
    ids = tokenizer.encode(text)
    ids_tensor = torch.tensor([ids], device=device)
    E = model.embed_dim

    # Get clean embedding
    x0 = model.embedding(ids_tensor)

    # Initialize noise
    z = torch.randn(1, x0.shape[1], E, device=device) * noise_scale

    # Build time schedule (linear spacing)
    dt = 1.0 / num_ode_steps
    t_values = torch.linspace(0, 1, num_ode_steps + 1, device=device)

    results = {}

    for step_i in range(num_ode_steps):
        t_current = t_values[step_i]
        t_batch = torch.full((1,), t_current.item(), device=device)

        # Velocity prediction
        v_pred, _ = model(z, t_batch, decoder_step=False)

        # Euler step
        z = z + v_pred * dt

        # Check if we need a snapshot
        next_t = t_values[step_i + 1].item()
        for snap_t in list(snapshot_times):
            if t_current.item() <= snap_t < next_t:
                t_decode = torch.full((1,), min(snap_t, 0.999), device=device)
                _, logits = model(z, t_decode, decoder_step=True)

                # Get softmax probabilities
                probs = F.softmax(logits[0], dim=-1)  # [seq_len, vocab]

                # Average probability across all token positions
                avg_probs = probs.mean(dim=0)  # [vocab]

                # Get top-k tokens
                topk_values, topk_indices = torch.topk(avg_probs, top_k)

                top_tokens = []
                for val, idx in zip(topk_values.tolist(), topk_indices.tolist()):
                    token_text = tokenizer.decode([idx])
                    top_tokens.append([token_text.strip(), round(val, 4)])

                results[str(snap_t)] = {
                    "top_tokens": top_tokens,
                    "top_k": top_k,
                    "entropy": float(-(avg_probs * torch.log(avg_probs + 1e-10)).sum()),
                }

                snapshot_times.remove(snap_t)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint .pt")
    parser.add_argument("--tokenizer_path", required=True, help="Path to tokenizer.json")
    parser.add_argument("--text", required=True, help="Input text for the running example")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--R", type=float, default=None, help="Known R value for labeling")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load tokenizer
    tokenizer = BPETokenizer.load(args.tokenizer_path)

    # Load model
    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        depth=6,
    )
    model = ELFModel(config)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    snapshot_times = [0.3, 0.6, 0.9]

    results = extract_prob_distributions(
        model=model,
        tokenizer=tokenizer,
        text=args.text,
        snapshot_times=snapshot_times,
        device=device,
    )

    output = {
        "text": args.text,
        "R": args.R,
        "checkpoint": args.checkpoint,
        "snapshots": results,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Saved to {args.output}")
    for t, snap in results.items():
        print(f"\nt={t} (entropy={snap['entropy']:.3f}):")
        for token, prob in snap["top_tokens"][:5]:
            print(f"  {token:20s} {prob:.4f}")


if __name__ == "__main__":
    main()
