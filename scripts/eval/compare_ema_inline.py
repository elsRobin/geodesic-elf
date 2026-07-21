#!/usr/bin/env python3
"""
Quick EMA geodesic ratio computation for two checkpoints.

Usage:
    python scripts/local_compare_ema.py \
        --ckpt_1 /path/to/checkpoint-70000 \
        --ckpt_2 /path/to/checkpoint-80000 \
        --data data/gsm8k/train.txt \
        --tokenizer /path/to/tokenizer.json
"""

import os, json, argparse, torch
from elf.config import ELFConfig
from elf.model import ELFModel
from elf.data.tokenizer import BPETokenizer
from experiments.geodesic_analysis.geodesic import geodesic_energy


def compute_ema_ratio(ckpt_dir: str, tokenizer_path: str, texts: list,
                      n_runs: int = 5) -> dict:
    """Compute EMA geodesic ratio for one checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg_path = os.path.join(ckpt_dir, "config.json")
    ckpt_path = os.path.join(ckpt_dir, "ema_model.pt")

    cfg = ELFConfig.from_dict(json.load(open(cfg_path)))
    model = ELFModel(cfg.model).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    tokenizer = BPETokenizer.load(tokenizer_path, vocab_size=cfg.model.vocab_size)
    model.eval()

    ratios = []
    for _ in range(n_runs):
        for text in texts:
            ids = tokenizer.encode(text)[:80]
            ids_t = torch.tensor([ids], device=device)
            x0 = model.embedding(ids_t)
            eps = torch.randn_like(x0)
            z = 0.2 * x0 + 0.8 * 2.0 * eps
            dt = 0.8 / 32
            trajectory = [z.clone()]
            for s in range(32):
                t_batch = torch.tensor([0.2 + s * dt], device=device)
                v_pred, _ = model(z, t_batch, decoder_step=False)
                z = z + v_pred * dt
                trajectory.append(z.clone())
            trajectory = torch.stack(trajectory)

            straight_e = geodesic_energy(model, trajectory[0, 0], trajectory[-1, 0], num_points=8)
            ode_e = sum(
                geodesic_energy(model, trajectory[i * 4, 0], trajectory[min(i * 4 + 4, 32), 0],
                                num_points=4).item()
                for i in range(8)
            )
            ratios.append(ode_e / max(straight_e.item(), 1e-8))

    avg = sum(ratios) / len(ratios)
    std = torch.tensor(ratios).std().item() if len(ratios) > 1 else 0.0
    return {"ratio": round(avg, 4), "std": round(std, 4), "n_samples": len(ratios)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_1", type=str, required=True)
    parser.add_argument("--ckpt_2", type=str, required=True)
    parser.add_argument("--data", type=str, default="data/gsm8k/train.txt")
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--num_texts", type=int, default=10)
    parser.add_argument("--num_runs", type=int, default=5)

    args = parser.parse_args()

    # Load texts
    with open(args.data, encoding="utf-8") as f:
        lines = [l.strip() for l in f if 60 < len(l.strip()) < 200]
    texts = lines[:args.num_texts]
    print(f"Loaded {len(texts)} texts")

    # Compute
    for label, ckpt in [("Checkpoint-70000 (pre-collapse)", args.ckpt_1),
                         ("Checkpoint-80000 (post-collapse)", args.ckpt_2)]:
        r = compute_ema_ratio(ckpt, args.tokenizer, texts, args.num_runs)
        print(f"  {label}")
        print(f"    EMA Ratio: {r['ratio']} ± {r['std']}  ({r['n_samples']} samples)")


if __name__ == "__main__":
    main()
