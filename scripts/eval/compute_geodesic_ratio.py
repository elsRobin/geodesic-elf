#!/usr/bin/env python3
"""
Standalone script: Load an ELF checkpoint and compute geodesic energy ratio R(k).

Reproduces the paper's key result: R(k) ~1.82 (Regime M) vs R(k) ~0.35 (Regime S).
Uses the same protocol as the paper: inline eval, EMA weights, 10 texts x N runs.

Usage:
    python eval_checkpoint.py --checkpoint quick_change/checkpoint-70000
    python eval_checkpoint.py --checkpoint quick_change/checkpoint-70000 --text "Janet's ducks..."
    python eval_checkpoint.py --checkpoint quick_change/checkpoint-80000 --data data/gsm8k/train.txt --num_texts 10 --num_runs 3
"""

import argparse
import json
import math
import os
import sys
import torch
import torch.nn.functional as F

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_checkpoint(checkpoint_dir, device):
    """Load ELF model from checkpoint directory. Prefers EMA weights."""
    from elf.config import ELFConfig
    from elf.model import ELFModel

    config_path = os.path.join(checkpoint_dir, "config.json")
    with open(config_path, "r") as f:
        config = ELFConfig.from_dict(json.load(f))

    model = ELFModel(config.model).to(device)

    ema_path = os.path.join(checkpoint_dir, "ema_model.pt")
    if os.path.exists(ema_path):
        model.load_state_dict(torch.load(ema_path, map_location=device, weights_only=False))
        print(f"  Loaded EMA weights ({os.path.getsize(ema_path)/1e6:.0f} MB)")
    else:
        model_path = os.path.join(checkpoint_dir, "model.pt")
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
        print(f"  Loaded training weights ({os.path.getsize(model_path)/1e6:.0f} MB)")

    model.eval()
    return model, config


@torch.no_grad()
def geodesic_energy(model, z_start, z_end, num_points=16, temperature=1.0):
    """
    Compute geodesic energy along a straight-line interpolation in latent space.

    Decodes each interpolated point through the LM Head at t=1.0 to get
    softmax probabilities, then sums squared differences between adjacent
    probability distributions. Adapted from RelativeGeodesics (Yu et al., 2024).

    Args:
        model: ELFModel in eval mode
        z_start, z_end: latent vectors, each (N, D)
        num_points: number of interpolation points (capped at 32)
        temperature: softmax temperature
    Returns:
        scalar energy (sum of squared probability differences)
    """
    device = next(model.parameters()).device

    if z_start.dim() == 1:
        z_start = z_start.unsqueeze(0)
    if z_end.dim() == 1:
        z_end = z_end.unsqueeze(0)
    z_start, z_end = z_start.to(device), z_end.to(device)

    num_points = min(num_points, 32)
    alphas = torch.linspace(0, 1, num_points, device=device).view(-1, 1, 1)
    z_path = (1 - alphas) * z_start.unsqueeze(0) + alphas * z_end.unsqueeze(0)

    # Decode each interpolated point through LM Head at t=1.0
    t_fake = torch.ones(1, device=device)
    outputs = []
    for point in z_path:
        _, logits = model(point.unsqueeze(0), t_fake, decoder_step=True)
        probs = F.softmax(logits / temperature, dim=-1)
        outputs.append(probs.squeeze(0))
    outputs = torch.stack(outputs, dim=0)           # (P, N, vocab)
    diffs = outputs[1:] - outputs[:-1]              # (P-1, N, vocab)
    return (diffs ** 2).sum()


@torch.no_grad()
def compute_rk(model, tokenizer, texts, t_start=0.2, noise_scale=2.0,
               num_steps=32, num_segments=8,
               straight_points=16, segment_points=8):
    """
    Compute geodesic energy ratio R(k) over a set of texts.

    Protocol (matching paper Section 3):
      (1) Tokenize + embed text → x0
      (2) Forward-diffuse: z = t*x0 + (1-t)*σ*ε  at t=t_start
      (3) ODE integration (Euler, num_steps steps, t_start → 1.0)
      (4) E_geo(straight) = geodesic_energy between endpoints
      (5) E_geo(ODE) = sum of geodesic_energy over ODE segments
      (6) R = E_ode / E_straight

    Returns {"ratio": float, "ratios": [per-text values]}
    """
    device = next(model.parameters()).device

    all_ratios = []
    for text in texts:
        ids = tokenizer.encode(text)[:80]
        ids_tensor = torch.tensor([ids], device=device)
        x0 = model.embedding(ids_tensor)                     # (1, N, D)

        # Forward diffusion
        eps = torch.randn_like(x0)
        z = t_start * x0 + (1 - t_start) * noise_scale * eps

        # ODE integration (Euler)
        dt = (1.0 - t_start) / num_steps
        trajectory = [z.clone()]
        for s in range(num_steps):
            t_val = t_start + s * dt
            t_batch = torch.tensor([t_val], device=device)
            v_pred, _ = model(z, t_batch, decoder_step=False)
            z = z + v_pred * dt
            trajectory.append(z.clone())

        traj = torch.stack(trajectory)                       # (T, 1, N, D)

        # Euclidean straight-line baseline
        straight_e = geodesic_energy(
            model, traj[0, 0], traj[-1, 0], num_points=straight_points
        )

        # ODE path energy (sum over segments)
        step_size = num_steps // num_segments
        total_ode = 0.0
        for i in range(0, num_steps, step_size):
            e = geodesic_energy(
                model,
                traj[i, 0],
                traj[min(i + step_size, num_steps), 0],
                num_points=segment_points,
            )
            total_ode += e.item()

        all_ratios.append(total_ode / max(straight_e.item(), 1e-8))

    avg = sum(all_ratios) / len(all_ratios)
    std = torch.tensor(all_ratios).std().item() if len(all_ratios) > 1 else 0.0
    return {"ratio": round(avg, 4), "std": round(std, 4), "ratios": all_ratios}


def classify_regime(r_mean, r_std):
    """Map R(k) value to geometric regime (paper Section 5.1)."""
    if r_mean > 1.15:
        return f"Regime M (manifold-aligned). ODE path follows manifold curvature."
    elif r_mean > 0.85:
        return f"Close to Euclidean. Path is geometrically trivial."
    else:
        return f"Regime S (shortcut). ODE path is more direct than straight line."


def main():
    parser = argparse.ArgumentParser(
        description="ELF Checkpoint Geodesic Energy Ratio R(k) Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python eval_checkpoint.py -c quick_change/checkpoint-70000
  python eval_checkpoint.py -c quick_change/checkpoint-80000 --num_runs 3
  python eval_checkpoint.py -c run1-100K --data data/gsm8k/train.txt -n 10 -r 5
        """,
    )
    parser.add_argument("-c", "--checkpoint", required=True,
                        help="Checkpoint directory (config.json + ema_model.pt)")
    parser.add_argument("-t", "--tokenizer", default=None,
                        help="tokenizer.json path (default: ../quick_change/tokenizer.json)")
    parser.add_argument("-d", "--data", default=None,
                        help="Text file for eval (one per line)")
    parser.add_argument("--text", default=None,
                        help="Single text for eval (bypasses --data)")
    parser.add_argument("-n", "--num_texts", type=int, default=10,
                        help="Number of texts to evaluate (default: 10)")
    parser.add_argument("-r", "--num_runs", type=int, default=1,
                        help="Runs per text with different noise seeds (default: 1)")
    parser.add_argument("--t_start", type=float, default=0.2)
    parser.add_argument("--noise_scale", type=float, default=2.0)
    parser.add_argument("--num_steps", type=int, default=32)
    parser.add_argument("--num_segments", type=int, default=8)
    parser.add_argument("--straight_points", type=int, default=16)
    parser.add_argument("--segment_points", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # ---- Device ----
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)
    print(f"Device: {device}")

    # ---- Tokenizer path ----
    tokenizer_path = args.tokenizer
    if tokenizer_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        tokenizer_path = os.path.join(script_dir, "..", "quick_change", "tokenizer.json")
        tokenizer_path = os.path.normpath(tokenizer_path)

    # ---- Load model ----
    print(f"\n[1/3] Loading checkpoint: {args.checkpoint}")
    model, config = load_checkpoint(args.checkpoint, device)
    print(f"       Model: ~{sum(p.numel() for p in model.parameters())/1e6:.1f}M params, "
          f"vocab={config.model.vocab_size}")

    # ---- Load tokenizer ----
    print(f"\n[2/3] Loading tokenizer: {tokenizer_path}")
    from elf.data.tokenizer import BPETokenizer
    tokenizer = BPETokenizer.load(tokenizer_path, vocab_size=config.model.vocab_size)
    print(f"       Vocab size: {tokenizer.vocab_size}")

    # ---- Prepare texts ----
    if args.text:
        texts = [args.text]
        print(f"\n[3/3] Evaluating single text ({len(args.text)} chars)")
    elif args.data and os.path.exists(args.data):
        with open(args.data, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if 40 < len(l.strip()) < 300]
        texts = lines[:args.num_texts]
        print(f"\n[3/3] Evaluating {len(texts)} texts from {args.data}")
    else:
        # Default: use the paper's running example
        texts = [
            "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast and "
            "uses 4 to bake muffins daily. She sells the remainder for $2 each. "
            "How much does she earn in 5 days?",
        ]
        print(f"\n[3/3] Using paper running example (1 text)")

    # ---- Compute R(k) ----
    print(f"\n{'='*60}")
    print(f"Computing R(k): {len(texts)} texts x {args.num_runs} runs")
    print(f"  t_start={args.t_start}, σ={args.noise_scale}, "
          f"ODE steps={args.num_steps}, segments={args.num_segments}")
    print(f"  Interp: straight={args.straight_points}pts, segment={args.segment_points}pts")
    print(f"{'='*60}")

    all_ratios = []
    for run in range(args.num_runs):
        torch.manual_seed(42 + run)  # reproducible per-run seeding
        result = compute_rk(
            model, tokenizer, texts,
            t_start=args.t_start, noise_scale=args.noise_scale,
            num_steps=args.num_steps, num_segments=args.num_segments,
            straight_points=args.straight_points,
            segment_points=args.segment_points,
        )
        all_ratios.extend(result["ratios"])
        if args.num_runs > 1:
            print(f"  Run {run+1}: R(k) = {result['ratio']:.4f} ± {result['std']:.4f}")

    final_avg = sum(all_ratios) / len(all_ratios)
    final_std = torch.tensor(all_ratios).std().item() if len(all_ratios) > 1 else 0.0

    print(f"\n{'='*60}")
    print(f"  R(k) = {final_avg:.4f} ± {final_std:.4f}  (n={len(all_ratios)} samples)")
    print(f"  --> {classify_regime(final_avg, final_std)}")
    print(f"{'='*60}")

    # ---- Reference values from the paper ----
    print(f"\nPaper reference values (for comparison):")
    print(f"  Regime M:  R ≈ 1.82 ± 0.05  (Runs 1 & 3 at 80K-100K)")
    print(f"  Regime S:  R ≈ 0.35 ± 0.04  (Run 2 at 80K-100K, post-collapse)")
    print(f"  Phase transition (Run 2): 65K→80K, R drops 1.17→0.42")


if __name__ == "__main__":
    main()
