#!/usr/bin/env python3
"""
LR Warmup Intervention — Reversibility of Low-R Regime.

Loads a low-R checkpoint, warms up LR from 3e-6 to 3e-4 over 5K steps,
then cosine-decays back. Tests whether the low-R geometric regime is
reversible under LR intervention.

The training loop:
  Step    0 ->  5K:  Linear warmup  LR = 3e-6 -> 3e-4
  Step  5K -> 15K:  Cosine decay    LR = 3e-4 -> 3e-6
  Step 15K -> 20K:  Hold at min     LR = 3e-6

Inline geodesic eval at every 1K steps to track R(k) response.

Usage:
    python scripts/analysis/lr_intervention.py \\
        --checkpoint checkpoints/model/checkpoint-100000 \\
        --data_dir data/train \\
        --output_dir checkpoints/intervene-warmup \\
        --eval_data data/train/train.txt \\
        --eval_tokenizer tokenizer.json
"""

import argparse
import copy
import json
import math
import os
import sys

import torch
import torch.nn.functional as F

EPS = 1e-10

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from elf.config import ELFConfig
from elf.model import ELFModel, update_ema
from elf.data.tokenizer import BPETokenizer
from elf.data.dataloader import create_dataloader
from elf.training import train_step


# ── Custom LR scheduler for warmup intervention ──────────────────
def intervention_lr(step: int,
                    warmup_steps: int = 5000,
                    decay_steps: int = 10000,
                    lr_min: float = 3e-6,
                    lr_peak: float = 3e-4) -> float:
    """LR schedule: warmup → cosine decay → hold."""
    if step < warmup_steps:
        return lr_min + (lr_peak - lr_min) * (step / max(1, warmup_steps))
    elif step < warmup_steps + decay_steps:
        progress = (step - warmup_steps) / max(1, decay_steps)
        return lr_min + (lr_peak - lr_min) * 0.5 * (1 + math.cos(math.pi * progress))
    else:
        return lr_min


# ── Inline geodesic eval (same protocol as paper) ────────────────
@torch.no_grad()
def inline_geodesic_ratio(
    model: ELFModel, tokenizer: BPETokenizer,
    texts: list[str], n_runs: int = 5,
    t_start: float = 0.2, noise_scale: float = 2.0,
    num_steps: int = 32, num_segments: int = 8,
) -> tuple[float, float]:
    """Compute R(k) for inline evaluation."""
    device = next(model.parameters()).device
    all_ratios = []

    for text in texts:
        ids = tokenizer.encode(text)[:80]
        ids_tensor = torch.tensor([ids], device=device)
        x0 = model.embedding(ids_tensor)

        for _ in range(n_runs):
            eps = torch.randn_like(x0)
            z = t_start * x0 + (1 - t_start) * noise_scale * eps

            dt = (1.0 - t_start) / num_steps
            trajectory = [z.clone()]
            for s in range(num_steps):
                t_val = t_start + s * dt
                t_batch = torch.tensor([t_val], device=device)
                v_pred, _ = model(z, t_batch, decoder_step=False)
                z = z + v_pred * dt
                trajectory.append(z.clone())

            traj = torch.stack(trajectory)

            # Euclidean straight-line baseline
            alphas = torch.linspace(0, 1, 16, device=device).view(-1, 1, 1)
            z_path = ((1 - alphas) * traj[0, 0].unsqueeze(0)
                      + alphas * traj[-1, 0].unsqueeze(0))
            t_fake = torch.ones(1, device=device)
            outputs = []
            for pt in z_path:
                _, logits = model(pt.unsqueeze(0), t_fake, decoder_step=True)
                outputs.append(F.softmax(logits, dim=-1).squeeze(0))
            outputs = torch.stack(outputs)
            straight_e = 0.0
            for i in range(len(outputs) - 1):
                p = outputs[i].clamp(min=EPS)
                q = outputs[i + 1].clamp(min=EPS)
                m = 0.5 * (p + q)
                straight_e += float(0.5 * torch.sum(
                    p * (p.log() - m.log()) + q * (q.log() - m.log())))

            # ODE path energy
            step_size = num_steps // num_segments
            total_ode = 0.0
            for i in range(0, num_steps, step_size):
                alphas_s = torch.linspace(0, 1, 8, device=device).view(-1, 1, 1)
                z_seg = ((1 - alphas_s) * traj[i, 0].unsqueeze(0)
                         + alphas_s * traj[min(i + step_size, num_steps), 0].unsqueeze(0))
                seg_out = []
                for pt in z_seg:
                    _, logits_pt = model(pt.unsqueeze(0), t_fake, decoder_step=True)
                    seg_out.append(F.softmax(logits_pt, dim=-1).squeeze(0))
                seg_out = torch.stack(seg_out)
                for j in range(len(seg_out) - 1):
                    p = seg_out[j].clamp(min=EPS)
                    q = seg_out[j + 1].clamp(min=EPS)
                    m = 0.5 * (p + q)
                    total_ode += float(0.5 * torch.sum(
                        p * (p.log() - m.log()) + q * (q.log() - m.log())))

            all_ratios.append(total_ode / max(straight_e, 1e-8))

    mean = sum(all_ratios) / len(all_ratios)
    std = torch.tensor(all_ratios).std().item() if len(all_ratios) > 1 else 0.0
    return mean, std


# ── Main intervention loop ───────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="LR Warmup Intervention on Regime S")
    parser.add_argument("--checkpoint", required=True,
                        help="Regime S checkpoint dir (Run2-100K)")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--eval_data", default=None)
    parser.add_argument("--eval_tokenizer", default=None)
    parser.add_argument("--total_steps", type=int, default=15000)
    parser.add_argument("--warmup_steps", type=int, default=5000)
    parser.add_argument("--lr_peak", type=float, default=3e-4)
    parser.add_argument("--lr_min", type=float, default=3e-6)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--eval_n_texts", type=int, default=10)
    parser.add_argument("--eval_n_runs", type=int, default=5)
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load checkpoint ──
    print(f"\n[1] Loading Regime S checkpoint: {args.checkpoint}")
    config_path = os.path.join(args.checkpoint, "config.json")
    with open(config_path) as f:
        config = ELFConfig.from_dict(json.load(f))
    model = ELFModel(config.model).to(device)
    model.load_state_dict(torch.load(
        os.path.join(args.checkpoint, "model.pt"),
        map_location=device, weights_only=False))

    ema_model = copy.deepcopy(model)
    ema_path = os.path.join(args.checkpoint, "ema_model.pt")
    if os.path.exists(ema_path):
        ema_model.load_state_dict(torch.load(ema_path, map_location=device,
                                             weights_only=False))
        print("  EMA weights loaded.")

    model.train()
    ema_model.eval()
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Params: {params:.1f}M")

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr_min,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )

    # ── Dataloader ──
    dataloader = create_dataloader(
        args.data_dir, config.data, batch_size=args.batch_size, shuffle=True)

    # ── Eval data ──
    eval_texts = None
    eval_tokenizer = None
    if args.eval_data:
        with open(args.eval_data, encoding="utf-8") as f:
            eval_texts = [l.strip() for l in f
                          if 60 < len(l.strip()) < 300][:args.eval_n_texts]
        eval_tokenizer = BPETokenizer.load(args.eval_tokenizer, vocab_size=8192)
        print(f"  Eval texts: {len(eval_texts)}")

    # ── Intervention loop ──
    print(f"\n[2] Intervention: {args.total_steps} steps")
    print(f"    Warmup: {args.warmup_steps} steps, "
          f"LR {args.lr_min:.0e} → {args.lr_peak:.0e}")
    print(f"    Decay:  {args.total_steps - args.warmup_steps} steps, "
          f"LR {args.lr_peak:.0e} → {args.lr_min:.0e}")
    print(f"    Eval:   every {args.eval_every} steps\n")

    results = []
    step = 0

    while step < args.total_steps:
        for batch in dataloader:
            if step >= args.total_steps:
                break

            # Update LR
            lr = intervention_lr(step, args.warmup_steps,
                                 args.total_steps - args.warmup_steps,
                                 args.lr_min, args.lr_peak)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # Train step
            ids = batch.to(device)
            loss, denoise_loss, decode_loss = train_step(
                model, ids, config.model, config.training)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # EMA update
            update_ema(ema_model, model, args.ema_decay)

            # Log loss
            if step % 100 == 0:
                print(f"  step {step:>5d}: loss={loss.item():.4f}  "
                      f"LR={lr:.2e}")

            # Inline geodesic eval
            if eval_texts and step % args.eval_every == 0:
                model.eval()
                r_mean, r_std = inline_geodesic_ratio(
                    model, eval_tokenizer, eval_texts, n_runs=args.eval_n_runs)
                model.train()
                results.append({
                    "step": step, "ratio": round(r_mean, 4),
                    "std": round(r_std, 4), "lr": lr,
                })
                regime = "M" if r_mean > 1.15 else "S" if r_mean < 0.85 else "E"
                print(f"    → R(k) = {r_mean:.4f} ± {r_std:.4f}  [{regime}]  "
                      f"LR={lr:.2e}")

            step += 1

    # ── Save ──
    output_file = os.path.join(args.output_dir, "lr_warmup_geodesic.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Done] Saved: {output_file}")

    if results:
        initial_r = results[0]["ratio"]
        final_r = results[-1]["ratio"]
        print(f"  R(k) change: {initial_r:.4f} → {final_r:.4f}  "
              f"(Δ = {final_r - initial_r:+.4f})")


if __name__ == "__main__":
    main()
