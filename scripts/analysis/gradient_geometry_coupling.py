#!/usr/bin/env python3
"""
Gradient Norm vs |dR/dk| Correlation Tracking.

Trains a model for 100K steps with dual-branch objective and inline
geodesic eval, plus gradient norm logging every 100 steps.
Computes Spearman correlation between |dR/dk| (geometric change rate)
and ||gradient|| (gradient magnitude) across training phases.

Tests the Optimization-Geometry Coupling conjecture:
  Corr(|dR/dk|, ||gradient||) > 0 during active learning.

Output: training checkpoint + gradient_norms.json + geodesic_inline.json
        + correlation analysis report.

Usage:
    python scripts/analysis/gradient_geometry_coupling.py \\
        --data_dir data/train \\
        --output_dir checkpoints/gradcoupling \\
        --max_steps 100000 \\
        --eval_data data/train/train.txt
"""

import argparse
import copy
import json
import math
import os
import re
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from elf.config import ModelConfig, TrainingConfig, ELFConfig
from elf.model import ELFModel, update_ema
from elf.diffusion import sample_log_normal_time, sample_decode_time
from elf.data.tokenizer import BPETokenizer
from elf.data.dataloader import create_dataloader

EPS = 1e-10


# ── Losses (inline copies to avoid import side effects) ──────────
def denoising_loss_fn(model, input_ids, t, noise_scale=2.0, t_eps=1e-2):
    B = input_ids.shape[0]
    x0 = model.embedding(input_ids)
    eps = torch.randn_like(x0)
    t_expanded = t.view(B, 1, 1)
    z = t_expanded * x0 + (1 - t_expanded) * noise_scale * eps
    v_target = x0 - noise_scale * eps
    v_pred, _ = model(z, t, decoder_step=False)
    return F.mse_loss(v_pred, v_target)


def decoding_loss_fn(model, input_ids, t, noise_scale=0.1):
    B = input_ids.shape[0]
    x0 = model.embedding(input_ids)
    eps = torch.randn_like(x0)
    t_expanded = t.view(B, 1, 1)
    z = t_expanded * x0 + (1 - t_expanded) * noise_scale * eps
    _, logits = model(z, t, decoder_step=True)
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        input_ids.view(-1),
        ignore_index=-100,
    )


def get_lr_scheduler(optimizer, warmup_steps, total_steps, schedule_type="cosine"):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        if schedule_type == "cosine":
            return 0.5 * (1 + math.cos(math.pi * progress))
        return 1.0 - progress
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── Inline geodesic eval ─────────────────────────────────────────
@torch.no_grad()
def compute_rk(model, tokenizer, texts, n_runs=5, n_segments=8):
    device = next(model.parameters()).device
    all_ratios = []
    for text in texts:
        ids = tokenizer.encode(text)[:80]
        ids_tensor = torch.tensor([ids], device=device)
        x0 = model.embedding(ids_tensor)
        for _ in range(n_runs):
            eps = torch.randn_like(x0)
            z = 0.2 * x0 + 0.8 * 2.0 * eps
            dt = 0.8 / 32
            traj = [z.clone()]
            for s in range(32):
                t_val = 0.2 + s * dt
                t_b = torch.tensor([t_val], device=device)
                v_pred, _ = model(z, t_b, decoder_step=False)
                z = z + v_pred * dt
                traj.append(z.clone())
            traj = torch.stack(traj)
            # Baseline
            alphas = torch.linspace(0, 1, 16, device=device).view(-1, 1, 1)
            zp = (1-alphas)*traj[0,0].unsqueeze(0)+alphas*traj[-1,0].unsqueeze(0)
            tf = torch.ones(1, device=device)
            probs = []
            for pt in zp:
                _, logits = model(pt.unsqueeze(0), tf, decoder_step=True)
                probs.append(F.softmax(logits, dim=-1).squeeze(0))
            probs = torch.stack(probs)
            s_e = 0.0
            for i in range(len(probs)-1):
                p = probs[i].clamp(min=EPS)
                q = probs[i+1].clamp(min=EPS)
                m = 0.5*(p+q)
                s_e += float(0.5*torch.sum(p*(p.log()-m.log())+q*(q.log()-m.log())))
            # ODE
            step_sz = 32 // n_segments
            ode_e = 0.0
            for i in range(0, 32, step_sz):
                end = min(i+step_sz, 32)
                al = torch.linspace(0,1,8,device=device).view(-1,1,1)
                zp2 = (1-al)*traj[i,0].unsqueeze(0)+al*traj[end,0].unsqueeze(0)
                sp = []
                for pt in zp2:
                    _, lgt = model(pt.unsqueeze(0), tf, decoder_step=True)
                    sp.append(F.softmax(lgt, dim=-1).squeeze(0))
                sp = torch.stack(sp)
                for j in range(len(sp)-1):
                    pj = sp[j].clamp(min=EPS)
                    qj = sp[j+1].clamp(min=EPS)
                    mj = 0.5*(pj+qj)
                    ode_e += float(0.5*torch.sum(pj*(pj.log()-mj.log())+qj*(qj.log()-mj.log())))
            all_ratios.append(ode_e / max(s_e, 1e-8))
    return sum(all_ratios)/len(all_ratios), torch.tensor(all_ratios).std().item()


# ── Gradient norm computation ────────────────────────────────────
@torch.no_grad()
def compute_grad_norms(model):
    """Compute L2 norm of gradients for all parameters."""
    total_sq = 0.0
    n_params = 0
    for p in model.parameters():
        if p.grad is not None:
            total_sq += p.grad.norm(2).item() ** 2
            n_params += p.numel()
    return math.sqrt(total_sq)


# ── Main ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Gradient-Geometry Coupling Experiment")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--decoder_prob", type=float, default=0.5)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--grad_log_every", type=int, default=100)
    parser.add_argument("--eval_data", default=None)
    parser.add_argument("--eval_n_texts", type=int, default=10)
    parser.add_argument("--eval_n_runs", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Device: {device}")

    # ── Tokenizer ──
    tok_path = os.path.join(args.output_dir, "tokenizer.json")
    from elf.data.tokenizer import BPETokenizer as BPE
    if not os.path.exists(tok_path):
        print("[Tokenizer] Training BPE...")
        t = BPE(vocab_size=8192)
        def gen():
            with open(os.path.join(args.data_dir, "train.txt")) as f:
                for line in f:
                    line = line.strip()
                    if line: yield line
        t.train_stream(gen(), output_dir=args.output_dir, max_texts=20000)
    tokenizer = BPE.load(tok_path, vocab_size=8192)
    eval_tokenizer = BPE.load(tok_path, vocab_size=8192)

    # ── Eval texts ──
    eval_texts = None
    if args.eval_data:
        with open(args.eval_data, encoding="utf-8") as f:
            eval_texts = [l.strip() for l in f
                          if 60 < len(l.strip()) < 300][:args.eval_n_texts]
        print(f"Eval texts: {len(eval_texts)}")

    # ── Model ──
    model_cfg = ModelConfig(
        vocab_size=8192, embed_dim=512, depth=8, num_heads=8,
        max_seq_len=256, ff_mult=4, rope_theta=10000.0,
        qk_norm=True, use_swiglu=True, dropout=0.0,
        weight_tying=True, bidirectional=True,
    )
    model = ELFModel(model_cfg).to(device)
    ema_model = copy.deepcopy(model)
    print(f"Model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    # ── Optimizer ──
    decode_names = {"embedding.weight", "final_norm.weight"}
    decode_params, base_params = [], []
    for name, param in model.named_parameters():
        (decode_params if name in decode_names else base_params).append(param)
    optimizer = torch.optim.AdamW([
        {"params": base_params, "lr": args.lr},
        {"params": decode_params, "lr": args.lr * 2.0},
    ], betas=(0.9, 0.999), weight_decay=0.01)
    scheduler = get_lr_scheduler(optimizer, 2000, args.max_steps, "cosine")

    # ── Dataloader ──
    data_cfg = type("obj", (object,), {
        "text_field": "text", "max_texts": None, "tokenizer_path": tok_path,
        "vocab_size": 8192, "max_seq_len": 256,
    })()
    dataloader = create_dataloader(args.data_dir, data_cfg,
                                   batch_size=args.batch_size, shuffle=True)

    # ── Training loop ──
    model.train()
    step = 0
    grad_log = []        # [{step, grad_norm, phase}]
    geodesic_log = []    # [{step, ratio, std}]
    start_time = time.time()

    print(f"\n{'='*60}")
    print(f"  Training {args.max_steps} steps with gradient logging")
    print(f"{'='*60}\n")

    while step < args.max_steps:
        for batch in dataloader:
            if step >= args.max_steps:
                break

            ids = batch.to(device) if isinstance(batch, torch.Tensor) \
                else batch["input_ids"].to(device)
            B = ids.shape[0]

            is_decoder = torch.rand(1).item() < args.decoder_prob
            if is_decoder:
                t = torch.rand(B, device=device) * 0.15 + 0.85
                loss = decoding_loss_fn(model, ids, t) / args.decoder_prob
            else:
                t = torch.randn(B, device=device).exp() * 1.5
                t = t.clamp(max=0.999)
                loss = denoising_loss_fn(model, ids, t) / (1 - args.decoder_prob)

            loss.backward()

            # Gradient norm logging
            if step % args.grad_log_every == 0:
                gn = compute_grad_norms(model)
                # Phase classification
                if step < 5000:
                    phase = "I"
                elif step < 30000:
                    phase = "II"
                elif step < 60000:
                    phase = "III"
                else:
                    phase = "IV"
                grad_log.append({"step": step, "grad_norm": gn, "phase": phase,
                                 "lr": scheduler.get_last_lr()[0]})

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            # EMA update
            update_ema(ema_model, model, 0.9999)

            if step % 200 == 0:
                elapsed = time.time() - start_time
                tok_sec = step * args.batch_size * 256 / max(elapsed, 1)
                print(f"  step {step:>6d}: loss={loss.item():.4f}  "
                      f"LR={scheduler.get_last_lr()[0]:.2e}  "
                      f"{tok_sec/1000:.0f}K tok/s")

            # Inline geodesic eval
            if eval_texts and step > 0 and step % args.eval_every == 0:
                model.eval()
                r_mean, r_std = compute_rk(
                    model, eval_tokenizer, eval_texts,
                    n_runs=args.eval_n_runs)
                model.train()
                geodesic_log.append({
                    "step": step, "ratio": round(r_mean, 4),
                    "std": round(r_std, 4),
                    "lr": scheduler.get_last_lr()[0],
                })
                regime = "M" if r_mean > 1.15 else "S" if r_mean < 0.85 else "E"
                print(f"    → R(k) = {r_mean:.4f} ± {r_std:.4f}  [{regime}]")

            # Save checkpoint
            if step > 0 and step % args.save_every == 0:
                ckpt_path = os.path.join(args.output_dir, f"checkpoint-{step}")
                os.makedirs(ckpt_path, exist_ok=True)
                torch.save(model.state_dict(),
                           os.path.join(ckpt_path, "model.pt"))
                torch.save(ema_model.state_dict(),
                           os.path.join(ckpt_path, "ema_model.pt"))
                torch.save(optimizer.state_dict(),
                           os.path.join(ckpt_path, "optimizer.pt"))
                config = ELFConfig(model=model_cfg, training=TrainingConfig(
                    output_dir=args.output_dir, max_steps=args.max_steps,
                    learning_rate=args.lr, decoder_prob=args.decoder_prob,
                ))
                with open(os.path.join(ckpt_path, "config.json"), "w") as f:
                    json.dump(config.to_dict(), f, indent=2)
                print(f"    [Saved checkpoint-{step}]")

            step += 1

    # ── Save logs ──
    grad_file = os.path.join(args.output_dir, "gradient_norms.json")
    with open(grad_file, "w", encoding="utf-8") as f:
        json.dump(grad_log, f, indent=2)

    geo_file = os.path.join(args.output_dir, "geodesic_inline.json")
    with open(geo_file, "w", encoding="utf-8") as f:
        json.dump(geodesic_log, f, indent=2)

    # ── Correlation analysis ──
    if geodesic_log and grad_log:
        print(f"\n{'='*60}")
        print("  Correlation Analysis: |dR/dk| vs ||∇L||")
        print(f"{'='*60}")

        # Align gradient norms to geodesic eval steps
        from collections import defaultdict
        phase_grad = defaultdict(list)
        phase_dr = defaultdict(list)

        geo_by_step = {g["step"]: g["ratio"] for g in geodesic_log}
        steps = sorted(geo_by_step.keys())

        # |dR/dk| at geodesic eval points
        for i in range(1, len(steps)):
            dr = abs(geo_by_step[steps[i]] - geo_by_step[steps[i-1]])
            phase = "I" if steps[i] < 5000 else \
                    "II" if steps[i] < 30000 else \
                    "III" if steps[i] < 60000 else "IV"
            # Average grad norm in the window
            window_gns = [g["grad_norm"] for g in grad_log
                          if steps[i-1] <= g["step"] < steps[i]]
            if window_gns:
                phase_dr[phase].append(dr)
                phase_grad[phase].append(sum(window_gns) / len(window_gns))

        import scipy.stats as stats
        for phase in ["I", "II", "III", "IV"]:
            if len(phase_dr[phase]) >= 5:
                rho, pval = stats.spearmanr(phase_dr[phase], phase_grad[phase])
                print(f"  Phase {phase}: ρ = {rho:.3f}  (p = {pval:.3f})  "
                      f"n = {len(phase_dr[phase])}")
            else:
                print(f"  Phase {phase}: insufficient data "
                      f"(n = {len(phase_dr[phase])})")

        # Save correlation data
        corr_file = os.path.join(args.output_dir, "correlation.json")
        corr_data = {
            phase: {"dr": phase_dr[phase], "grad_norm": phase_grad[phase]}
            for phase in ["I", "II", "III", "IV"]
            if len(phase_dr[phase]) >= 3
        }
        with open(corr_file, "w", encoding="utf-8") as f:
            json.dump(corr_data, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Done. Total steps: {step}")
    print(f"  Grad norms: {grad_file}")
    print(f"  Geodesic:   {geo_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
