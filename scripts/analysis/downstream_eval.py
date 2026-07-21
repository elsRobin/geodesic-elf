#!/usr/bin/env python3
"""
Downstream Evaluation: Denoising and Decoding Loss Comparison.

Computes denoising MSE and decoding CE loss on a test set for
multiple checkpoints. Compares whether models with divergent R(k)
achieve comparable loss — verifying that the geometric diagnostic
captures structure orthogonal to standard loss metrics.

Usage:
    python scripts/analysis/downstream_eval.py \\
        --checkpoints checkpoints/model_a/checkpoint-100000 \\
                      checkpoints/model_b/checkpoint-100000 \\
                      checkpoints/model_c/checkpoint-100000 \\
        --test_data data/test.txt \\
        --tokenizer tokenizer.json \\
        --output results/downstream_eval.json
"""

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from elf.config import ELFConfig
from elf.model import ELFModel
from elf.data.tokenizer import BPETokenizer


# ── Checkpoint loading ────────────────────────────────────────────
def load_checkpoint(ckpt_dir: str, device: torch.device,
                    use_ema: bool = False):
    config_path = os.path.join(ckpt_dir, "config.json")
    with open(config_path) as f:
        config = ELFConfig.from_dict(json.load(f))
    model = ELFModel(config.model).to(device)
    wpath = os.path.join(ckpt_dir, "model.pt")
    model.load_state_dict(torch.load(wpath, map_location=device,
                                     weights_only=False))
    model.eval()
    return model, config


# ── Loss computation ──────────────────────────────────────────────
@torch.no_grad()
def compute_losses(model: ELFModel, input_ids: torch.Tensor,
                   noise_scale: float = 2.0, t_eps: float = 1e-2,
                   n_denoise: int = 5, n_decode: int = 5,
                   ) -> dict:
    """Compute denoising MSE and decoding CE on a single sequence.

    Averages over n_denoise and n_decode random time samples.
    """
    device = next(model.parameters()).device
    B, N = input_ids.shape
    x0 = model.embedding(input_ids)

    # ── Denoising loss (flow matching MSE) ──
    # t ~ sigmoid(N(-1.5, 0.8²)) — heavily skewed to small t (high noise)
    denoise_losses = []
    for _ in range(n_denoise):
        logit_t = -1.5 + 0.8 * torch.randn(B, device=device)
        t = torch.sigmoid(logit_t).clamp(min=t_eps, max=0.999)
        t_exp = t.view(B, 1, 1)

        eps = torch.randn_like(x0)
        z = t_exp * x0 + (1 - t_exp) * noise_scale * eps
        v_target = x0 - noise_scale * eps
        v_pred, _ = model(z, t, decoder_step=False)
        denoise_losses.append(F.mse_loss(v_pred, v_target).item())

    # ── Decoding loss (cross-entropy on LM Head) ──
    # t ~ uniform(0.7, 1.0), noise_scale = 2.0 (same as training config)
    decode_losses = []
    decode_correct = 0
    decode_total = 0
    for _ in range(n_decode):
        t = torch.rand(B, device=device) * 0.3 + 0.7  # U[0.7, 1.0]
        t_exp = t.view(B, 1, 1)

        eps = torch.randn_like(x0)
        z = t_exp * x0 + (1 - t_exp) * noise_scale * eps
        _, logits = model(z, t, decoder_step=True)
        # logits and input_ids have matching seq lengths
        ce = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            input_ids.view(-1),
            ignore_index=-100,
        )
        decode_losses.append(ce.item())
        # Token-level accuracy
        preds = logits.argmax(dim=-1)  # (1, N)
        decode_correct += (preds == input_ids).sum().item()
        decode_total += input_ids.numel()

    return {
        "denoise_mse": sum(denoise_losses) / len(denoise_losses),
        "decode_ce": sum(decode_losses) / len(decode_losses),
        "decode_acc": decode_correct / max(decode_total, 1),
    }


# ── Main ──────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="GSM8K Test-Set Loss: Regime M vs S")
    parser.add_argument("--checkpoints", nargs="+", required=True,
                        help="Checkpoint dirs (Run1, Run2, Run3)")
    parser.add_argument("--test_data", required=True,
                        help="GSM8K test.txt")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", default="loss_comparison.json")
    parser.add_argument("--max_problems", type=int, default=0,
                        help="Max test problems (0=all)")
    parser.add_argument("--n_denoise", type=int, default=10,
                        help="Denoising loss samples per problem")
    parser.add_argument("--n_decode", type=int, default=10,
                        help="Decoding loss samples per problem")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load test data
    with open(args.test_data, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip() and len(l.strip()) > 40]
    if args.max_problems > 0:
        lines = lines[:args.max_problems]
    print(f"Test problems: {len(lines)}")

    # Load tokenizer once
    tokenizer = BPETokenizer.load(args.tokenizer, vocab_size=8192)

    # Pre-tokenize all texts
    print("Tokenizing...")
    all_ids = []
    for line in lines:
        ids = tokenizer.encode(line)[:256]
        all_ids.append(torch.tensor([ids]))

    results = {}
    for ckpt_dir in args.checkpoints:
        run_name = os.path.basename(os.path.dirname(ckpt_dir)) or "unknown"
        print(f"\n{'='*60}")
        print(f"Evaluating: {run_name}")
        print(f"  Checkpoint: {ckpt_dir}")
        print(f"{'='*60}")

        model, _ = load_checkpoint(ckpt_dir, device, use_ema=False)
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  Params: {n_params:.1f}M")

        denoise_total = 0.0
        decode_total = 0.0
        acc_total = 0.0
        acc_n = 0
        n_done = 0

        for i, ids_tensor in enumerate(all_ids):
            ids_tensor = ids_tensor.to(device)
            losses = compute_losses(
                model, ids_tensor,
                n_denoise=args.n_denoise,
                n_decode=args.n_decode)

            denoise_total += losses["denoise_mse"]
            decode_total += losses["decode_ce"]
            acc_total += losses["decode_acc"]
            acc_n += 1
            n_done += 1

            if (i + 1) % 100 == 0:
                d_mean = denoise_total / n_done
                c_mean = decode_total / n_done
                a_mean = 100 * acc_total / acc_n
                print(f"  [{i+1}/{len(all_ids)}] denoise MSE={d_mean:.4f}  "
                      f"decode CE={c_mean:.2f}  acc={a_mean:.1f}%")

        denoise_mean = denoise_total / n_done
        decode_mean = decode_total / n_done
        acc_mean = acc_total / acc_n
        print(f"\n  FINAL: denoise MSE = {denoise_mean:.4f}  "
              f"decode CE = {decode_mean:.2f}  acc = {100*acc_mean:.1f}%")

        results[run_name] = {
            "checkpoint": ckpt_dir,
            "denoise_mse": round(denoise_mean, 6),
            "decode_ce": round(decode_mean, 2),
            "decode_acc": round(100 * acc_mean, 1),
            "n_problems": n_done,
        }

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY: Test-Set Loss Comparison")
    print(f"{'='*60}")
    print(f"  {'Run':20s}  {'Regime':6s}  {'Denoise MSE':>12s}  "
          f"{'Decode CE':>10s}")
    print("  " + "-" * 56)
    for name, r in results.items():
        regime = "S" if "run2" in name.lower() else "M"
        print(f"  {name:20s}  {regime:6s}  {r['denoise_mse']:>12.6f}  "
              f"{r['decode_ce']:>10.4f}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
