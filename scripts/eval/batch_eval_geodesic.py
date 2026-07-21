#!/usr/bin/env python3
"""
Geodesic Convergence Evaluation Script for ELF.

Computes ODE/Straight ratio across all saved checkpoints to quantify
how the denoising path aligns with latent-space geodesics over training.

Usage:
    python scripts/eval_geodesic.py \
        --checkpoint_dir /root/autodl-tmp/elf-checkpoints \
        --data /root/autodl-tmp/ELF_mini/data/gsm8k/train.txt \
        --tokenizer /root/autodl-tmp/elf-checkpoints/tokenizer.json \
        --output /root/autodl-tmp/elf-checkpoints/geodesic_curve.json \
        --num_problems 5 --num_runs 3
"""

import os
import sys
import json
import argparse
from typing import List, Dict
import torch

from elf.config import ELFConfig
from elf.model import ELFModel
from elf.data.tokenizer import BPETokenizer
from experiments.geodesic_analysis.geodesic import geodesic_energy


def load_checkpoint(ckpt_dir: str, tokenizer_path: str, device: torch.device):
    """Load model and tokenizer from checkpoint directory."""
    config_path = os.path.join(ckpt_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"No config.json in {ckpt_dir}")

    with open(config_path, "r") as f:
        config = ELFConfig.from_dict(json.load(f))

    cfg = config.model

    model = ELFModel(cfg).to(device)
    ema_path = os.path.join(ckpt_dir, "ema_model.pt")
    if os.path.exists(ema_path):
        model.load_state_dict(torch.load(ema_path, map_location=device))
    else:
        model.load_state_dict(torch.load(os.path.join(ckpt_dir, "model.pt"), map_location=device))
    model.eval()

    tokenizer = BPETokenizer.load(tokenizer_path, vocab_size=cfg.vocab_size)

    return model, tokenizer


def compute_ratio(
    model: ELFModel,
    problem_ids: torch.Tensor,
    t_start: float = 0.2,
    noise_scale: float = 2.0,
    num_steps: int = 32,
    num_segments: int = 8,
) -> float:
    """
    Compute ODE/Straight geodesic energy ratio for one problem.

    Returns ratio: ODE_path_energy / straight_line_energy
    """
    device = next(model.parameters()).device
    ids_tensor = problem_ids.unsqueeze(0).to(device)  # (1, N)
    x0 = model.embedding(ids_tensor)                    # (1, N, 512)

    # Add noise
    eps = torch.randn_like(x0)
    z = t_start * x0 + (1 - t_start) * noise_scale * eps

    # ODE integration
    dt = (1.0 - t_start) / num_steps
    trajectory = [z.clone()]
    for s in range(num_steps):
        t_val = t_start + s * dt
        t_batch = torch.tensor([t_val], device=device)
        v_pred, _ = model(z, t_batch, decoder_step=False)
        z = z + v_pred * dt
        trajectory.append(z.clone())

    trajectory = torch.stack(trajectory)  # (T, 1, N, 512)

    # Straight-line energy: start → end
    straight_e = geodesic_energy(model, trajectory[0, 0], trajectory[-1, 0], num_points=16)

    # ODE path energy: sum of segment geodesic energies
    step_size = num_steps // num_segments
    total_ode = 0.0
    for i in range(0, num_steps, step_size):
        e = geodesic_energy(
            model, trajectory[i, 0],
            trajectory[min(i + step_size, num_steps), 0],
            num_points=8
        )
        total_ode += e.item()

    return total_ode / max(straight_e.item(), 1e-8)


def load_problems(data_path: str, tokenizer: BPETokenizer, num_problems: int, max_len: int = 200) -> List[str]:
    """Load and filter real problems from data file."""
    with open(data_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if 60 < len(l.strip()) < max_len]

    return lines[:num_problems]


def load_texts_from_dataset(dataset_name: str, config: str, num_problems: int, max_len: int = 200) -> List[str]:
    """Load random text segments from a HuggingFace dataset."""
    try:
        from datasets import load_dataset
        ds = load_dataset(dataset_name, config, split="train", streaming=True)
    except ImportError:
        raise ImportError("Please install datasets: pip install datasets")

    texts = []
    for item in ds:
        text = item.get("text", "") or item.get("content", "") or ""
        text = text.strip().replace("\n", " ")
        if 60 < len(text) < max_len * 3:  # allow longer, we'll truncate
            texts.append(text)
        if len(texts) >= num_problems * 10:  # buffer for filter
            break

    # Filter to consistent length range and return first N
    filtered = [t for t in texts if 60 < len(t) < max_len]
    return filtered[:num_problems]


def evaluate_all_checkpoints(
    checkpoint_dir: str,
    tokenizer_path: str,
    data_path: str = None,
    dataset_name: str = None,
    dataset_config: str = None,
    num_problems: int = 3,
    num_runs: int = 3,
) -> List[Dict]:
    """
    Evaluate ODE/Straight ratio across ALL checkpoints in a directory.

    Provide either data_path (local .txt file) or dataset_name (HF dataset).

    Returns list of {"step": int, "ratio": float, "std": float, "individual": [float]}
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Find all checkpoints
    import glob, re
    all_dirs = sorted(glob.glob(os.path.join(checkpoint_dir, "checkpoint-*")))
    ckpt_dirs = [
        d for d in all_dirs
        if os.path.isdir(d)
        and os.path.exists(os.path.join(d, "config.json"))
        and re.search(r'checkpoint-(\d+)', d)  # skip checkpoint-final, etc.
    ]

    if not ckpt_dirs:
        print(f"ERROR: No checkpoints found in {checkpoint_dir}")
        return []

    # Sort by step number
    ckpt_dirs.sort(key=lambda d: int(re.search(r'checkpoint-(\d+)', d).group(1)))
    print(f"Found {len(ckpt_dirs)} checkpoints")

    # Load shared tokenizer from first checkpoint
    _, tokenizer = load_checkpoint(ckpt_dirs[0], tokenizer_path, device)

    # Load evaluation texts (same across all checkpoints)
    if dataset_name:
        print(f"Loading texts from HF dataset: {dataset_name}")
        problems = load_texts_from_dataset(dataset_name, dataset_config or "", num_problems)
    elif data_path:
        problems = load_problems(data_path, tokenizer, num_problems)
    else:
        print("ERROR: Must specify --data or --dataset")
        return []

    if len(problems) < num_problems:
        print(f"WARNING: Only {len(problems)} texts available (requested {num_problems})")
    print(f"Evaluating {len(problems)} texts × {num_runs} noise seeds")

    results = []

    for ckpt_dir in ckpt_dirs:
        step = int(re.search(r'checkpoint-(\d+)', ckpt_dir).group(1))

        try:
            model, _ = load_checkpoint(ckpt_dir, tokenizer_path, device)
        except Exception as e:
            print(f"  checkpoint-{step:>5d}: SKIP ({e})")
            continue

        # Tokenize each problem once
        all_ids = []
        for p in problems:
            ids = tokenizer.encode(p)[:80]
            all_ids.append(torch.tensor(ids))

        # Run multiple noise seeds
        run_ratios = []
        for run in range(num_runs):
            ratios = []
            for ids_tensor in all_ids:
                ratio = compute_ratio(model, ids_tensor)
                ratios.append(ratio)
            run_ratios.append(ratios)

        # Aggregate: mean across all problems × runs
        all_vals = [r for run in run_ratios for r in run]
        avg_ratio = sum(all_vals) / len(all_vals)
        std_ratio = torch.tensor(all_vals).std().item() if len(all_vals) > 1 else 0.0

        # Per-problem means (across runs)
        per_problem = []
        for i in range(num_problems):
            vals = [run_ratios[r][i] for r in range(num_runs)]
            per_problem.append(round(sum(vals) / len(vals), 4))

        results.append({
            "step": step,
            "ratio": round(avg_ratio, 4),
            "std": round(std_ratio, 4),
            "per_problem": per_problem,
            "n_problems": num_problems,
            "n_runs": num_runs,
        })

        status = "✓" if 0.5 < avg_ratio < 1.5 else "?" if avg_ratio < 0.5 else "↑"
        print(f"  checkpoint-{step:>5d}: ODE/Straight = {avg_ratio:.4f} ± {std_ratio:.4f}  "
              f"{status}  ({num_problems}p × {num_runs}r)")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate geodesic convergence across ELF checkpoints"
    )
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Directory containing checkpoint-* subdirs")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to training text file (e.g. data/gsm8k/train.txt)")
    parser.add_argument("--dataset", type=str, default=None,
                        help="HuggingFace dataset name (e.g. wikitext-2-raw-v1)")
    parser.add_argument("--dataset_config", type=str, default=None,
                        help="HF dataset config/subset (e.g. wikitext-2-v1)")
    parser.add_argument("--tokenizer", type=str, required=True,
                        help="Path to tokenizer.json")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to JSON file")
    parser.add_argument("--num_problems", type=int, default=3,
                        help="Number of problems to evaluate")
    parser.add_argument("--num_runs", type=int, default=3,
                        help="Noise seeds per problem")
    parser.add_argument("--step_start", type=int, default=None,
                        help="Skip checkpoints before this step")
    parser.add_argument("--step_end", type=int, default=None,
                        help="Skip checkpoints after this step")

    args = parser.parse_args()

    print("=" * 65)
    print("  ELF Geodesic Convergence Evaluation")
    print("=" * 65)

    if not args.data and not args.dataset:
        print("ERROR: Must specify --data or --dataset")
        sys.exit(1)

    results = evaluate_all_checkpoints(
        args.checkpoint_dir,
        args.tokenizer,
        data_path=args.data,
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        num_problems=args.num_problems,
        num_runs=args.num_runs,
    )

    # Filter by step range
    if args.step_start is not None or args.step_end is not None:
        results = [
            r for r in results
            if (args.step_start is None or r["step"] >= args.step_start)
            and (args.step_end is None or r["step"] <= args.step_end)
        ]

    if not results:
        print("No results to save.")
        return

    # Summary
    print("\n" + "=" * 65)
    print("  Summary")
    print("=" * 65)

    ratios = [r["ratio"] for r in results]
    steps = [r["step"] for r in results]

    print(f"  Checkpoints evaluated: {len(results)}")
    print(f"  Step range:             {min(steps)} → {max(steps)}")
    print(f"  Ratio range:            {min(ratios):.4f} → {max(ratios):.4f}")

    # Phase detection
    early = [r for r in results if r["ratio"] < 0.8]
    mid = [r for r in results if 0.8 <= r["ratio"] <= 1.2]
    late = [r for r in results if r["ratio"] > 1.2]

    if early and mid:
        transition_step = mid[0]["step"]
        print(f"  Phase transition:       ~step {transition_step} (ratio crosses 0.8)")

    if len(results) >= 2:
        last_half = ratios[len(ratios)//2:]
        print(f"  Late-phase mean:        {sum(last_half)/len(last_half):.4f} ± "
              f"{torch.tensor(last_half).std().item():.4f} (steps "
              f"{steps[len(steps)//2]}–{steps[-1]})")

    # Save
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Saved: {args.output}")


if __name__ == "__main__":
    main()
