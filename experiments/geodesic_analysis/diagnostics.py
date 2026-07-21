"""Direction A+F: Intermediate-state decoding and semantic progression analysis.

Observes the ODE/SDE denoising path by capturing and decoding intermediate
latent states. Quantifies whether the evolution shows Chain-of-Thought structure.
"""

import json
import math
import torch
import torch.nn.functional as F
from typing import Optional, List, Dict
from dataclasses import dataclass, field

from elf.model import ELFModel
from elf.data.tokenizer import BPETokenizer
from elf.diffusion import build_schedule, decode_logits


@dataclass
class TraceConfig:
    """Intermediate-state tracing configuration."""

    method: str = "sde"
    num_steps: int = 64
    noise_scale: float = 2.0
    sde_gamma: float = 0.5
    time_schedule: str = "logit_normal"
    seq_len: int = 128
    batch_size: int = 1
    num_snapshots: int = 12
    snapshot_times: Optional[List[float]] = None
    decode_strategy: str = "argmax"
    temperature: float = 1.0
    top_k: int = 10
    top_p: float = 0.9
    output_json: Optional[str] = None
    verbose: bool = True


@torch.no_grad()
def generate_with_intermediates(
    model: ELFModel,
    tokenizer: BPETokenizer,
    config: TraceConfig,
    device: Optional[torch.device] = None,
) -> Dict:
    """Execute sampling and capture all intermediate-state decodings."""
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    B = config.batch_size
    N = config.seq_len
    E = model.embed_dim

    z = torch.randn(B, N, E, device=device) * config.noise_scale
    t_steps = build_schedule(config.num_steps, device, config.time_schedule)

    if config.snapshot_times is not None:
        snapshot_times = sorted(set(config.snapshot_times))
    else:
        snapshot_times = [round(float(i), 3) for i in
                          torch.linspace(0, 1, config.num_snapshots).tolist()]

    snapshots_wanted = set(snapshot_times)
    snapshots_collected = []

    for step_i in range(config.num_steps):
        t_current = t_steps[step_i]
        dt = t_steps[step_i + 1] - t_current
        t_batch = torch.full((B,), t_current.item(), device=device)

        v_pred, _ = model(z, t_batch, decoder_step=False)
        z = z + v_pred * dt

        if config.method == "sde" and step_i < config.num_steps - 1:
            z = z + config.sde_gamma * math.sqrt(dt.item()) * torch.randn_like(z)

        next_t = t_steps[step_i + 1].item()
        for snap_t in list(snapshots_wanted):
            if t_current.item() <= snap_t < next_t or abs(next_t - snap_t) < 1e-4:
                t_decode = torch.full((B,), min(snap_t, 0.999), device=device)
                _, logits = model(z, t_decode, decoder_step=True)
                token_ids = decode_logits(
                    logits, strategy=config.decode_strategy,
                    temperature=config.temperature,
                    top_k=config.top_k, top_p=config.top_p,
                )
                ids_list = token_ids[0].tolist()
                text = tokenizer.decode(ids_list)
                snapshots_collected.append({
                    "step": step_i, "t": snap_t,
                    "token_ids": ids_list, "text": text,
                })
                snapshots_wanted.discard(snap_t)

    # Ensure final decoding at t=1
    if 1.0 in snapshot_times and not any(abs(s["t"] - 1.0) < 1e-4 for s in snapshots_collected):
        t_one = torch.ones(B, device=device)
        _, logits = model(z, t_one, decoder_step=True)
        token_ids = decode_logits(
            logits, strategy=config.decode_strategy,
            temperature=config.temperature,
            top_k=config.top_k, top_p=config.top_p,
        )
        snapshots_collected.append({
            "step": config.num_steps, "t": 1.0,
            "token_ids": token_ids[0].tolist(),
            "text": tokenizer.decode(token_ids[0].tolist()),
        })

    snapshots_collected.sort(key=lambda s: s["t"])

    return {
        "method": config.method,
        "num_steps": config.num_steps,
        "decode_strategy": config.decode_strategy,
        "trajectory": snapshots_collected,
        "final_t": float(t_steps[-1].item()),
    }


@torch.no_grad()
def generate_multiple_traces(
    model: ELFModel,
    tokenizer: BPETokenizer,
    config: TraceConfig,
    num_traces: int = 3,
    device: Optional[torch.device] = None,
) -> List[Dict]:
    """Generate multiple SDE trajectories for comparison."""
    traces = []
    for _ in range(num_traces):
        traces.append(generate_with_intermediates(model, tokenizer, config, device=device))
    return traces


def analyze_semantic_progression(trace: Dict) -> Dict:
    """Quantify semantic evolution metrics along the path."""
    results = []
    for snap in trace["trajectory"]:
        ids = [x for x in snap["token_ids"] if x not in (0, 2, 3)]
        special_count = sum(1 for c in [0, 1, 2, 3, 4] for tid in snap["token_ids"] if tid == c)

        unique_ratio = len(set(ids)) / max(1, len(ids)) if ids else 0
        rep_count = sum(1 for a, b in zip(snap["token_ids"], snap["token_ids"][1:]) if a == b)
        rep_rate = rep_count / max(1, len(snap["token_ids"]) - 1)

        results.append({
            "t": snap["t"],
            "text_length": len(snap["text"]),
            "num_tokens": len(snap["token_ids"]),
            "num_unique_tokens": len(set(ids)),
            "unique_token_ratio": round(unique_ratio, 4),
            "repetition_rate": round(rep_rate, 4),
            "special_token_ratio": round(special_count / max(1, len(snap["token_ids"])), 4),
        })

    return {"method": trace["method"], "metrics": results}


def print_trace(trace: Dict, max_chars: int = 200):
    """Print intermediate-state evolution in human-readable format."""
    method = trace["method"].upper()
    strategy = trace["decode_strategy"]
    print(f"\n{'=' * 72}")
    print(f"  ELF {method} Sampling · Intermediate Decoding Trace ({strategy})")
    print(f"  Steps: {trace['num_steps']}  |  Snapshots: {len(trace['trajectory'])}")
    print(f"{'=' * 72}")

    for snap in trace["trajectory"]:
        t_label = f"t={snap['t']:.3f}"
        step_label = f"step {snap['step']}"
        text = snap["text"]
        if len(text) > max_chars:
            display_text = text[:max_chars] + " ..."
        else:
            display_text = text
        clean = display_text.replace("<pad>", "").replace("<bos>", "").replace("<eos>", "").replace("<unk>", "?").strip()

        bar_len = 20
        filled = int(snap["t"] * bar_len)
        bar = "\u2588" * filled + "\u2591" * (bar_len - filled)

        print(f"\n  [{bar}] {t_label:>8s}  ({step_label:>8s})")
        print(f"  {clean}")

    print(f"\n{'=' * 72}")


def print_trace_comparison(traces: List[Dict], max_chars: int = 120):
    """Compare multiple trajectories side-by-side."""
    if len(traces) < 2:
        print_trace(traces[0])
        return

    ref_times = [s["t"] for s in traces[0]["trajectory"]]
    print(f"\n{'=' * 80}")
    print(f"  Multi-Trace Comparison — {len(traces)} {traces[0]['method'].upper()} paths")
    print(f"{'=' * 80}")

    for i, t_val in enumerate(ref_times):
        bar_len = 15
        filled = int(t_val * bar_len)
        bar = "\u2588" * filled + "\u2591" * (bar_len - filled)
        print(f"\n  [{bar}] t={t_val:.3f}")
        print(f"  {'─' * 72}")
        for j, trace in enumerate(traces):
            snap = trace["trajectory"][i]
            clean = snap["text"].replace("<pad>", "").replace("<bos>", "").replace("<eos>", "").replace("<unk>", "?").strip()
            if len(clean) > max_chars:
                clean = clean[:max_chars] + "..."
            print(f"  [{j}] {clean}")
    print(f"\n{'=' * 80}")


def print_analysis(analysis: Dict):
    """Print semantic evolution metrics table."""
    print(f"\n{'=' * 72}")
    print(f"  Semantic Evolution Analysis ({analysis['method'].upper()})")
    print(f"  {'t':>8s}  {'len':>5s}  {'unique%':>8s}  {'repet%':>8s}  {'special%':>10s}")
    print(f"  {'─' * 64}")
    for m in analysis["metrics"]:
        print(f"  {m['t']:8.3f}  {m['text_length']:5d}  "
              f"{m['unique_token_ratio']:8.4f}  {m['repetition_rate']:8.4f}  "
              f"{m['special_token_ratio']:10.4f}")
    print(f"{'=' * 72}")


def trace_to_json(trace: Dict) -> str:
    """Export trace results to JSON string."""
    export = {
        "method": trace["method"],
        "num_steps": trace["num_steps"],
        "decode_strategy": trace["decode_strategy"],
        "trajectory": trace["trajectory"],
    }
    return json.dumps(export, ensure_ascii=False, indent=2)
