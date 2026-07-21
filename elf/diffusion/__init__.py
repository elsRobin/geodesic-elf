"""Diffusion utilities: time sampling, ODE/SDE samplers, noise schedules."""

import math
import torch
from typing import Optional


# ── Time Sampling ──────────────────────────────────────

def sample_log_normal_time(
    batch_size: int,
    p_mean: float = -1.5,
    p_std: float = 0.8,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Sample time t ~ logit-normal(p_mean, p_std)."""
    logit_t = p_mean + p_std * torch.randn(batch_size, device=device)
    return torch.sigmoid(logit_t)


def sample_uniform_time(
    batch_size: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Uniform time sampling in [0, 1]."""
    return torch.rand(batch_size, device=device)


def sample_decode_time(
    batch_size: int,
    t_min: float = 0.7,
    t_max: float = 1.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Sample time t for decoder branch, uniform in [t_min, t_max]."""
    return t_min + (t_max - t_min) * torch.rand(batch_size, device=device)


# ── Time Schedules ─────────────────────────────────────

def build_schedule(
    num_steps: int,
    device: torch.device,
    schedule_type: str = "logit_normal",
) -> torch.Tensor:
    """Build time schedule for ODE/SDE integration."""
    if schedule_type == "logit_normal":
        t = torch.linspace(0, 1, num_steps + 1, device=device)
        eps = 1e-6
        t = torch.clamp(t, eps, 1 - eps)
        logit_t = torch.logit(t)
        logit_t = logit_t * 1.2
        return torch.sigmoid(logit_t)
    elif schedule_type == "cosine":
        t = 0.5 * (1 + torch.cos(torch.linspace(0, 1, num_steps + 1, device=device) * math.pi))
        return 1 - t
    else:  # linear
        return torch.linspace(0, 1, num_steps + 1, device=device)


# ── Forward Process ────────────────────────────────────

def forward_diffusion(
    x0: torch.Tensor,
    t: torch.Tensor,
    noise_scale: float = 2.0,
) -> torch.Tensor:
    """z = t * x0 + (1-t) * noise_scale * eps"""
    eps = torch.randn_like(x0)
    t_expanded = t.view(-1, 1, 1)
    return t_expanded * x0 + (1 - t_expanded) * noise_scale * eps


def velocity_target(
    x0: torch.Tensor,
    eps: torch.Tensor,
    noise_scale: float = 2.0,
) -> torch.Tensor:
    """v_target = x0 - noise_scale * eps (for linear interpolation path)"""
    return x0 - noise_scale * eps


# ── Decode Strategies ──────────────────────────────────

def decode_logits(
    logits: torch.Tensor,
    strategy: str = "argmax",
    temperature: float = 1.0,
    top_k: int = 10,
    top_p: float = 0.9,
) -> torch.Tensor:
    """Decode token IDs from logits using various strategies."""
    import torch.nn.functional as F

    if strategy == "argmax":
        return logits.argmax(dim=-1)

    if temperature != 1.0:
        logits = logits / temperature

    probs = F.softmax(logits, dim=-1)

    if strategy == "temperature":
        return torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(logits.shape[0], logits.shape[1])

    if strategy == "topk":
        topk_vals, topk_indices = torch.topk(probs, k=min(top_k, probs.size(-1)), dim=-1)
        topk_probs = F.normalize(topk_vals, p=1, dim=-1)
        sampled_idx = torch.multinomial(topk_probs.view(-1, topk_probs.size(-1)), 1).view(probs.shape[0], probs.shape[1])
        return topk_indices.gather(-1, sampled_idx.unsqueeze(-1)).squeeze(-1)

    if strategy == "topp":
        sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        mask = cumsum <= top_p
        mask[..., 0] = True
        sorted_probs[~mask] = 0.0
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
        sampled_idx = torch.multinomial(sorted_probs.view(-1, sorted_probs.size(-1)), 1).view(probs.shape[0], probs.shape[1])
        return sorted_indices.gather(-1, sampled_idx.unsqueeze(-1)).squeeze(-1)

    raise ValueError(f"Unknown decode strategy: {strategy}")
