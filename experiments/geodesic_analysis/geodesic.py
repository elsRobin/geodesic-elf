"""Direction A+B: Geodesic distance computation in ELF's latent space.

Adapts the RelativeGeodesics framework (NeurIPS 2024) to ELF:
  - Uses ELF's LM Head as the decoder f(z) to compute "geodesic energy"
    along interpolation paths in the continuous latent space.
  - Measures path curvature, which may reveal whether ODE trajectories
    approach true geodesics.

Reference: RelativeGeodesics-main/vision_foundation_models/src/representations/geo_euc.py
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from elf.model import ELFModel


def geodesic_energy(
    model: ELFModel,
    z_start: torch.Tensor,
    z_end: torch.Tensor,
    num_points: int = 50,
    energy_type: str = "l2",
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Compute geodesic energy along linear interpolation between z_start and z_end.

    Supports shapes: (N, D) for full sequence or (D,) for single vector.

    Args:
        model: ELFModel with trained LM Head
        z_start / z_end: (N, D) or (D,)
        num_points: interpolation points (capped at 32 for memory)
        energy_type: "l2" (energy) or "l1" (length)
        temperature: softmax temperature

    Returns:
        scalar energy value
    """
    model.eval()
    device = next(model.parameters()).device

    # Normalize to 2D: (N, D)
    if z_start.dim() == 1:
        z_start = z_start.unsqueeze(0)  # (1, D)
    if z_end.dim() == 1:
        z_end = z_end.unsqueeze(0)

    z_start = z_start.to(device)
    z_end = z_end.to(device)
    N, D = z_start.shape

    num_points = min(num_points, 32)

    # Interpolation: (num_points, N, D)
    alphas = torch.linspace(0, 1, num_points, device=device).view(-1, 1, 1)
    z_path = (1 - alphas) * z_start.unsqueeze(0) + alphas * z_end.unsqueeze(0)

    # Decode through LM Head
    outputs = []
    t_fake = torch.ones(1, device=device)
    with torch.no_grad():
        for point in z_path:
            # point: (N, D) → unsqueeze → (1, N, D) for model
            _, logits = model(point.unsqueeze(0), t_fake, decoder_step=True)
            probs = F.softmax(logits / temperature, dim=-1)
            outputs.append(probs.squeeze(0))  # (N, vocab)

    outputs = torch.stack(outputs, dim=0)  # (num_points, N, vocab)
    diffs = outputs[1:] - outputs[:-1]     # (num_points-1, N, vocab)

    if energy_type == "l2":
        energy = (diffs ** 2).sum()
    else:
        energy = diffs.abs().sum()

    return energy


def path_curvature(
    model: ELFModel,
    z_trajectory: torch.Tensor,  # (T, B, N, D)
    num_segments: int = 10,
) -> Dict:
    """
    Analyze curvature of the ODE path by computing geodesic energy
    between consecutive segments.

    Returns:
        dict with per-segment energies and total curvature estimate
    """
    T = z_trajectory.shape[0]
    indices = torch.linspace(0, T - 1, num_segments + 1, dtype=torch.long)

    segment_energies = []
    for i in range(len(indices) - 1):
        z_start = z_trajectory[indices[i], 0]     # (N, D)
        z_end = z_trajectory[indices[i + 1], 0]
        energy = geodesic_energy(model, z_start, z_end, num_points=8)
        segment_energies.append(energy.item())

    return {
        "num_segments": num_segments,
        "segment_energies": segment_energies,
        "total_energy": sum(segment_energies),
        "energy_variance": float(
            torch.tensor(segment_energies).var().item() if len(segment_energies) > 1 else 0
        ),
    }


def compare_paths(
    model: ELFModel,
    z_straight: torch.Tensor,
    z_ode: torch.Tensor,
    num_points: int = 16,
) -> Dict:
    """
    Compare geodesic energy of straight line vs ODE path.
    Ratio > 1 suggests ODE follows manifold curvature.
    """
    straight_energy = geodesic_energy(
        model, z_straight[0, 0], z_straight[-1, 0], num_points=num_points
    )
    ode_energy = geodesic_energy(
        model, z_ode[0, 0], z_ode[-1, 0], num_points=num_points
    )

    return {
        "straight_line_energy": straight_energy.item(),
        "ode_path_energy": ode_energy.item(),
        "ratio": ode_energy.item() / max(straight_energy.item(), 1e-8),
    }
