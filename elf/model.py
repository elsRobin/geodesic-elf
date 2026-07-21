"""
ELF-S: Embedded Language Flow — Small PyTorch implementation (~27M params).

Architecture:
    Embedding → [Time(4)+Mode(4)] → Transformer×6 → RMSNorm → LM Head

Dual-branch training (Flow Matching + Token Decoding):
  - 80% Denoising (L2 velocity loss): predict velocity field v = (x0 - z) / (1-t)
  - 20% Decoding  (CE token loss): predict discrete tokens from noisy latents
"""

import math
import torch
import torch.nn as nn
from typing import Optional, Tuple

from elf.modules import RMSNorm, ELFBlock
from elf.config import ModelConfig


def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal encoding for continuous time t in [0, 1]."""
    assert dim % 2 == 0
    half_dim = dim // 2
    exponent = -math.log(10000) * torch.arange(0, half_dim, dtype=t.dtype, device=t.device) / half_dim
    emb = t.unsqueeze(-1) * exponent.exp().unsqueeze(0)
    return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ELFModel(nn.Module):
    """
    ELF-S continuous diffusion language model.

    Args:
        config: ModelConfig dataclass
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_dim = config.embed_dim
        self.hidden_size = config.hidden_size

        # Token embedding
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)

        # Learnable prefix tokens
        self.time_tokens = nn.Parameter(torch.randn(config.num_time_tokens, config.embed_dim) * 0.02)
        self.mode_tokens = nn.Parameter(torch.randn(config.num_mode_tokens, config.embed_dim) * 0.02)

        # Time MLP
        time_emb_dim = config.embed_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, config.num_time_tokens * config.embed_dim),
        )

        # Transformer blocks
        total_max_len = config.max_seq_len + config.num_time_tokens + config.num_mode_tokens
        self.blocks = nn.ModuleList([
            ELFBlock(
                hidden_size=config.hidden_size,
                num_heads=config.num_heads,
                head_dim=config.head_dim,
                mlp_ratio=config.mlp_ratio,
                dropout=config.dropout,
                max_seq_len=total_max_len,
                rope_theta=config.rope_theta,
            )
            for _ in range(config.depth)
        ])

        # Final norm + LM head
        self.final_norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight  # weight tying

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def _get_time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        B = t.shape[0]
        t_sinusoidal = sinusoidal_time_embedding(t, self.embed_dim)
        return self.time_mlp(t_sinusoidal).view(B, self.config.num_time_tokens, self.embed_dim)

    def _get_mode_tokens(self, batch_size: int, decoder_step: bool) -> torch.Tensor:
        mode = self.mode_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        if not decoder_step:
            mode = torch.zeros_like(mode)
        return mode

    def forward(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        decoder_step: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            z: (B, N, embed_dim) noisy continuous latents
            t: (B,) float time step in [0, 1]
            decoder_step: if True, also compute token logits

        Returns:
            v_pred: (B, N, embed_dim) predicted velocity field
            logits: (B, N, vocab_size) or None
        """
        B, N, _ = z.shape

        time_emb = self._get_time_embedding(t)
        mode_emb = self._get_mode_tokens(B, decoder_step)

        h = torch.cat([time_emb, mode_emb, z], dim=1)

        for block in self.blocks:
            h = block(h)

        h = self.final_norm(h)
        prefix_len = self.config.num_time_tokens + self.config.num_mode_tokens
        h_seq = h[:, prefix_len:, :]
        v_pred = h_seq

        logits = None
        if decoder_step:
            logits = self.lm_head(h_seq)

        return v_pred, logits

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def sample(
        self,
        batch_size: int = 1,
        seq_len: int = 128,
        num_steps: int = 32,
        noise_scale: float = 2.0,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Generate text via iterative denoising (Euler ODE sampler)."""
        self.eval()
        if device is None:
            device = next(self.parameters()).device

        z = torch.randn(batch_size, seq_len, self.embed_dim, device=device) * noise_scale
        dt = 1.0 / num_steps

        for step in range(num_steps):
            t = torch.full((batch_size,), step * dt, device=device)
            v_pred, _ = self(z, t, decoder_step=False)
            z = z + v_pred * dt

        t_one = torch.ones(batch_size, device=device)
        _, logits = self(z, t_one, decoder_step=True)
        return logits.argmax(dim=-1)

    @torch.no_grad()
    def sample_sde(
        self,
        batch_size: int = 1,
        seq_len: int = 128,
        num_steps: int = 32,
        noise_scale: float = 2.0,
        sde_gamma: float = 1.0,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Generate text via SDE sampling (stochastic differential equation)."""
        self.eval()
        if device is None:
            device = next(self.parameters()).device

        z = torch.randn(batch_size, seq_len, self.embed_dim, device=device) * noise_scale
        dt = 1.0 / num_steps

        for step in range(num_steps):
            t = torch.full((batch_size,), step * dt, device=device)
            v_pred, _ = self(z, t, decoder_step=False)
            z = z + v_pred * dt
            if step < num_steps - 1:
                z = z + sde_gamma * math.sqrt(dt) * torch.randn_like(z)

        t_one = torch.ones(batch_size, device=device)
        _, logits = self(z, t_one, decoder_step=True)
        return logits.argmax(dim=-1)


def EMACopy(model: nn.Module, decay: float = 0.9999):
    ema_model = type(model)(model.config)
    ema_model.load_state_dict(model.state_dict())
    return ema_model


@torch.no_grad()
def update_ema(ema_model: ELFModel, model: ELFModel, decay: float = 0.9999):
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.data.lerp_(p.data, 1 - decay)
