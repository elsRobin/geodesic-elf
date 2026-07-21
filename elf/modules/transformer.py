"""
ELF Transformer Block: RMSNorm → Attention → Residual → RMSNorm → SwiGLU → Residual.

Follows the same architectural pattern as the original JAX ELF:
- Pre-norm (RMSNorm before each sub-layer)
- Bidirectional attention (no causal mask — diffusion model)
- SwiGLU activation in FFN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import RMSNorm, BidirectionalAttention


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network with gated activation."""

    def __init__(self, hidden_size: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        # In ELF/compute-optimal: hidden_dim = int(mlp_ratio * hidden_size * 2/3)
        hidden_dim = int(mlp_ratio * hidden_size * 2 / 3)

        self.gate_proj = nn.Linear(hidden_size, hidden_dim, bias=False)
        self.up_proj = nn.Linear(hidden_size, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, hidden_size) — already norm'ed
        Returns:
            (B, N, hidden_size)
        """
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        # SwiGLU: silu(gate) * up
        return self.dropout(self.down_proj(F.silu(gate) * up))


class ELFBlock(nn.Module):
    """
    Single ELF Transformer block:
        x = x + Attention(RMSNorm(x))
        x = x + SwiGLU(RMSNorm(x))
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        max_seq_len: int = 2048,
        rope_theta: float = 10000.0,
    ):
        super().__init__()

        self.attn_norm = RMSNorm(hidden_size)
        self.attn = BidirectionalAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
        )
        self.attn_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.ffn_norm = RMSNorm(hidden_size)
        self.ffn = SwiGLUFFN(
            hidden_size=hidden_size,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention sub-layer
        x = x + self.attn_dropout(self.attn(self.attn_norm(x)))
        # FFN sub-layer
        x = x + self.ffn(self.ffn_norm(x))
        return x
