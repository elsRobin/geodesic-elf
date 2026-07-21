"""
Multi-Head Bidirectional Self-Attention with:
- RMSNorm (pre-norm)
- Rotary Position Embedding (RoPE) 1D
- QK-Norm (RMSNorm on Q and K)
- Scaled dot-product attention (no causal mask — bidirectional for diffusion)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (more efficient than LayerNorm)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * rms).to(dtype) * self.weight

    def reset_parameters(self):
        nn.init.ones_(self.weight)


class RotaryEmbedding(nn.Module):
    """1D Rotary Position Embedding (RoPE)."""

    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, freqs)  # (max_seq_len, dim/2)
        self.register_buffer("freqs_cos", freqs.cos(), persistent=False)
        self.register_buffer("freqs_sin", freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Apply RoPE to x of shape (B, H, N, D)."""
        cos = self.freqs_cos[:seq_len].unsqueeze(0).unsqueeze(0)  # (1, 1, N, D/2)
        sin = self.freqs_sin[:seq_len].unsqueeze(0).unsqueeze(0)
        x_rot = x[..., : self.dim // 2]
        x_pass = x[..., self.dim // 2 :]
        x1 = x_rot * cos + self._rotate_half(x_rot) * sin
        return torch.cat([x1, x_pass], dim=-1)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)


class BidirectionalAttention(nn.Module):
    """Multi-head bidirectional self-attention with QK-Norm + RoPE."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        dropout: float = 0.0,
        max_seq_len: int = 2048,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        assert hidden_size == num_heads * head_dim, (
            f"hidden_size ({hidden_size}) must equal num_heads ({num_heads}) * head_dim ({head_dim})"
        )

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.qkv_dim = num_heads * head_dim * 3

        # QKV projection (no bias, modern convention)
        self.qkv = nn.Linear(hidden_size, self.qkv_dim, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # QK-Norm: RMSNorm on Q and K per head
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)

        # RoPE
        self.rope = RotaryEmbedding(head_dim // 2, max_seq_len, rope_theta)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.scale = head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, hidden_size) — already norm'ed input
        Returns:
            (B, N, hidden_size)
        """
        B, N, D = x.shape

        # QKV projection
        qkv = self.qkv(x)  # (B, N, 3*num_heads*head_dim)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each (B, N, H, D_h)

        # QK-Norm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Transpose to (B, H, N, D_h) for attention
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # RoPE
        q = self.rope(q, N)
        k = self.rope(k, N)

        # Scaled dot-product attention (bidirectional — no causal mask)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum
        attn_out = torch.matmul(attn_weights, v)  # (B, H, N, D_h)
        attn_out = attn_out.transpose(1, 2).reshape(B, N, D)

        # Output projection
        return self.out_proj(attn_out)
