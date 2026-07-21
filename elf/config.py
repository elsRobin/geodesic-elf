"""
ELF-S Configuration.
Mirrors the semantic structure of the original JAX config system,
simplified for single-GPU PyTorch training.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """ELF-S model architecture hyperparameters (~27M params)."""

    # Tokenizer
    vocab_size: int = 8192
    pad_token_id: int = 0
    mask_token_id: int = 1
    bos_token_id: int = 2
    eos_token_id: int = 3

    # Embedding
    embed_dim: int = 512

    # Transformer backbone
    depth: int = 6
    hidden_size: int = 512
    num_heads: int = 8
    head_dim: int = 64
    mlp_ratio: float = 4.0
    dropout: float = 0.0

    # RoPE
    rope_theta: float = 10000.0

    # Prefix tokens
    num_time_tokens: int = 4
    num_mode_tokens: int = 4

    # Max sequence length
    max_seq_len: int = 256

    @property
    def total_params_estimate(self) -> str:
        emb = self.vocab_size * self.embed_dim
        prefix = (self.num_time_tokens + self.num_mode_tokens) * self.embed_dim
        hidden_dim = int(self.mlp_ratio * self.hidden_size * 2 / 3)
        per_block = (
            3 * self.hidden_size * self.hidden_size
            + self.hidden_size * self.hidden_size
            + 2 * self.hidden_size * hidden_dim
            + hidden_dim * self.hidden_size
            + 2 * self.hidden_size
        )
        blocks = self.depth * per_block
        final = self.hidden_size + self.hidden_size * self.vocab_size
        total = emb + prefix + blocks + final
        total -= self.hidden_size * self.vocab_size
        if total > 1e9:
            return f"{total / 1e9:.1f}B"
        return f"{total / 1e6:.1f}M"


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    batch_size: int = 32
    grad_accum_steps: int = 1
    max_seq_len: int = 256
    max_steps: int = 100_000
    eval_every: int = 1_000
    save_every: int = 5_000
    log_every: int = 50

    # Optimizer
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.98
    eps: float = 1e-8

    # LR schedule
    warmup_steps: int = 2_000
    lr_schedule: str = "cosine"

    # EMA
    ema_decay: float = 0.9999

    # Flow Matching
    denoiser_p_mean: float = -1.5
    denoiser_p_std: float = 0.8
    denoiser_noise_scale: float = 2.0
    t_eps: float = 1e-2

    # Dual-branch training
    decoder_prob: float = 0.5
    decoder_noise_scale: float = 2.0
    decoder_t_min: float = 0.7
    decoder_t_max: float = 1.0
    decode_lr_multiplier: float = 2.0

    # Precision
    dtype: str = "float32"
    use_compile: bool = False

    # Checkpoint
    output_dir: str = "./checkpoints"
    keep_checkpoints: int = 0  # 0 = keep all, N = keep only last N

    # Logging
    use_wandb: bool = False
    wandb_project: str = "elf-s"
    wandb_run_name: Optional[str] = None

    # Data
    dataset_name: Optional[str] = None
    dataset_config: Optional[str] = None
    text_field: str = "text"
    data_dir: Optional[str] = None


@dataclass
class GenerationConfig:
    """Text generation / sampling hyperparameters."""

    method: str = "sde"
    num_steps: int = 32
    noise_scale: float = 2.0
    sde_gamma: float = 1.0
    max_new_tokens: int = 128
    temperature: float = 1.0
    time_schedule: str = "logit_normal"


@dataclass
class ELFConfig:
    """Top-level config aggregating all sub-configs."""

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)

    @classmethod
    def from_dict(cls, d: dict) -> "ELFConfig":
        model = ModelConfig(**d.get("model", {}))
        training = TrainingConfig(**d.get("training", {}))
        generation = GenerationConfig(**d.get("generation", {}))
        return cls(model=model, training=training, generation=generation)
