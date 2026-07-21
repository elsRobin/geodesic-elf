"""
ELF: Embedded Language Flows — Continuous Diffusion Language Model.

Core package providing model definition, training, data loading, and sampling.
"""

from elf.config import ModelConfig, TrainingConfig, GenerationConfig, ELFConfig
from elf.model import ELFModel, EMACopy, update_ema

__version__ = "0.2.0"
__all__ = [
    "ModelConfig",
    "TrainingConfig",
    "GenerationConfig",
    "ELFConfig",
    "ELFModel",
    "EMACopy",
    "update_ema",
]
