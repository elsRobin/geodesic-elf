# ============================================================
# ELF-S AutoDL Docker Image
# Based on NVIDIA PyTorch official, adapted for AutoDL
#
# Build: docker build -t elf-s:latest -f docker/Dockerfile .
# ============================================================

FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

LABEL maintainer="elf-s"
LABEL description="ELF-S Continuous Diffusion LM — CoT-Geodesic Research"
LABEL platform="AutoDL"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="6.0;7.0;7.5;8.0;8.6;8.9;9.0"

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget vim htop tmux unzip tree \
    && rm -rf /var/lib/apt/lists/*

# Python deps
RUN pip install --no-cache-dir \
    transformers==4.40.0 \
    datasets==2.19.0 \
    tokenizers==0.19.1 \
    accelerate==0.30.0 \
    wandb==0.16.6 \
    tensorboard==2.16.2 \
    matplotlib==3.8.4 \
    rich==13.7.1 \
    pyyaml

# Copy ELF project
WORKDIR /workspace/ELF_mini

COPY pyproject.toml .
COPY elf/ ./elf/
COPY experiments/ ./experiments/
COPY scripts/ ./scripts/

RUN pip install -e . --no-deps
RUN mkdir -p data checkpoints

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 6006
ENTRYPOINT ["/entrypoint.sh"]
