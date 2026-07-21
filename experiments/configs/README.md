# Experiment Configurations

Training configurations are YAML (or JSON) files that define model architecture and training hyperparameters. CLI arguments override config values.

## Example: Dual-Branch Training

```yaml
# experiments/configs/gsm8k_dual.yaml
model:
  vocab_size: 8192
  embed_dim: 512
  depth: 6
  hidden_size: 512
  num_heads: 8
training:
  max_steps: 100000
  batch_size: 32
  decoder_prob: 0.5          # 50% denoising, 50% decoding
  decoder_t_min: 0.7
  decoder_t_max: 1.0
  denoiser_noise_scale: 2.0
  decoder_noise_scale: 2.0
  learning_rate: 3e-4
  decode_lr_multiplier: 2.0
  warmup_steps: 2000
  use_wandb: true
  wandb_project: "elf-s"
  wandb_run_name: "my-experiment"
  output_dir: "./checkpoints/my-experiment"
```

## Key Parameters

| Parameter | Description |
|-----------|-------------|
| `decoder_prob` | Probability of selecting the decoding branch (vs denoising). 0.5 = balanced |
| `decoder_prob = 1.0` | Pure CE pretrain mode — only training the embedding to be decodable |
| `decode_lr_multiplier` | LR multiplier for embedding + final norm parameters |
