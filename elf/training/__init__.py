"""Training utilities: loss functions, LR schedules, checkpointing."""

import os
import math
import json
import time
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any
from contextlib import nullcontext

from elf.config import ModelConfig, TrainingConfig, ELFConfig
from elf.model import ELFModel, update_ema
from elf.diffusion import sample_log_normal_time, sample_decode_time

# Lazy import — only loaded when eval_geodesic is active
def _load_geodesic():
    from experiments.geodesic_analysis.geodesic import geodesic_energy
    return geodesic_energy


# ── Loss Functions ─────────────────────────────────────

def denoising_loss(
    model: ELFModel,
    input_ids: torch.Tensor,
    t: torch.Tensor,
    noise_scale: float = 2.0,
    t_eps: float = 1e-2,
) -> torch.Tensor:
    """Flow Matching denoising loss (L2 on velocity field)."""
    B, N = input_ids.shape
    x0 = model.embedding(input_ids)
    eps = torch.randn_like(x0)
    t_expanded = t.view(B, 1, 1)
    z = t_expanded * x0 + (1 - t_expanded) * noise_scale * eps

    v_target = x0 - noise_scale * eps
    v_pred, _ = model(z, t, decoder_step=False)

    return F.mse_loss(v_pred, v_target)


def decoding_loss(
    model: ELFModel,
    input_ids: torch.Tensor,
    t: torch.Tensor,
    noise_scale: float = 2.0,
) -> torch.Tensor:
    """Decoding (token prediction) loss with cross-entropy."""
    B, N = input_ids.shape
    x0 = model.embedding(input_ids)
    eps = torch.randn_like(x0)
    t_expanded = t.view(B, 1, 1)
    z = t_expanded * x0 + (1 - t_expanded) * noise_scale * eps

    _, logits = model(z, t, decoder_step=True)

    return F.cross_entropy(
        logits.view(-1, model.vocab_size),
        input_ids.view(-1),
        ignore_index=model.config.pad_token_id,
    )


# ── LR Scheduler ───────────────────────────────────────

def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    schedule_type: str = "cosine",
) -> torch.optim.lr_scheduler.LambdaLR:

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        if schedule_type == "cosine":
            return 0.5 * (1 + math.cos(math.pi * progress))
        elif schedule_type == "linear":
            return 1.0 - progress
        else:
            return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── Checkpointing ──────────────────────────────────────

def save_checkpoint(
    model: ELFModel,
    ema_model: ELFModel,
    optimizer: torch.optim.Optimizer,
    scheduler,
    step: int,
    config: ELFConfig,
    path: str,
):
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(path, "model.pt"))
    torch.save(ema_model.state_dict(), os.path.join(path, "ema_model.pt"))
    torch.save(optimizer.state_dict(), os.path.join(path, "optimizer.pt"))
    config_dict = {
        "model": config.model.__dict__,
        "training": config.training.__dict__,
        "generation": config.generation.__dict__,
    }
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(path: str, device: torch.device) -> Dict[str, Any]:
    config_path = os.path.join(path, "config.json")
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    config = ELFConfig.from_dict(config_dict)
    model = ELFModel(config.model).to(device)
    model.load_state_dict(torch.load(os.path.join(path, "model.pt"), map_location=device))
    ema_model = ELFModel(config.model).to(device)
    ema_model.load_state_dict(torch.load(os.path.join(path, "ema_model.pt"), map_location=device))
    return {"model": model, "ema_model": ema_model, "config": config}


# ── Evaluation ─────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: ELFModel,
    dataloader: DataLoader,
    config: ELFConfig,
    device: torch.device,
    max_batches: int = 50,
) -> float:
    train_cfg = config.training
    model.eval()
    total_loss = 0.0
    count = 0

    for batch in dataloader:
        if count >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        B = input_ids.shape[0]
        t = sample_log_normal_time(B, p_mean=train_cfg.denoiser_p_mean, p_std=train_cfg.denoiser_p_std, device=device)
        loss = denoising_loss(model, input_ids, t, noise_scale=train_cfg.denoiser_noise_scale, t_eps=train_cfg.t_eps)
        total_loss += loss.item()
        count += 1

    model.train()
    return total_loss / max(1, count)


# ── Main Training Loop ─────────────────────────────────

def train(config: ELFConfig, dataloader: DataLoader, resume_path: Optional[str] = None,
          eval_geodesic: Optional[Dict] = None):
    """Main training loop for ELF-S.
    
    If eval_geodesic is provided, runs inline geodesic ratio eval after each
    checkpoint save and appends to {output_dir}/geodesic_inline.json.
    eval_geodesic = {"data_path": str, "tokenizer_path": str, "n_samples": int, "n_runs": int}
    """
    model_cfg = config.model
    train_cfg = config.training

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    print(f"Model parameters: {model_cfg.total_params_estimate}")

    start_step = 0

    if resume_path:
        print(f"Resuming from {resume_path}...")
        ckpt = load_checkpoint(resume_path, device)
        model = ckpt["model"]
        ema_model = ckpt["ema_model"]
        # Detect step from directory name (e.g. "checkpoint-10000" → 10000)
        import re
        m = re.search(r'checkpoint-(\d+)', resume_path)
        start_step = int(m.group(1)) if m else 0
    else:
        model = ELFModel(model_cfg).to(device)
        ema_model = ELFModel(model_cfg).to(device)
        ema_model.load_state_dict(model.state_dict())
        ema_model.eval()

    actual_params = model.get_num_params()
    print(f"Actual parameters: {actual_params:,} ({actual_params/1e6:.1f}M)")

    # torch.compile
    if train_cfg.use_compile and not resume_path and hasattr(torch, "compile"):
        print("Applying torch.compile...")
        model = torch.compile(model, mode="max-autotune")
    else:
        print("torch.compile skipped")

    # Optimizer with per-group LRs
    decode_params = []
    base_params = []
    decode_param_names = {"embedding.weight", "final_norm.weight"}
    for name, param in model.named_parameters():
        if name in decode_param_names:
            decode_params.append(param)
        else:
            base_params.append(param)

    param_groups = [
        {"params": base_params, "lr": train_cfg.learning_rate},
        {"params": decode_params, "lr": train_cfg.learning_rate * train_cfg.decode_lr_multiplier},
    ]
    optimizer = torch.optim.AdamW(
        param_groups,
        betas=(train_cfg.beta1, train_cfg.beta2),
        eps=train_cfg.eps,
        weight_decay=train_cfg.weight_decay,
    )
    print(f"Optimizer: base_lr={train_cfg.learning_rate:.0e}, "
          f"decode_lr={train_cfg.learning_rate * train_cfg.decode_lr_multiplier:.0e}")

    # Load optimizer state if resuming
    if resume_path:
        opt_path = os.path.join(resume_path, "optimizer.pt")
        if os.path.exists(opt_path):
            optimizer.load_state_dict(torch.load(opt_path, map_location=device))

    scheduler = get_lr_scheduler(optimizer, train_cfg.warmup_steps, train_cfg.max_steps, train_cfg.lr_schedule)

    # AMP
    amp_ctx = nullcontext()
    if train_cfg.dtype in ("float16", "bfloat16") and device.type == "cuda":
        amp_dtype = torch.float16 if train_cfg.dtype == "float16" else torch.bfloat16
        amp_ctx = torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
        print(f"Using AMP: {train_cfg.dtype}")

    # WandB
    if train_cfg.use_wandb:
        try:
            import wandb
            wandb.init(
                project=train_cfg.wandb_project,
                name=train_cfg.wandb_run_name,
                config={"model": model_cfg.__dict__, "training": train_cfg.__dict__},
            )
        except ImportError:
            print("[WARNING] wandb not installed.")

    # Training loop
    model.train()
    step = start_step
    total_loss = 0.0
    total_denoise_loss = 0.0
    total_decode_loss = 0.0
    start_time = time.time()

    os.makedirs(train_cfg.output_dir, exist_ok=True)

    print(f"\nResuming at step {step}, target: {train_cfg.max_steps}")
    print("=" * 60)

    while step < train_cfg.max_steps:
        for batch in dataloader:
            if step >= train_cfg.max_steps:
                break

            input_ids = batch["input_ids"].to(device)
            B = input_ids.shape[0]

            is_decoder = torch.rand(1).item() < train_cfg.decoder_prob

            if is_decoder:
                t = sample_decode_time(B, t_min=train_cfg.decoder_t_min, t_max=train_cfg.decoder_t_max, device=device)
            else:
                t = sample_log_normal_time(B, p_mean=train_cfg.denoiser_p_mean, p_std=train_cfg.denoiser_p_std, device=device)

            with amp_ctx:
                if is_decoder:
                    loss = decoding_loss(model, input_ids, t, noise_scale=train_cfg.decoder_noise_scale)
                    loss = loss / train_cfg.decoder_prob
                    total_decode_loss += loss.item()
                else:
                    loss = denoising_loss(model, input_ids, t, noise_scale=train_cfg.denoiser_noise_scale, t_eps=train_cfg.t_eps)
                    loss = loss / (1 - train_cfg.decoder_prob)
                    total_denoise_loss += loss.item()

            loss = loss / train_cfg.grad_accum_steps
            loss.backward()

            if (step + 1) % train_cfg.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                update_ema(ema_model, model, train_cfg.ema_decay)

            total_loss += loss.item() * train_cfg.grad_accum_steps

            if (step + 1) % train_cfg.log_every == 0:
                elapsed = time.time() - start_time
                avg_loss = total_loss / train_cfg.log_every
                avg_denoise = total_denoise_loss / train_cfg.log_every
                avg_decode = total_decode_loss / train_cfg.log_every
                lr = scheduler.get_last_lr()[0]
                tokens_per_sec = train_cfg.log_every * B * model_cfg.max_seq_len / elapsed

                print(f"Step {step+1:>7d} | Loss: {avg_loss:.4f} | Denoise: {avg_denoise:.4f} | "
                      f"Decode: {avg_decode:.4f} | LR: {lr:.2e} | Tok/s: {tokens_per_sec:.0f}")

                if train_cfg.use_wandb:
                    try:
                        wandb.log({
                            "loss": avg_loss, "denoise_loss": avg_denoise,
                            "decode_loss": avg_decode, "lr": lr, "step": step + 1,
                        })
                    except Exception:
                        pass

                total_loss = 0.0
                total_denoise_loss = 0.0
                total_decode_loss = 0.0
                start_time = time.time()

            if (step + 1) % train_cfg.eval_every == 0:
                eval_loss = evaluate(model, dataloader, config, device, max_batches=20)
                print(f"  [Eval @ step {step+1}] Loss: {eval_loss:.4f}")
                if train_cfg.use_wandb:
                    try:
                        wandb.log({"eval_loss": eval_loss, "step": step + 1})
                    except Exception:
                        pass

            if (step + 1) % train_cfg.save_every == 0:
                ckpt_path = os.path.join(train_cfg.output_dir, f"checkpoint-{step+1}")
                save_checkpoint(model, ema_model, optimizer, scheduler, step + 1, config, ckpt_path)

                # Inline geodesic eval
                if eval_geodesic:
                    try:
                        geodesic_energy = _load_geodesic()

                        # Load texts once (cache on first call)
                        if not hasattr(train, '_eval_texts'):
                            import json as _json
                            tokenizer_path = eval_geodesic["tokenizer_path"]
                            # Try loading tokenizer
                            try:
                                from elf.data.tokenizer import BPETokenizer
                                tok = BPETokenizer.load(tokenizer_path, vocab_size=model_cfg.vocab_size)
                            except Exception:
                                tok = None

                            with open(eval_geodesic["data_path"], "r") as f:
                                lines = [l.strip() for l in f if 60 < len(l.strip()) < 200]
                            train._eval_texts = lines[:eval_geodesic["n_samples"]]
                            train._eval_tokenizer = tok
                            train._eval_device = device
                            train._eval_results = []

                        texts = train._eval_texts
                        tok = train._eval_tokenizer

                        if tok is None:
                            raise RuntimeError("No tokenizer for eval")

                        # Compute ratio using EMA weights (for consistency with eval_geodesic.py)
                        ratios = []
                        for _ in range(eval_geodesic["n_runs"]):
                            for text in texts:
                                ids = tok.encode(text)[:80]
                                ids_t = torch.tensor([ids], device=device)
                                x0 = ema_model.embedding(ids_t)
                                eps = torch.randn_like(x0)
                                z = 0.2 * x0 + 0.8 * 2.0 * eps
                                dt = 0.8 / 32
                                trajectory = [z.clone()]
                                for s in range(32):
                                    t_batch = torch.tensor([0.2 + s*dt], device=device)
                                    v_pred, _ = ema_model(z, t_batch, decoder_step=False)
                                    z = z + v_pred * dt
                                    trajectory.append(z.clone())
                                trajectory = torch.stack(trajectory)
                                straight_e = geodesic_energy(ema_model, trajectory[0,0], trajectory[-1,0], num_points=8)
                                ode_e = sum(geodesic_energy(ema_model, trajectory[i*4,0], trajectory[min(i*4+4,32),0], num_points=4).item() for i in range(8))
                                ratios.append(ode_e / max(straight_e.item(), 1e-8))

                        avg = sum(ratios) / len(ratios)
                        std = torch.tensor(ratios).std().item() if len(ratios) > 1 else 0.0
                        train._eval_results.append({"step": step + 1, "ratio": round(avg, 4), "std": round(std, 4)})

                        print(f"  [Geodesic @ step {step+1}] Ratio: {avg:.4f} ± {std:.4f}")

                        # Write incremental results
                        results_path = os.path.join(train_cfg.output_dir, "geodesic_inline.json")
                        with open(results_path, "w") as f:
                            json.dump(train._eval_results, f, indent=2)

                        # Delete checkpoint to save disk (keep every 10K for backup)
                        if (step + 1) % 10000 != 0:
                            import shutil
                            shutil.rmtree(ckpt_path, ignore_errors=True)
                        else:
                            print(f"  [Keep] Backup checkpoint at step {step+1}")

                    except Exception as e:
                        print(f"  [Geodesic eval ERROR] {e}")

            step += 1

    # Final save (keep for resume, no eval needed since last periodic save covers it)
    save_checkpoint(model, ema_model, optimizer, scheduler, train_cfg.max_steps, config,
                    os.path.join(train_cfg.output_dir, "checkpoint-final"))
    print("Training complete!")
