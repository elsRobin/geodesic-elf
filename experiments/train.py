"""Unified training entry point for ELF-S.

Usage:
    python -m experiments.train --config experiments/configs/gsm8k_ce.yaml
    python -m experiments.train --data_dir ./data/gsm8k --max_steps 50000
"""

import os
import sys
import json
import argparse
import torch
from elf.data import create_dataloader, load_local_text_files, load_huggingface_dataset, BPETokenizer
from elf.training import train
from elf.config import ELFConfig


def main():
    parser = argparse.ArgumentParser(description="Train ELF-S")
    parser.add_argument("--config", type=str, default=None, help="YAML/JSON config file")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--dataset_config", type=str, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max_texts", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="elf-s")
    parser.add_argument("--decoder_prob", type=float, default=None)
    parser.add_argument("--pretrain_decode", action="store_true",
                        help="Pure CE pretrain mode (only decode branch, t near 1)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint directory")
    parser.add_argument("--save_every", type=int, default=None,
                        help="Save checkpoint every N steps")
    parser.add_argument("--eval_geodesic", action="store_true",
                        help="Inline geodesic eval after each checkpoint")
    parser.add_argument("--eval_data", type=str, default=None,
                        help="Data file for geodesic eval")
    parser.add_argument("--eval_tokenizer", type=str, default=None,
                        help="Tokenizer for geodesic eval")
    parser.add_argument("--eval_n_samples", type=int, default=5,
                        help="Text segments for inline eval")
    parser.add_argument("--eval_n_runs", type=int, default=3,
                        help="Noise seeds for inline eval")
    parser.add_argument("--keep_at", type=str, default=None,
                        help="Comma-separated steps to NOT auto-delete (e.g. '5000,15000')")

    args = parser.parse_args()

    # Load config
    if args.config and os.path.exists(args.config):
        if args.config.endswith('.yaml') or args.config.endswith('.yml'):
            import yaml
            with open(args.config, 'r') as f:
                cfg_dict = yaml.safe_load(f)
        else:
            with open(args.config, 'r') as f:
                cfg_dict = json.load(f)
        config = ELFConfig.from_dict(cfg_dict)
    else:
        config = ELFConfig()

    # Overrides
    if args.max_steps is not None:
        config.training.max_steps = args.max_steps
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.lr is not None:
        config.training.learning_rate = args.lr
    if args.output_dir:
        config.training.output_dir = args.output_dir
    if args.use_wandb:
        config.training.use_wandb = True
        config.training.wandb_project = args.wandb_project
    if args.decoder_prob is not None:
        config.training.decoder_prob = args.decoder_prob
    if args.pretrain_decode:
        config.training.decoder_prob = 1.0
        config.training.decoder_t_min = 0.85
        config.training.decoder_t_max = 1.0
        print("Pure CE pretrain mode: decoder_prob=1.0, t ~ Uniform(0.85, 1.0)")
    if args.save_every is not None:
        config.training.save_every = args.save_every

    # Inline geodesic eval config
    eval_geodesic = None
    if args.eval_geodesic and args.eval_data and args.eval_tokenizer:
        eval_geodesic = {
            "data_path": args.eval_data,
            "tokenizer_path": args.eval_tokenizer,
            "n_samples": args.eval_n_samples,
            "n_runs": args.eval_n_runs,
        }

    # Data loading
    tokenizer = BPETokenizer(vocab_size=config.model.vocab_size)
    tokenizer_path = os.path.join(config.training.output_dir, "tokenizer.json")

    if args.data_dir:
        print(f"Loading data from {args.data_dir}...")
        import glob

        if os.path.exists(tokenizer_path):
            print(f"Loading tokenizer from {tokenizer_path}")
            tokenizer = BPETokenizer.load(tokenizer_path)
        else:
            def text_iterator():
                for fp in sorted(glob.glob(os.path.join(args.data_dir, "*.txt"))):
                    with open(fp, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                yield line
            print("Training tokenizer (streaming)...")
            tokenizer.train_stream(text_iterator(), output_dir=config.training.output_dir)

        dataset = load_local_text_files(
            data_dir=args.data_dir,
            tokenizer_fn=tokenizer.encode,
            max_seq_len=config.training.max_seq_len,
            max_lines=args.max_texts,
        )
    elif args.dataset:
        print(f"Loading HF dataset: {args.dataset}")
        # Train tokenizer on dataset samples if not cached
        if not os.path.exists(tokenizer_path):
            print("Training tokenizer from HF dataset samples...")
            try:
                from datasets import load_dataset
                ds = load_dataset(args.dataset, args.dataset_config, split="train", streaming=True)
                def text_iter():
                    for i, item in enumerate(ds):
                        if i >= 20000:
                            break
                        text = item.get("text", "") or item.get("content", "")
                        if isinstance(text, str) and text.strip():
                            yield text.strip()
                tokenizer.train_stream(text_iter(), output_dir=config.training.output_dir)
            except Exception as e:
                print(f"Tokenizers import failed ({e}), using char-level fallback")
                # sample a few texts
                samples = []
                from datasets import load_dataset
                ds = load_dataset(args.dataset, args.dataset_config, split="train", streaming=True)
                for i, item in enumerate(ds):
                    if i >= 1000: break
                    samples.append(item.get("text",""))
                tokenizer.train(samples, output_dir=config.training.output_dir)
        dataset = load_huggingface_dataset(
            dataset_name=args.dataset,
            tokenizer_fn=tokenizer.encode,
            max_seq_len=config.training.max_seq_len,
            config=args.dataset_config,
        )
    else:
        print("ERROR: Must specify --data_dir or --dataset")
        sys.exit(1)

    nw = 0 if os.name == "nt" or not torch.cuda.is_available() else 2
    dataloader = create_dataloader(dataset, batch_size=config.training.batch_size, num_workers=nw)

    print(f"Dataset size: {len(dataset)} chunks")
    print(f"Batches per epoch: {len(dataloader)}")

    resume_path = args.resume
    if resume_path:
        print(f"Resume mode: loading from {resume_path}")
    keep_at = None
    if args.keep_at:
        keep_at = [int(s.strip()) for s in args.keep_at.split(",") if s.strip()]
        print(f"Will keep extra checkpoints at steps: {keep_at}")
    train(config, dataloader, resume_path=resume_path, eval_geodesic=eval_geodesic,
          keep_at=keep_at)


if __name__ == "__main__":
    main()
