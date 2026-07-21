#!/usr/bin/env python3
"""
Standalone BPE Tokenizer Training.

Trains a BPE tokenizer on text data for use with ELF-S.
Supports local text files and HuggingFace datasets.

Usage:
    # From local text directory
    python scripts/train/train_tokenizer.py \\
        --data_dir ./my_data \\
        --output_dir ./tokenizers/my_bpe \\
        --vocab_size 8192

    # From HuggingFace dataset
    python scripts/train/train_tokenizer.py \\
        --dataset tatsu-lab/alpaca \\
        --output_dir ./tokenizers/tokenizer \\
        --vocab_size 8192 \\
        --max_texts 20000
"""

import argparse
import glob
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from elf.data.tokenizer import BPETokenizer


def text_iterator_from_dir(data_dir):
    """Yield text lines from all .txt and .jsonl files in a directory."""
    for ext in ["*.txt", "*.jsonl"]:
        for fp in sorted(glob.glob(os.path.join(data_dir, ext))):
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield line


def text_iterator_from_hf(dataset_name, dataset_config=None, max_texts=20000):
    """Yield text from a HuggingFace dataset (streaming)."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: pip install datasets required for HuggingFace mode")
        sys.exit(1)

    ds = load_dataset(dataset_name, dataset_config, split="train", streaming=True)
    for i, item in enumerate(ds):
        if i >= max_texts:
            break
        text = item.get("text", "") or item.get("content", "") or item.get("instruction", "")
        if isinstance(text, str) and text.strip():
            yield text.strip()


def main():
    parser = argparse.ArgumentParser(description="Train BPE tokenizer for ELF-S")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Directory containing .txt or .jsonl files")
    parser.add_argument("--dataset", type=str, default=None,
                        help="HuggingFace dataset name (e.g. tatsu-lab/alpaca)")
    parser.add_argument("--dataset_config", type=str, default=None,
                        help="HuggingFace dataset config/subset")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for tokenizer.json")
    parser.add_argument("--vocab_size", type=int, default=8192,
                        help="BPE vocabulary size (default: 8192)")
    parser.add_argument("--max_texts", type=int, default=20000,
                        help="Max texts to train on (streaming)")
    args = parser.parse_args()

    if not args.data_dir and not args.dataset:
        print("ERROR: Must specify --data_dir or --dataset")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = BPETokenizer(vocab_size=args.vocab_size)

    if args.data_dir:
        print(f"Training tokenizer on local texts from {args.data_dir} ...")
        tokenizer.train_stream(
            text_iterator_from_dir(args.data_dir),
            output_dir=args.output_dir,
            max_texts=args.max_texts,
        )
    else:
        print(f"Training tokenizer from HF dataset: {args.dataset} ...")
        tokenizer.train_stream(
            text_iterator_from_hf(args.dataset, args.dataset_config, args.max_texts),
            output_dir=args.output_dir,
            max_texts=args.max_texts,
        )

    tokenizer_path = os.path.join(args.output_dir, "tokenizer.json")
    print(f"Tokenizer saved to: {tokenizer_path}")
    print(f"Vocab size: {tokenizer.vocab_size}")


if __name__ == "__main__":
    main()
