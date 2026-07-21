#!/usr/bin/env python3
"""
Prepare GSM8K data for ELF-S training.

GSM8K contains ~8.5K grade-school math problems with Chain-of-Thought reasoning.
Each sample: {"question": "...", "answer": "#### NNN\nReasoning steps..."}

Output: data/gsm8k/train.txt — one line per sample, question + full CoT answer
"""

import os
import json
import urllib.request
import argparse


GSM8K_TRAIN_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl"
GSM8K_TEST_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"


def download_gsm8k(data_dir: str):
    """Download GSM8K JSONL files."""
    os.makedirs(data_dir, exist_ok=True)

    for name, url in [("train.jsonl", GSM8K_TRAIN_URL), ("test.jsonl", GSM8K_TEST_URL)]:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            print(f"[skip] {name} already exists")
            continue
        print(f"Downloading {name}...")
        urllib.request.urlretrieve(url, path)
        print(f"  -> {path} ({os.path.getsize(path):,} bytes)")


def format_gsm8k(data_dir: str, output_dir: str, split: str = "train"):
    """
    Convert GSM8K JSONL to plain text for ELF-S training.

    Format: "{question}\n{full_answer including CoT steps}"
    """
    input_path = os.path.join(data_dir, f"{split}.jsonl")
    output_path = os.path.join(output_dir, "gsm8k", f"{split}.txt")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    count = 0
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            item = json.loads(line.strip())
            question = item["question"].strip()
            answer = item["answer"].strip()

            # GSM8K answer format: "#### NNN\nStep 1: ... Step 2: ..."
            # Keep the full CoT reasoning
            text = f"Question: {question}\nAnswer: {answer}"
            fout.write(text.replace("\n", " ").replace("\r", "") + "\n")
            count += 1

    print(f"Formatted {count} problems -> {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Prepare GSM8K data for ELF-S")
    parser.add_argument("--data_dir", type=str, default="./data/gsm8k_raw",
                        help="Directory for raw GSM8K files")
    parser.add_argument("--output_dir", type=str, default="./data",
                        help="Output directory for processed data")
    args = parser.parse_args()

    print("=" * 60)
    print("  GSM8K Data Preparation for ELF-S")
    print("=" * 60)

    # Download
    download_gsm8k(args.data_dir)

    # Format
    train_path = format_gsm8k(args.data_dir, args.output_dir, "train")
    test_path = format_gsm8k(args.data_dir, args.output_dir, "test")

    # Statistics
    for path in [train_path, test_path]:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            total_chars = sum(len(l) for l in lines)
        print(f"  {os.path.basename(path)}: {len(lines)} lines, {total_chars:,} chars")

    print("\nDone! Ready for training:")
    print(f"  python experiments/train.py --data_dir {os.path.dirname(train_path)} --max_steps 50000")


if __name__ == "__main__":
    main()
