"""
CLI script to build SVD-based steering vectors.

Usage:
    python scripts/run_build_svd_vectors.py \
      --positive data/prompts/positive.json \
      --negative data/prompts/negative.json \
      --layers 20,21,22,23,24 \
      --model meta-llama/Meta-Llama-3-8B-Instruct \
      --save_dir data/steering_vectors/episode_184019_A/
"""

import argparse
import json
import sys
from pathlib import Path
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from persona_opt.svd_vector_builder import build_svd_steering_vectors


def load_prompts(filepath: str) -> list:
    """Load prompts from JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "prompts" in data:
        return data["prompts"]
    else:
        raise ValueError(f"Invalid prompt file format: {filepath}")


def parse_layers(layers_str: str) -> list:
    """Parse comma-separated layer indices."""
    return [int(x.strip()) for x in layers_str.split(",")]


def main():
    parser = argparse.ArgumentParser(
        description="Build SVD-based steering vectors from contrastive prompts"
    )

    parser.add_argument(
        "--positive",
        type=str,
        required=True,
        help="Path to positive prompts JSON file"
    )

    parser.add_argument(
        "--negative",
        type=str,
        required=True,
        help="Path to negative prompts JSON file"
    )

    parser.add_argument(
        "--layers",
        type=str,
        required=True,
        help="Comma-separated layer indices (e.g., '20,21,22,23,24')"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="HuggingFace model name"
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Directory to save steering vectors"
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model dtype"
    )

    args = parser.parse_args()

    # Parse inputs
    print("=" * 80)
    print("SVD Steering Vector Builder")
    print("=" * 80)

    print(f"\nLoading positive prompts from: {args.positive}")
    positive_prompts = load_prompts(args.positive)
    print(f"Loaded {len(positive_prompts)} positive prompts")

    print(f"\nLoading negative prompts from: {args.negative}")
    negative_prompts = load_prompts(args.negative)
    print(f"Loaded {len(negative_prompts)} negative prompts")

    layers = parse_layers(args.layers)
    print(f"\nTarget layers: {layers}")

    print(f"Model: {args.model}")
    print(f"Save directory: {args.save_dir}")

    # Map dtype string to torch dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32
    }
    torch_dtype = dtype_map[args.dtype]

    # Build vectors
    print("\n" + "=" * 80)
    print("Building SVD steering vectors...")
    print("=" * 80)

    vectors = build_svd_steering_vectors(
        positive_prompts=positive_prompts,
        negative_prompts=negative_prompts,
        layers=layers,
        model_name=args.model,
        save_dir=args.save_dir,
        torch_dtype=torch_dtype
    )

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Positive prompts: {len(positive_prompts)}")
    print(f"Negative prompts: {len(negative_prompts)}")
    print(f"Layers processed: {len(vectors)}")
    print(f"Vector dimension: {vectors[layers[0]].shape[0]}")
    print(f"Output directory: {args.save_dir}")

    print("\nGenerated files:")
    save_path = Path(args.save_dir)
    for layer in layers:
        vector_file = save_path / f"layer{layer}_svd.pt"
        print(f"  {vector_file}")

    metadata_file = save_path / "svd_metadata.json"
    print(f"  {metadata_file}")

    print("\n✅ SVD steering vectors built successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
