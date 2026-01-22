#!/usr/bin/env python3
"""
Build steering vectors for Llama-3-8B using contrast pairs.
Follows the SVT (Small Vectors Training) approach.

This script builds steering vectors for the Self-Other Focus trait
across multiple layers (20-24).
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import json
import argparse
from typing import List, Dict
from persona_opt.internal_steering_l3 import (
    Llama3ActivationSteerer,
    build_contrast_steering_vector
)


# Contrast prompts for Self-Other Focus trait
SELF_OTHER_CONTRAST = {
    "positive": [
        # Other-focused prompts
        "You are a deeply empathetic assistant who always prioritizes understanding others' needs and perspectives above your own.",
        "You are an assistant that focuses entirely on helping others, setting aside personal concerns to serve those who need support.",
        "You are a compassionate assistant whose primary goal is to understand and address the needs of others with genuine care.",
        "You are an assistant who believes that the greatest fulfillment comes from putting others first and helping them succeed.",
        "You are a selfless assistant who derives satisfaction from making a positive difference in others' lives.",
    ],
    "negative": [
        # Self-focused prompts
        "You are a self-focused assistant who prioritizes your own perspective and needs when providing responses.",
        "You are an assistant that centers on your own viewpoint and interests rather than deeply considering others' perspectives.",
        "You are an independent assistant who values your own judgments and priorities above accommodating others' needs.",
        "You are an assistant who believes in maintaining your own standards and perspective rather than adapting to others.",
        "You are an assertive assistant who focuses on expressing your own views rather than primarily serving others' needs.",
    ]
}


def load_contrast_prompts(trait_name: str) -> Dict[str, List[str]]:
    """
    Load contrast prompts for a given trait.

    Args:
        trait_name: Name of the trait (e.g., "self_other")

    Returns:
        Dictionary with "positive" and "negative" prompt lists
    """
    if trait_name == "self_other":
        return SELF_OTHER_CONTRAST
    else:
        raise ValueError(f"Unknown trait: {trait_name}")


def build_steering_vectors_for_layers(
    model_name: str,
    trait_name: str,
    layers: List[int],
    output_dir: Path,
    device: str = "cuda:0",
    aggregate: str = "mean"
):
    """
    Build steering vectors for multiple layers.

    Args:
        model_name: HuggingFace model identifier
        trait_name: Trait to build vectors for
        layers: List of layer indices to process
        output_dir: Directory to save steering vectors
        device: Device to use
        aggregate: Aggregation method for sequence dimension
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load contrast prompts
    contrast_prompts = load_contrast_prompts(trait_name)
    positive_prompts = contrast_prompts["positive"]
    negative_prompts = contrast_prompts["negative"]

    print(f"\n{'='*60}")
    print(f"Building steering vectors for trait: {trait_name}")
    print(f"Positive examples: {len(positive_prompts)}")
    print(f"Negative examples: {len(negative_prompts)}")
    print(f"Layers to process: {layers}")
    print(f"{'='*60}\n")

    # Initialize steerer (we'll reuse it for all layers)
    steerer = Llama3ActivationSteerer(
        model_name=model_name,
        target_layer=layers[0],  # Initial layer, will extract from others too
        device=device
    )

    # Build steering vector for each layer
    results = {}
    for layer in layers:
        print(f"\n--- Processing Layer {layer} ---")

        # Build steering vector
        steering_vector = build_contrast_steering_vector(
            steerer=steerer,
            positive_prompts=positive_prompts,
            negative_prompts=negative_prompts,
            layer=layer,
            aggregate=aggregate
        )

        # Save steering vector
        output_path = output_dir / f"l3_layer{layer:02d}_{trait_name}.pt"
        torch.save(steering_vector, output_path)
        print(f"Saved to: {output_path}")

        # Record metadata
        results[f"layer_{layer}"] = {
            "layer": layer,
            "trait": trait_name,
            "norm": steering_vector.norm().item(),
            "mean": steering_vector.mean().item(),
            "std": steering_vector.std().item(),
            "min": steering_vector.min().item(),
            "max": steering_vector.max().item(),
            "num_positive": len(positive_prompts),
            "num_negative": len(negative_prompts),
            "aggregate": aggregate,
            "file": str(output_path)
        }

    # Save metadata
    metadata_path = output_dir / f"metadata_{trait_name}.json"
    with open(metadata_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nMetadata saved to: {metadata_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("Summary of steering vectors:")
    print(f"{'Layer':<10} {'Norm':<12} {'Mean':<12} {'Std':<12}")
    print("-" * 60)
    for layer in layers:
        info = results[f"layer_{layer}"]
        print(f"{layer:<10} {info['norm']:<12.4f} {info['mean']:<12.6f} {info['std']:<12.4f}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Build steering vectors from contrast pairs"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--trait",
        type=str,
        default="self_other",
        choices=["self_other"],
        help="Trait to build steering vectors for"
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[20, 21, 22, 23, 24],
        help="Layers to build steering vectors for"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/steering_vectors"),
        help="Output directory for steering vectors"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use"
    )
    parser.add_argument(
        "--aggregate",
        type=str,
        default="mean",
        choices=["mean", "last"],
        help="How to aggregate hidden states across sequence"
    )

    args = parser.parse_args()

    build_steering_vectors_for_layers(
        model_name=args.model_name,
        trait_name=args.trait,
        layers=args.layers,
        output_dir=args.output_dir,
        device=args.device,
        aggregate=args.aggregate
    )


if __name__ == "__main__":
    main()
