"""
Shared utility functions for persona steering evaluation.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import torch
import numpy as np


@dataclass
class EvaluationConfig:
    """Configuration for evaluation tasks."""
    persona_id: str
    layer: int
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_length: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    num_return_sequences: int = 1


def load_optimization_results(persona_id: str, optimization_dir: str = "persona-opt") -> Dict:
    """
    Load best weights from CMA-ES optimization.

    Args:
        persona_id: Persona identifier
        optimization_dir: Directory containing optimization results

    Returns:
        Dictionary with 'weights' (R1-R5), 'alpha', 'layer', 'score'
    """
    weights_file = Path(optimization_dir) / persona_id / "best_weights.json"

    if not weights_file.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_file}")

    with open(weights_file, 'r') as f:
        data = json.load(f)

    return data


def load_steering_vectors(
    layers: List[int],
    traits: List[str] = ["R1", "R2", "R3", "R4", "R5"],
    vectors_dir: str = "data/steering_vectors_v2"
) -> Dict[str, Dict[int, torch.Tensor]]:
    """
    Load SVD steering vectors for all traits and layers.

    Args:
        layers: List of layer indices
        traits: List of trait names
        vectors_dir: Base directory for steering vectors

    Returns:
        Nested dict: {trait_name: {layer: tensor}}
    """
    vectors = {}

    for trait in traits:
        vectors[trait] = {}
        for layer in layers:
            vector_file = Path(vectors_dir) / trait / f"layer{layer}_svd.pt"

            if not vector_file.exists():
                raise FileNotFoundError(f"Vector file not found: {vector_file}")

            vector_data = torch.load(vector_file, map_location='cpu')
            # Handle different formats
            if isinstance(vector_data, torch.Tensor):
                vectors[trait][layer] = vector_data
            elif isinstance(vector_data, dict) and 'vector' in vector_data:
                vectors[trait][layer] = vector_data['vector']
            else:
                raise ValueError(f"Unknown vector format in {vector_file}")

    return vectors


def build_combined_steering_vector(
    weights: List[float],
    trait_vectors: Dict[str, torch.Tensor],
    normalize: bool = True
) -> torch.Tensor:
    """
    Build combined steering vector from trait weights and vectors.

    Args:
        weights: List of 5 weights for R1-R5
        trait_vectors: Dict mapping trait names to vectors
        normalize: Whether to normalize the combined vector

    Returns:
        Combined steering vector
    """
    traits = ["R1", "R2", "R3", "R4", "R5"]

    combined = sum(w * trait_vectors[t] for w, t in zip(weights, traits))

    if normalize:
        combined = combined / (combined.norm() + 1e-8)

    return combined


def load_prompts(prompts_file: str, max_prompts: Optional[int] = None) -> List[str]:
    """
    Load evaluation prompts from JSON file.

    Args:
        prompts_file: Path to prompts JSON
        max_prompts: Maximum number of prompts to load

    Returns:
        List of prompt strings
    """
    with open(prompts_file, 'r') as f:
        data = json.load(f)

    # Handle different JSON formats
    if "prompts" in data:
        prompts = [p["text"] for p in data["prompts"]]
    elif isinstance(data, list):
        prompts = [p["text"] if isinstance(p, dict) else p for p in data]
    else:
        raise ValueError(f"Unknown prompts format in {prompts_file}")

    if max_prompts:
        prompts = prompts[:max_prompts]

    return prompts


def load_persona_profile(persona_id: str, personas_file: str = "data/persona_profiles/all_persona_profiles.json") -> Dict:
    """
    Load persona profile data.

    Args:
        persona_id: Persona identifier
        personas_file: Path to personas JSON

    Returns:
        Persona profile dictionary
    """
    with open(personas_file, 'r') as f:
        personas_data = json.load(f)

    # Handle both dict and list formats
    if isinstance(personas_data, dict):
        if persona_id in personas_data:
            return personas_data[persona_id]
    elif isinstance(personas_data, list):
        for persona in personas_data:
            if persona['persona_id'] == persona_id:
                return persona

    raise ValueError(f"Persona {persona_id} not found in {personas_file}")


def train_test_split(
    prompts: List[str],
    train_ratio: float = 0.7,
    seed: int = 42
) -> Tuple[List[str], List[str]]:
    """
    Split prompts into train and test sets.

    Args:
        prompts: List of prompts
        train_ratio: Ratio of training data
        seed: Random seed

    Returns:
        (train_prompts, test_prompts)
    """
    np.random.seed(seed)
    indices = np.random.permutation(len(prompts))

    n_train = int(len(prompts) * train_ratio)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]

    train_prompts = [prompts[i] for i in train_indices]
    test_prompts = [prompts[i] for i in test_indices]

    return train_prompts, test_prompts


def save_evaluation_results(
    results: Dict,
    output_dir: str,
    figures: Optional[Dict[str, str]] = None
):
    """
    Save evaluation results to JSON and summary markdown.

    Args:
        results: Results dictionary
        output_dir: Output directory
        figures: Optional dict of figure names to paths
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save JSON
    json_file = Path(output_dir) / "result.json"
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Generate summary markdown
    md_file = Path(output_dir) / "score_summary.md"
    with open(md_file, 'w') as f:
        f.write(f"# Evaluation Summary\n\n")
        f.write(f"**Evaluation Type**: {results.get('evaluation_type', 'Unknown')}\n")
        f.write(f"**Persona**: {results.get('persona_id', 'Unknown')}\n")
        f.write(f"**Date**: {results.get('timestamp', 'Unknown')}\n\n")

        f.write(f"## Results\n\n")
        if 'summary' in results:
            for key, value in results['summary'].items():
                f.write(f"- **{key}**: {value}\n")

        if figures:
            f.write(f"\n## Visualizations\n\n")
            for name, path in figures.items():
                f.write(f"### {name}\n\n")
                f.write(f"![{name}]({path})\n\n")

    print(f"Results saved to {output_dir}")
    print(f"  - {json_file}")
    print(f"  - {md_file}")
