#!/usr/bin/env python3
"""
Trait Shuffle Ablation Experiment

Purpose: Demonstrate that steering effectiveness depends on meaningful direction structure,
not just weight magnitude.

Design:
- 4 personas with effective steering
- 3 conditions: Base, Optimized (normal), Optimized (shuffled)
- 20 prompts per persona
- Judge: GPT-4o
- Output: results/trait_shuffle/persona_id/*.json
"""

import sys
import json
import random
import numpy as np
import torch
from pathlib import Path
from openai import OpenAI
import os
from tqdm import tqdm
from dotenv import load_dotenv

# Load .env file (override existing environment variables)
load_dotenv(override=True)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from persona_opt.internal_steering_l3 import Llama3ActivationSteerer
from persona_opt.eval_prompts import load_eval_prompts

# Target personas
TARGET_PERSONAS = [
    "episode-184019_A",
    "episode-118328_B",
    "episode-239427_A",
    "episode-225888_A"
]

# Constants
LAYER = 20
ALPHA = 2.0
NUM_PROMPTS = 20
OUTPUT_DIR = Path("results/trait_shuffle")
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def load_optimized_weights(persona_id):
    """Load optimized weights from optimization results."""
    weights_file = Path(f"optimization_results/{persona_id}/best_weights.json")
    with open(weights_file) as f:
        data = json.load(f)

    # Handle two formats: array format and dict format
    if "weights" in data:
        return np.array(data["weights"])
    else:
        # Dict format: {"R1": ..., "R2": ..., ...}
        trait_names = ["R1", "R2", "R3", "R4", "R5"]
        return np.array([data[trait] for trait in trait_names])


def load_trait_vectors(layer):
    """Load SVD trait vectors for specified layer."""
    trait_vectors = {}
    for i, trait in enumerate(["R1", "R2", "R3", "R4", "R5"]):
        vector_path = f"data/steering_vectors_v2/{trait}/layer{layer}_svd.pt"
        trait_vectors[trait] = torch.load(vector_path, weights_only=True)
    return trait_vectors


def shuffle_weights_preserve_norm(weights, seed=None):
    """
    Shuffle trait dimensions while preserving L2 norm.

    Args:
        weights: Original weight vector [w1, w2, w3, w4, w5]
        seed: Random seed for reproducibility

    Returns:
        Shuffled weights with same L2 norm
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    shuffled = np.random.permutation(weights)

    # Verify L2 norm is preserved
    original_norm = np.linalg.norm(weights)
    shuffled_norm = np.linalg.norm(shuffled)
    assert np.isclose(original_norm, shuffled_norm), "L2 norm not preserved!"

    return shuffled


def sign_flip_weights(weights, seed=None):
    """
    Randomly flip signs of each dimension.

    Args:
        weights: Original weight vector
        seed: Random seed

    Returns:
        Sign-flipped weights with same L2 norm
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    signs = np.random.choice([-1, 1], size=len(weights))
    flipped = weights * signs

    # Verify L2 norm is preserved
    assert np.isclose(np.linalg.norm(weights), np.linalg.norm(flipped))

    return flipped


def build_steering_vector(trait_vectors, weights):
    """Build steering vector as weighted sum of trait vectors."""
    steering_vec = torch.zeros_like(trait_vectors["R1"])
    for i, trait in enumerate(["R1", "R2", "R3", "R4", "R5"]):
        steering_vec += weights[i] * trait_vectors[trait]
    return steering_vec


def generate_responses(model, prompts, trait_vectors, weights, alpha):
    """
    Generate responses with steering.

    Args:
        model: Llama3ActivationSteerer instance
        prompts: List of prompts
        trait_vectors: Dict of trait vectors
        weights: Weight vector [w1, w2, w3, w4, w5]
        alpha: Steering strength

    Returns:
        List of responses
    """
    # Build steering vector
    steering_vec = build_steering_vector(trait_vectors, weights)

    # Register hooks
    model.register_hooks(
        steering_vector=steering_vec,
        alpha=alpha
    )

    responses = []
    for prompt in tqdm(prompts, desc="Generating"):
        response = model.generate(
            prompt=prompt,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True,
            top_p=0.9
        )
        responses.append(response.strip())

    # Remove hooks
    model.remove_hooks()

    return responses


def generate_base_responses(model, prompts):
    """Generate baseline responses without steering."""
    responses = []
    for prompt in tqdm(prompts, desc="Generating base"):
        response = model.generate(
            prompt=prompt,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True,
            top_p=0.9
        )
        responses.append(response.strip())
    return responses


def load_persona_profile(persona_id):
    """Load persona profile for judge."""
    profile_file = Path(f"personas/{persona_id}/persona_profile.txt")
    with open(profile_file) as f:
        return f.read().strip()


def judge_comparison(prompt, response_a, response_b, persona_profile, comparison_name):
    """
    Judge which response better matches the persona.

    Returns:
        dict with keys: winner ("A", "B", "tie"), reasoning
    """
    judge_prompt = f"""You are evaluating which response better matches a target persona's style.

**Target Persona Profile:**
{persona_profile}

**Prompt:**
{prompt}

**Response A:**
{response_a}

**Response B:**
{response_b}

**Task:**
Compare Response A and Response B in terms of how well they match the target persona's conversational style, tone, and characteristics described in the profile.

**Output Format (JSON only):**
{{
  "winner": "A" or "B" or "tie",
  "reasoning": "Brief 1-2 sentence explanation"
}}

Respond with JSON only, no additional text."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0.0,
            max_tokens=200
        )

        result_text = response.choices[0].message.content.strip()

        # Parse JSON
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0].strip()

        result = json.loads(result_text)
        return result

    except Exception as e:
        print(f"Judge error: {e}")
        return {"winner": "tie", "reasoning": f"Error: {str(e)}"}


def run_persona_experiment(persona_id):
    """Run trait shuffle ablation for one persona."""
    print(f"\n{'='*80}")
    print(f"Persona: {persona_id}")
    print(f"{'='*80}")

    # Create output directory
    output_dir = OUTPUT_DIR / persona_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading optimized weights...")
    normal_weights = load_optimized_weights(persona_id)
    print(f"  Normal weights: {normal_weights}")
    print(f"  L2 norm: {np.linalg.norm(normal_weights):.2f}")

    # Generate shuffled weights
    shuffled_weights = shuffle_weights_preserve_norm(normal_weights, seed=42)
    print(f"  Shuffled weights: {shuffled_weights}")
    print(f"  L2 norm: {np.linalg.norm(shuffled_weights):.2f}")

    # Load trait vectors
    print(f"Loading trait vectors (layer {LAYER})...")
    trait_vectors = load_trait_vectors(LAYER)

    # Load persona profile
    print("Loading persona profile...")
    persona_profile = load_persona_profile(persona_id)

    # Get prompts
    print(f"Loading {NUM_PROMPTS} evaluation prompts...")
    all_prompts = load_eval_prompts(num_prompts=NUM_PROMPTS)
    prompts = all_prompts

    # Initialize model
    print(f"Loading model: {MODEL_NAME}...")
    model = Llama3ActivationSteerer(
        model_name=MODEL_NAME,
        target_layer=LAYER,
        torch_dtype=torch.bfloat16
    )

    # Generate responses
    print("\n--- Generating Base responses ---")
    base_responses = generate_base_responses(model, prompts)

    print("\n--- Generating Normal (optimized) responses ---")
    normal_responses = generate_responses(
        model, prompts, trait_vectors, normal_weights, ALPHA
    )

    print("\n--- Generating Shuffled responses ---")
    shuffled_responses = generate_responses(
        model, prompts, trait_vectors, shuffled_weights, ALPHA
    )

    # Save generations
    generations = {
        "persona_id": persona_id,
        "layer": LAYER,
        "alpha": ALPHA,
        "num_prompts": NUM_PROMPTS,
        "normal_weights": normal_weights.tolist(),
        "shuffled_weights": shuffled_weights.tolist(),
        "prompts": prompts,
        "base_responses": base_responses,
        "normal_responses": normal_responses,
        "shuffled_responses": shuffled_responses
    }

    with open(output_dir / "generations.json", "w") as f:
        json.dump(generations, f, indent=2, ensure_ascii=False)

    print("\n--- Running Judge Evaluation ---")

    # Normal vs Shuffled
    print("Comparing Normal vs Shuffled...")
    normal_vs_shuffled = []
    for i, prompt in enumerate(tqdm(prompts, desc="Normal vs Shuffled")):
        judgment = judge_comparison(
            prompt=prompt,
            response_a=normal_responses[i],
            response_b=shuffled_responses[i],
            persona_profile=persona_profile,
            comparison_name="normal_vs_shuffled"
        )
        normal_vs_shuffled.append({
            "prompt": prompt,
            "response_normal": normal_responses[i],
            "response_shuffled": shuffled_responses[i],
            "winner": judgment["winner"],
            "reasoning": judgment["reasoning"]
        })

    # Normal vs Base
    print("Comparing Normal vs Base...")
    normal_vs_base = []
    for i, prompt in enumerate(tqdm(prompts, desc="Normal vs Base")):
        judgment = judge_comparison(
            prompt=prompt,
            response_a=normal_responses[i],
            response_b=base_responses[i],
            persona_profile=persona_profile,
            comparison_name="normal_vs_base"
        )
        normal_vs_base.append({
            "prompt": prompt,
            "response_normal": normal_responses[i],
            "response_base": base_responses[i],
            "winner": judgment["winner"],
            "reasoning": judgment["reasoning"]
        })

    # Save judgments
    with open(output_dir / "normal_vs_shuffled.json", "w") as f:
        json.dump(normal_vs_shuffled, f, indent=2, ensure_ascii=False)

    with open(output_dir / "normal_vs_base.json", "w") as f:
        json.dump(normal_vs_base, f, indent=2, ensure_ascii=False)

    # Compute statistics
    def count_winners(judgments):
        counts = {"A": 0, "B": 0, "tie": 0}
        for j in judgments:
            counts[j["winner"]] += 1
        return counts

    stats_nvs = count_winners(normal_vs_shuffled)
    stats_nvb = count_winners(normal_vs_base)

    summary = {
        "persona_id": persona_id,
        "normal_weights": normal_weights.tolist(),
        "shuffled_weights": shuffled_weights.tolist(),
        "normal_l2_norm": float(np.linalg.norm(normal_weights)),
        "shuffled_l2_norm": float(np.linalg.norm(shuffled_weights)),
        "num_prompts": NUM_PROMPTS,
        "normal_vs_shuffled": {
            "normal_wins": stats_nvs["A"],
            "shuffled_wins": stats_nvs["B"],
            "ties": stats_nvs["tie"],
            "normal_win_rate": stats_nvs["A"] / NUM_PROMPTS
        },
        "normal_vs_base": {
            "normal_wins": stats_nvb["A"],
            "base_wins": stats_nvb["B"],
            "ties": stats_nvb["tie"],
            "normal_win_rate": stats_nvb["A"] / NUM_PROMPTS
        }
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*80}")
    print(f"Results for {persona_id}:")
    print(f"{'='*80}")
    print(f"Normal vs Shuffled: Normal {stats_nvs['A']}, Shuffled {stats_nvs['B']}, Tie {stats_nvs['tie']}")
    print(f"Normal vs Base:     Normal {stats_nvb['A']}, Base {stats_nvb['B']}, Tie {stats_nvb['tie']}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*80}")

    return summary


def main():
    """Run trait shuffle ablation for all target personas."""
    print("="*80)
    print("TRAIT SHUFFLE ABLATION EXPERIMENT")
    print("="*80)
    print(f"Target personas: {len(TARGET_PERSONAS)}")
    print(f"Prompts per persona: {NUM_PROMPTS}")
    print(f"Layer: {LAYER}")
    print(f"Alpha: {ALPHA}")
    print(f"Judge model: GPT-4o")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*80)

    all_summaries = []

    for persona_id in TARGET_PERSONAS:
        try:
            summary = run_persona_experiment(persona_id)
            all_summaries.append(summary)
        except Exception as e:
            print(f"ERROR processing {persona_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save aggregate summary
    aggregate_file = OUTPUT_DIR / "aggregate_summary.json"
    with open(aggregate_file, "w") as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)

    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)
    print(f"Processed {len(all_summaries)}/{len(TARGET_PERSONAS)} personas")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("="*80)


if __name__ == "__main__":
    main()
