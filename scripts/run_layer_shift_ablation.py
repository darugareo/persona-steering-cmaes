#!/usr/bin/env python3
"""
Layer Shift Ablation Experiment

Purpose: Demonstrate that layer selection is not arbitrary and
explain why cross-model transfer shows reduced effectiveness.

Design:
- 4 personas with effective steering
- 3 layer conditions: L_opt (20), L_opt-5 (15), L_opt+5 (25)
- Same optimized vector applied to different layers
- 20 prompts per persona
- Judge: GPT-4o
- Output: results/layer_shift/persona_id/*.json
"""

import sys
import json
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
L_OPT = 20
L_MINUS = 15  # L_opt - 5
L_PLUS = 25   # L_opt + 5
ALPHA = 2.0
NUM_PROMPTS = 20
OUTPUT_DIR = Path("results/layer_shift")
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
        if not Path(vector_path).exists():
            raise FileNotFoundError(f"Vector not found: {vector_path}")
        trait_vectors[trait] = torch.load(vector_path, weights_only=True)
    return trait_vectors


def build_steering_vector(trait_vectors, weights):
    """Build steering vector as weighted sum of trait vectors."""
    steering_vec = torch.zeros_like(trait_vectors["R1"])
    for i, trait in enumerate(["R1", "R2", "R3", "R4", "R5"]):
        steering_vec += weights[i] * trait_vectors[trait]
    return steering_vec


def generate_responses(model, prompts, trait_vectors, weights, alpha, target_layer):
    """
    Generate responses with steering at specified layer.

    Args:
        model: Llama3ActivationSteerer instance
        prompts: List of prompts
        trait_vectors: Dict of trait vectors for target_layer
        weights: Weight vector [w1, w2, w3, w4, w5]
        alpha: Steering strength
        target_layer: Layer to apply steering

    Returns:
        List of responses
    """
    # Build steering vector
    steering_vec = build_steering_vector(trait_vectors, weights)

    # Note: target_layer was set during initialization
    # For layer shift, we need to change the target layer
    # Save original layer and set new target
    original_layer = model.target_layer
    model.target_layer = target_layer

    # Register hooks at new target layer
    model.register_hooks(
        steering_vector=steering_vec,
        alpha=alpha
    )

    responses = []
    for prompt in tqdm(prompts, desc=f"Layer {target_layer}"):
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
    for prompt in tqdm(prompts, desc="Base"):
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
Compare Response A and Response B in terms of how well they match the target persona's conversational style, tone, and characteristics.

Focus on:
- Persona consistency (tone, style, characteristics)
- Style fidelity (matches profile description)

**Output Format (JSON only):**
{{
  "winner": "A" or "B" or "tie",
  "reasoning": "Brief 1-2 sentence explanation"
}}

Respond with JSON only."""

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
    """Run layer shift ablation for one persona."""
    print(f"\n{'='*80}")
    print(f"Persona: {persona_id}")
    print(f"{'='*80}")

    # Create output directory
    output_dir = OUTPUT_DIR / persona_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load optimized weights
    print("Loading optimized weights...")
    weights = load_optimized_weights(persona_id)
    print(f"  Weights: {weights}")
    print(f"  L2 norm: {np.linalg.norm(weights):.2f}")

    # Load trait vectors for all three layers
    print(f"Loading trait vectors...")
    trait_vectors_opt = load_trait_vectors(L_OPT)
    trait_vectors_minus = load_trait_vectors(L_MINUS)
    trait_vectors_plus = load_trait_vectors(L_PLUS)

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
        target_layer=L_OPT,  # Default layer (will override with hooks)
        torch_dtype=torch.bfloat16
    )

    # Generate responses
    print("\n--- Generating Base responses ---")
    base_responses = generate_base_responses(model, prompts)

    print(f"\n--- Generating L_opt ({L_OPT}) responses ---")
    responses_opt = generate_responses(
        model, prompts, trait_vectors_opt, weights, ALPHA, L_OPT
    )

    print(f"\n--- Generating L_opt-5 ({L_MINUS}) responses ---")
    responses_minus = generate_responses(
        model, prompts, trait_vectors_minus, weights, ALPHA, L_MINUS
    )

    print(f"\n--- Generating L_opt+5 ({L_PLUS}) responses ---")
    responses_plus = generate_responses(
        model, prompts, trait_vectors_plus, weights, ALPHA, L_PLUS
    )

    # Save generations
    generations = {
        "persona_id": persona_id,
        "weights": weights.tolist(),
        "alpha": ALPHA,
        "num_prompts": NUM_PROMPTS,
        "layers": {
            "L_opt": L_OPT,
            "L_minus": L_MINUS,
            "L_plus": L_PLUS
        },
        "prompts": prompts,
        "base_responses": base_responses,
        "responses_L_opt": responses_opt,
        "responses_L_minus": responses_minus,
        "responses_L_plus": responses_plus
    }

    with open(output_dir / "generations.json", "w") as f:
        json.dump(generations, f, indent=2, ensure_ascii=False)

    print("\n--- Running Judge Evaluation ---")

    # L_opt vs L_minus
    print(f"Comparing L_opt ({L_OPT}) vs L_minus ({L_MINUS})...")
    opt_vs_minus = []
    for i, prompt in enumerate(tqdm(prompts, desc="L_opt vs L_minus")):
        judgment = judge_comparison(
            prompt=prompt,
            response_a=responses_opt[i],
            response_b=responses_minus[i],
            persona_profile=persona_profile,
            comparison_name="opt_vs_minus"
        )
        opt_vs_minus.append({
            "prompt": prompt,
            "response_L_opt": responses_opt[i],
            "response_L_minus": responses_minus[i],
            "winner": judgment["winner"],
            "reasoning": judgment["reasoning"]
        })

    # L_opt vs L_plus
    print(f"Comparing L_opt ({L_OPT}) vs L_plus ({L_PLUS})...")
    opt_vs_plus = []
    for i, prompt in enumerate(tqdm(prompts, desc="L_opt vs L_plus")):
        judgment = judge_comparison(
            prompt=prompt,
            response_a=responses_opt[i],
            response_b=responses_plus[i],
            persona_profile=persona_profile,
            comparison_name="opt_vs_plus"
        )
        opt_vs_plus.append({
            "prompt": prompt,
            "response_L_opt": responses_opt[i],
            "response_L_plus": responses_plus[i],
            "winner": judgment["winner"],
            "reasoning": judgment["reasoning"]
        })

    # Save judgments
    with open(output_dir / "layer_opt_vs_minus5.json", "w") as f:
        json.dump(opt_vs_minus, f, indent=2, ensure_ascii=False)

    with open(output_dir / "layer_opt_vs_plus5.json", "w") as f:
        json.dump(opt_vs_plus, f, indent=2, ensure_ascii=False)

    # Compute statistics
    def count_winners(judgments):
        counts = {"A": 0, "B": 0, "tie": 0}
        for j in judgments:
            counts[j["winner"]] += 1
        return counts

    stats_minus = count_winners(opt_vs_minus)
    stats_plus = count_winners(opt_vs_plus)

    summary = {
        "persona_id": persona_id,
        "weights": weights.tolist(),
        "l2_norm": float(np.linalg.norm(weights)),
        "num_prompts": NUM_PROMPTS,
        "layers": {
            "L_opt": L_OPT,
            "L_minus": L_MINUS,
            "L_plus": L_PLUS
        },
        "L_opt_vs_L_minus": {
            "L_opt_wins": stats_minus["A"],
            "L_minus_wins": stats_minus["B"],
            "ties": stats_minus["tie"],
            "L_opt_win_rate": stats_minus["A"] / NUM_PROMPTS
        },
        "L_opt_vs_L_plus": {
            "L_opt_wins": stats_plus["A"],
            "L_plus_wins": stats_plus["B"],
            "ties": stats_plus["tie"],
            "L_opt_win_rate": stats_plus["A"] / NUM_PROMPTS
        }
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*80}")
    print(f"Results for {persona_id}:")
    print(f"{'='*80}")
    print(f"L_opt vs L_minus: L_opt {stats_minus['A']}, L_minus {stats_minus['B']}, Tie {stats_minus['tie']}")
    print(f"L_opt vs L_plus:  L_opt {stats_plus['A']}, L_plus {stats_plus['B']}, Tie {stats_plus['tie']}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*80}")

    return summary


def main():
    """Run layer shift ablation for all target personas."""
    print("="*80)
    print("LAYER SHIFT ABLATION EXPERIMENT")
    print("="*80)
    print(f"Target personas: {len(TARGET_PERSONAS)}")
    print(f"Prompts per persona: {NUM_PROMPTS}")
    print(f"Layers: L_opt={L_OPT}, L_minus={L_MINUS}, L_plus={L_PLUS}")
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
