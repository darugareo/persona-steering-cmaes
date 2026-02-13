#!/usr/bin/env python3
"""
10 Personas Complete Evaluation with GPT-4o

Purpose: Re-evaluate 10 personas with GPT-4o (not mini) to improve
judge sensitivity and reduce tie rate.

Design:
- 10 personas (from personas_final_10.txt)
- 4 methods: Base, Prompt, Equal, Optimized
- 28 prompts per persona (standard evaluation set)
- Judge: GPT-4o (NOT gpt-4o-mini)
- Comparisons: Base vs Prompt, Base vs Equal, Base vs Optimized
- Output: results/10personas_gpt4o/persona_id/*.json
"""

import sys
import json
import numpy as np
import torch
from pathlib import Path
from openai import OpenAI
import os
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from persona_opt.internal_steering_l3 import Llama3ActivationSteerer
from persona_opt.eval_prompts import get_eval_prompts

# Constants
LAYER = 20
ALPHA = 2.0
NUM_PROMPTS = 28
OUTPUT_DIR = Path("results/10personas_gpt4o")
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
PERSONAS_FILE = "data/personas_final_10.txt"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def load_personas_list():
    """Load list of 10 target personas."""
    with open(PERSONAS_FILE) as f:
        personas = [line.strip() for line in f if line.strip()]
    return personas


def load_optimized_weights(persona_id):
    """Load optimized weights from optimization results."""
    weights_file = Path(f"optimization_results/{persona_id}/best_weights.json")
    with open(weights_file) as f:
        data = json.load(f)
    return np.array(data["weights"])


def load_trait_vectors(layer):
    """Load SVD trait vectors for specified layer."""
    trait_vectors = {}
    for i, trait in enumerate(["R1", "R2", "R3", "R4", "R5"]):
        vector_path = f"data/steering_vectors_v2/{trait}/layer{layer}_svd.pt"
        trait_vectors[trait] = torch.load(vector_path, weights_only=True)
    return trait_vectors


def build_steering_vector(trait_vectors, weights):
    """Build steering vector as weighted sum of trait vectors."""
    steering_vec = torch.zeros_like(trait_vectors["R1"])
    for i, trait in enumerate(["R1", "R2", "R3", "R4", "R5"]):
        steering_vec += weights[i] * trait_vectors[trait]
    return steering_vec


def load_persona_profile_for_prompt(persona_id):
    """Load persona profile for prompt engineering method."""
    profile_file = Path(f"data/all_persona_profiles.json")
    with open(profile_file) as f:
        profiles = json.load(f)

    if persona_id in profiles:
        return profiles[persona_id]
    else:
        # Fallback to persona_profile.txt
        txt_file = Path(f"personas/{persona_id}/persona_profile.txt")
        with open(txt_file) as f:
            return f.read().strip()


def load_persona_profile(persona_id):
    """Load persona profile for judge."""
    profile_file = Path(f"personas/{persona_id}/persona_profile.txt")
    with open(profile_file) as f:
        return f.read().strip()


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


def generate_prompt_responses(model, prompts, persona_profile):
    """Generate responses with prompt engineering."""
    responses = []
    for prompt in tqdm(prompts, desc="Prompt"):
        augmented_prompt = f"[Style: {persona_profile}]\n\n{prompt}"
        response = model.generate(
            prompt=augmented_prompt,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True,
            top_p=0.9
        )
        responses.append(response.strip())
    return responses


def generate_steering_responses(model, prompts, trait_vectors, weights, alpha, method_name):
    """Generate responses with steering."""
    steering_vec = build_steering_vector(trait_vectors, weights)

    model.register_hooks(
        steering_vector=steering_vec,
        alpha=alpha,
        layers=[model.layer]
    )

    responses = []
    for prompt in tqdm(prompts, desc=method_name):
        response = model.generate(
            prompt=prompt,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True,
            top_p=0.9
        )
        responses.append(response.strip())

    model.remove_hooks()
    return responses


def judge_comparison(prompt, response_a, response_b, persona_profile, comparison_name):
    """
    Judge which response better matches the persona using GPT-4o.

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

Be decisive and avoid ties unless responses are truly indistinguishable.

**Output Format (JSON only):**
{{
  "winner": "A" or "B" or "tie",
  "reasoning": "Brief 1-2 sentence explanation"
}}

Respond with JSON only, no additional text."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # Explicitly use GPT-4o, not mini
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0.0,
            max_tokens=250
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
    """Run complete evaluation for one persona."""
    print(f"\n{'='*80}")
    print(f"Persona: {persona_id}")
    print(f"{'='*80}")

    # Create output directory
    output_dir = OUTPUT_DIR / persona_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading persona data...")
    persona_profile_judge = load_persona_profile(persona_id)
    persona_profile_prompt = load_persona_profile_for_prompt(persona_id)

    print("Loading optimized weights...")
    try:
        optimized_weights = load_optimized_weights(persona_id)
        print(f"  Weights: {optimized_weights}")
    except FileNotFoundError:
        print(f"  WARNING: No optimized weights found for {persona_id}, skipping...")
        return None

    print(f"Loading trait vectors (layer {LAYER})...")
    trait_vectors = load_trait_vectors(LAYER)

    # Get prompts
    print(f"Loading {NUM_PROMPTS} evaluation prompts...")
    all_prompts = get_eval_prompts()
    prompts = all_prompts[:NUM_PROMPTS]

    # Initialize model
    print(f"Loading model: {MODEL_NAME}...")
    model = Llama3ActivationSteerer(
        model_name=MODEL_NAME,
        layer=LAYER,
        torch_dtype=torch.bfloat16
    )

    # Generate responses for all methods
    print("\n--- Generating responses ---")

    print("Method: Base")
    base_responses = generate_base_responses(model, prompts)

    print("Method: Prompt")
    prompt_responses = generate_prompt_responses(model, prompts, persona_profile_prompt)

    print("Method: Equal")
    equal_weights = np.ones(5)
    equal_responses = generate_steering_responses(
        model, prompts, trait_vectors, equal_weights, ALPHA, "Equal"
    )

    print("Method: Optimized")
    optimized_responses = generate_steering_responses(
        model, prompts, trait_vectors, optimized_weights, ALPHA, "Optimized"
    )

    # Save generations
    generations = {
        "persona_id": persona_id,
        "layer": LAYER,
        "alpha": ALPHA,
        "num_prompts": NUM_PROMPTS,
        "optimized_weights": optimized_weights.tolist(),
        "equal_weights": equal_weights.tolist(),
        "prompts": prompts,
        "base_responses": base_responses,
        "prompt_responses": prompt_responses,
        "equal_responses": equal_responses,
        "optimized_responses": optimized_responses
    }

    with open(output_dir / "generations.json", "w") as f:
        json.dump(generations, f, indent=2, ensure_ascii=False)

    print("\n--- Running Judge Evaluation (GPT-4o) ---")

    # Base vs Prompt
    print("Comparing Base vs Prompt...")
    base_vs_prompt = []
    for i, prompt in enumerate(tqdm(prompts, desc="Base vs Prompt")):
        judgment = judge_comparison(
            prompt=prompt,
            response_a=base_responses[i],
            response_b=prompt_responses[i],
            persona_profile=persona_profile_judge,
            comparison_name="base_vs_prompt"
        )
        base_vs_prompt.append({
            "prompt": prompt,
            "response_base": base_responses[i],
            "response_prompt": prompt_responses[i],
            "winner": judgment["winner"],
            "reasoning": judgment["reasoning"]
        })

    # Base vs Equal
    print("Comparing Base vs Equal...")
    base_vs_equal = []
    for i, prompt in enumerate(tqdm(prompts, desc="Base vs Equal")):
        judgment = judge_comparison(
            prompt=prompt,
            response_a=base_responses[i],
            response_b=equal_responses[i],
            persona_profile=persona_profile_judge,
            comparison_name="base_vs_equal"
        )
        base_vs_equal.append({
            "prompt": prompt,
            "response_base": base_responses[i],
            "response_equal": equal_responses[i],
            "winner": judgment["winner"],
            "reasoning": judgment["reasoning"]
        })

    # Base vs Optimized
    print("Comparing Base vs Optimized...")
    base_vs_optimized = []
    for i, prompt in enumerate(tqdm(prompts, desc="Base vs Optimized")):
        judgment = judge_comparison(
            prompt=prompt,
            response_a=base_responses[i],
            response_b=optimized_responses[i],
            persona_profile=persona_profile_judge,
            comparison_name="base_vs_optimized"
        )
        base_vs_optimized.append({
            "prompt": prompt,
            "response_base": base_responses[i],
            "response_optimized": optimized_responses[i],
            "winner": judgment["winner"],
            "reasoning": judgment["reasoning"]
        })

    # Save judgments
    with open(output_dir / "base_vs_prompt.json", "w") as f:
        json.dump(base_vs_prompt, f, indent=2, ensure_ascii=False)

    with open(output_dir / "base_vs_equal.json", "w") as f:
        json.dump(base_vs_equal, f, indent=2, ensure_ascii=False)

    with open(output_dir / "base_vs_optimized.json", "w") as f:
        json.dump(base_vs_optimized, f, indent=2, ensure_ascii=False)

    # Compute statistics
    def count_winners(judgments):
        counts = {"A": 0, "B": 0, "tie": 0}
        for j in judgments:
            counts[j["winner"]] += 1
        return counts

    stats_prompt = count_winners(base_vs_prompt)
    stats_equal = count_winners(base_vs_equal)
    stats_opt = count_winners(base_vs_optimized)

    summary = {
        "persona_id": persona_id,
        "optimized_weights": optimized_weights.tolist(),
        "num_prompts": NUM_PROMPTS,
        "judge_model": "gpt-4o",
        "base_vs_prompt": {
            "base_wins": stats_prompt["A"],
            "prompt_wins": stats_prompt["B"],
            "ties": stats_prompt["tie"],
            "tie_rate": stats_prompt["tie"] / NUM_PROMPTS
        },
        "base_vs_equal": {
            "base_wins": stats_equal["A"],
            "equal_wins": stats_equal["B"],
            "ties": stats_equal["tie"],
            "tie_rate": stats_equal["tie"] / NUM_PROMPTS
        },
        "base_vs_optimized": {
            "base_wins": stats_opt["A"],
            "optimized_wins": stats_opt["B"],
            "ties": stats_opt["tie"],
            "tie_rate": stats_opt["tie"] / NUM_PROMPTS
        }
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*80}")
    print(f"Results for {persona_id}:")
    print(f"{'='*80}")
    print(f"Base vs Prompt:    Base {stats_prompt['A']}, Prompt {stats_prompt['B']}, Tie {stats_prompt['tie']} (tie rate: {stats_prompt['tie']/NUM_PROMPTS:.1%})")
    print(f"Base vs Equal:     Base {stats_equal['A']}, Equal {stats_equal['B']}, Tie {stats_equal['tie']} (tie rate: {stats_equal['tie']/NUM_PROMPTS:.1%})")
    print(f"Base vs Optimized: Base {stats_opt['A']}, Optimized {stats_opt['B']}, Tie {stats_opt['tie']} (tie rate: {stats_opt['tie']/NUM_PROMPTS:.1%})")
    print(f"Output directory: {output_dir}")
    print(f"{'='*80}")

    return summary


def main():
    """Run complete evaluation for all 10 personas."""
    print("="*80)
    print("10 PERSONAS COMPLETE EVALUATION (GPT-4o)")
    print("="*80)

    # Load personas list
    personas = load_personas_list()
    print(f"Target personas: {len(personas)}")
    for p in personas:
        print(f"  - {p}")

    print(f"\nPrompts per persona: {NUM_PROMPTS}")
    print(f"Layer: {LAYER}")
    print(f"Alpha: {ALPHA}")
    print(f"Judge model: GPT-4o")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*80)

    all_summaries = []

    for persona_id in personas:
        try:
            summary = run_persona_experiment(persona_id)
            if summary:
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

    # Compute aggregate statistics
    if all_summaries:
        print("\n" + "="*80)
        print("AGGREGATE STATISTICS")
        print("="*80)

        # Average tie rates
        avg_tie_prompt = np.mean([s["base_vs_prompt"]["tie_rate"] for s in all_summaries])
        avg_tie_equal = np.mean([s["base_vs_equal"]["tie_rate"] for s in all_summaries])
        avg_tie_opt = np.mean([s["base_vs_optimized"]["tie_rate"] for s in all_summaries])

        print(f"Average tie rate:")
        print(f"  Base vs Prompt:    {avg_tie_prompt:.1%}")
        print(f"  Base vs Equal:     {avg_tie_equal:.1%}")
        print(f"  Base vs Optimized: {avg_tie_opt:.1%}")

        # Win rates (excluding ties)
        def compute_win_rate(summaries, comparison):
            total_wins_b = sum(s[comparison]["B_wins" if "B_wins" in s[comparison] else list(s[comparison].keys())[1].split("_wins")[0]+"_wins"] for s in summaries)
            total_wins_a = sum(s[comparison][list(s[comparison].keys())[0]] for s in summaries)
            total = total_wins_a + total_wins_b
            return total_wins_b / total if total > 0 else 0

        print(f"\nWin rates (method vs base, excluding ties):")
        print(f"  Prompt:    {compute_win_rate(all_summaries, 'base_vs_prompt'):.1%}")
        print(f"  Equal:     {compute_win_rate(all_summaries, 'base_vs_equal'):.1%}")
        print(f"  Optimized: {compute_win_rate(all_summaries, 'base_vs_optimized'):.1%}")

    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)
    print(f"Processed {len(all_summaries)}/{len(personas)} personas")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("="*80)


if __name__ == "__main__":
    main()
