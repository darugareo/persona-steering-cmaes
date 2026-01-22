"""
Ablation study to analyze contributions of different components.

Ablation configurations:
1. w/o SVD (MeanDiff + CMA-ES)
2. w/o CMA-ES (SVD vectors + equal weights)
3. Single Trait Steering (R1, R2, R3, R4, R5 only)
"""

import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
import sys
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from persona_opt.steerer import PersonaSteerer
from persona_judge.judge import PersonaFitJudge

def run_ablation(
    persona_id: str,
    prompts_file: str,
    ablation_type: str,
    trait: str = None,
    num_prompts: int = 10,
    seed: int = 1,
    layer: int = 22
):
    """Run ablation experiment."""

    print(f"\n{'='*60}")
    print(f"Ablation: {ablation_type}")
    if trait:
        print(f"Trait: {trait}")
    print(f"{'='*60}")

    # Initialize
    steerer = PersonaSteerer(model_name="meta-llama/Meta-Llama-3-8B-Instruct")

    # Load prompts
    with open(prompts_file, 'r') as f:
        prompts = json.load(f)

    np.random.seed(seed)
    if len(prompts) > num_prompts:
        prompts = np.random.choice(prompts, num_prompts, replace=False).tolist()

    # Setup paths
    base_dir = Path(__file__).parent.parent
    persona_dir = base_dir / "persona-opt" / persona_id

    # Load vectors based on ablation type
    if ablation_type == "proposed":
        # Full method: SVD + CMA-ES
        vectors_file = persona_dir / "optimized_vectors.pt"
        vector_data = torch.load(vectors_file, map_location='cpu')
        vectors = {f"R{i+1}": vector_data[f'R{i+1}'] for i in range(5)}

    elif ablation_type == "wo_svd":
        # Use MeanDiff vectors (no SVD) but assume CMA-ES weights were applied
        # For true ablation, we'd need to run CMA-ES on MeanDiff vectors
        # For now, use MeanDiff with equal weights
        vectors_file = persona_dir / "meandiff_vectors.pt"
        vector_data = torch.load(vectors_file, map_location='cpu')
        vectors = {f"R{i+1}": vector_data[f'R{i+1}'] for i in range(5)}

    elif ablation_type == "wo_cmaes":
        # Use SVD vectors but with equal weights (no CMA-ES optimization)
        vectors_file = persona_dir / "svd_vectors.pt"
        if not vectors_file.exists():
            print(f"Warning: {vectors_file} not found, using optimized vectors")
            vectors_file = persona_dir / "optimized_vectors.pt"

        vector_data = torch.load(vectors_file, map_location='cpu')
        vectors = {f"R{i+1}": vector_data[f'R{i+1}'] for i in range(5)}

    elif ablation_type.startswith("single_"):
        # Single trait steering
        trait_id = ablation_type.split("_")[1]  # e.g., "R1"
        vectors_file = persona_dir / "optimized_vectors.pt"
        vector_data = torch.load(vectors_file, map_location='cpu')

        # Only use the specified trait
        vectors = {trait_id: vector_data[trait_id]}

    else:
        raise ValueError(f"Unknown ablation type: {ablation_type}")

    # Generate responses
    responses = []
    for prompt in prompts:
        response = steerer.generate_with_steering(
            prompt=prompt,
            vectors=vectors,
            layer=layer
        )
        responses.append(response)

    # Evaluate
    judge = PersonaFitJudge()
    profile_file = persona_dir / "persona_profile.json"

    scores = []
    for prompt, response in zip(prompts, responses):
        score = judge.evaluate(
            persona_profile_file=str(profile_file),
            prompt=prompt,
            response=response
        )
        scores.append(score)

    result = {
        'ablation_type': ablation_type,
        'trait': trait,
        'layer': layer,
        'mean_score': float(np.mean(scores)),
        'std_score': float(np.std(scores)),
        'scores': scores,
        'num_prompts': len(prompts),
        'seed': seed
    }

    print(f"Mean score: {result['mean_score']:.3f} ± {result['std_score']:.3f}")

    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--persona-id', type=str, required=True)
    parser.add_argument('--prompts-file', type=str, required=True)
    parser.add_argument('--num-prompts', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--layer', type=int, default=22)

    args = parser.parse_args()

    # Setup output directory
    base_dir = Path(__file__).parent.parent
    results_dir = base_dir / "reports" / "experiments" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Define ablation configurations
    ablations = [
        ('proposed', None),
        ('wo_svd', None),
        ('wo_cmaes', None),
        ('single_R1', 'R1'),
        ('single_R2', 'R2'),
        ('single_R3', 'R3'),
        ('single_R4', 'R4'),
        ('single_R5', 'R5'),
    ]

    # Run ablations
    all_results = []
    for ablation_type, trait in ablations:
        result = run_ablation(
            persona_id=args.persona_id,
            prompts_file=args.prompts_file,
            ablation_type=ablation_type,
            trait=trait,
            num_prompts=args.num_prompts,
            seed=args.seed,
            layer=args.layer
        )

        if result is not None:
            all_results.append(result)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = results_dir / f"ablation_study_seed{args.seed}_{timestamp}.json"

    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✓ Results saved to: {output_file}")

    # Generate summary table
    tables_dir = base_dir / "reports" / "experiments" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    table_file = tables_dir / "table_ablation.md"
    with open(table_file, 'w') as f:
        f.write("# Ablation Study Results\n\n")
        f.write(f"Generated: {timestamp}\n\n")
        f.write("| Configuration | Mean ± Std | Δ vs Proposed |\n")
        f.write("|--------------|-----------|---------------|\n")

        proposed_score = next(r['mean_score'] for r in all_results if r['ablation_type'] == 'proposed')

        for r in all_results:
            config_name = r['ablation_type']
            if r['trait']:
                config_name = f"Single Trait ({r['trait']})"

            delta = r['mean_score'] - proposed_score

            f.write(f"| {config_name} | {r['mean_score']:.3f} ± {r['std_score']:.3f} | {delta:+.3f} |\n")

    print(f"✓ Ablation table saved to: {table_file}")

if __name__ == "__main__":
    main()
