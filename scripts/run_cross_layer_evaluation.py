"""
Cross-layer evaluation for all baseline methods.
Evaluates each method at layers 20, 21, 22, 23, 24.
"""

import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from persona_opt.steerer import PersonaSteerer
from persona_opt.baselines.mean_diff import MeanDiffMethod
from persona_opt.baselines.pca_method import PCAMethod
from persona_opt.baselines.random_search import RandomSearchMethod
from persona_opt.baselines.grid_search import GridSearchMethod
from persona_judge.judge import PersonaFitJudge

def run_layer_evaluation(
    persona_id: str,
    prompts_file: str,
    method_name: str,
    layer: int,
    num_prompts: int = 20,
    seed: int = 1
):
    """Run evaluation for a specific method at a specific layer."""

    print(f"\n{'='*60}")
    print(f"Method: {method_name}, Layer: {layer}")
    print(f"{'='*60}")

    # Initialize steerer
    steerer = PersonaSteerer(model_name="meta-llama/Meta-Llama-3-8B-Instruct")

    # Load prompts
    with open(prompts_file, 'r') as f:
        prompts = json.load(f)

    np.random.seed(seed)
    if len(prompts) > num_prompts:
        prompts = np.random.choice(prompts, num_prompts, replace=False).tolist()

    # Prepare method-specific paths
    base_dir = Path(__file__).parent.parent
    persona_dir = base_dir / "persona-opt" / persona_id

    # Get steering vectors based on method
    if method_name == "Base":
        # No steering
        responses = []
        for prompt in prompts:
            response = steerer.generate_with_steering(
                prompt=prompt,
                vectors={},
                layer=layer
            )
            responses.append(response)

    elif method_name == "Prompt_Persona":
        # Load persona description
        profile_file = persona_dir / "persona_profile.json"
        with open(profile_file, 'r') as f:
            profile = json.load(f)

        persona_desc = profile.get('persona_description', '')

        responses = []
        for prompt in prompts:
            augmented_prompt = f"{persona_desc}\n\n{prompt}"
            response = steerer.generate_with_steering(
                prompt=augmented_prompt,
                vectors={},
                layer=layer
            )
            responses.append(response)

    elif method_name == "MeanDiff":
        vectors_file = persona_dir / "meandiff_vectors.pt"
        if not vectors_file.exists():
            print(f"Error: {vectors_file} not found!")
            return None

        import torch
        vector_data = torch.load(vectors_file, map_location='cpu')
        vectors = {f"R{i+1}": vector_data[f'R{i+1}'] for i in range(5)}

        responses = []
        for prompt in prompts:
            response = steerer.generate_with_steering(
                prompt=prompt,
                vectors=vectors,
                layer=layer
            )
            responses.append(response)

    elif method_name == "PCA":
        vectors_file = persona_dir / "pca_vectors.pt"
        if not vectors_file.exists():
            print(f"Error: {vectors_file} not found!")
            return None

        import torch
        vector_data = torch.load(vectors_file, map_location='cpu')
        vectors = {f"R{i+1}": vector_data[f'R{i+1}'] for i in range(5)}

        responses = []
        for prompt in prompts:
            response = steerer.generate_with_steering(
                prompt=prompt,
                vectors=vectors,
                layer=layer
            )
            responses.append(response)

    elif method_name == "Random_Search":
        result_file = persona_dir / "random_search_result.json"
        vectors_file = persona_dir / "random_search_vectors.pt"

        if not vectors_file.exists():
            print(f"Error: {vectors_file} not found!")
            return None

        import torch
        vector_data = torch.load(vectors_file, map_location='cpu')
        vectors = {f"R{i+1}": vector_data['vectors'][f'R{i+1}'] for i in range(5)}

        responses = []
        for prompt in prompts:
            response = steerer.generate_with_steering(
                prompt=prompt,
                vectors=vectors,
                layer=layer
            )
            responses.append(response)

    elif method_name == "Grid_Search":
        result_file = persona_dir / "grid_search_result.json"
        vectors_file = persona_dir / "grid_search_vectors.pt"

        if not vectors_file.exists():
            print(f"Error: {vectors_file} not found!")
            return None

        import torch
        vector_data = torch.load(vectors_file, map_location='cpu')
        vectors = {f"R{i+1}": vector_data['vectors'][f'R{i+1}'] for i in range(5)}

        responses = []
        for prompt in prompts:
            response = steerer.generate_with_steering(
                prompt=prompt,
                vectors=vectors,
                layer=layer
            )
            responses.append(response)

    elif method_name == "Proposed":
        # SVD + CMA-ES
        vectors_file = persona_dir / "optimized_vectors.pt"
        if not vectors_file.exists():
            print(f"Error: {vectors_file} not found!")
            return None

        import torch
        vector_data = torch.load(vectors_file, map_location='cpu')
        vectors = {f"R{i+1}": vector_data[f'R{i+1}'] for i in range(5)}

        responses = []
        for prompt in prompts:
            response = steerer.generate_with_steering(
                prompt=prompt,
                vectors=vectors,
                layer=layer
            )
            responses.append(response)

    else:
        print(f"Error: Unknown method {method_name}")
        return None

    # Evaluate with judge
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
        'method': method_name,
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
    parser.add_argument('--num-prompts', type=int, default=20)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--methods', type=str, nargs='+',
                        default=['Base', 'Prompt_Persona', 'MeanDiff', 'PCA',
                                'Random_Search', 'Grid_Search', 'Proposed'])
    parser.add_argument('--layers', type=int, nargs='+',
                        default=[20, 21, 22, 23, 24])

    args = parser.parse_args()

    # Setup output directory
    base_dir = Path(__file__).parent.parent
    results_dir = base_dir / "reports" / "experiments" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Run evaluations
    all_results = []

    for method in args.methods:
        for layer in args.layers:
            result = run_layer_evaluation(
                persona_id=args.persona_id,
                prompts_file=args.prompts_file,
                method_name=method,
                layer=layer,
                num_prompts=args.num_prompts,
                seed=args.seed
            )

            if result is not None:
                all_results.append(result)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = results_dir / f"cross_layer_eval_seed{args.seed}_{timestamp}.json"

    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✓ Results saved to: {output_file}")

    # Print summary
    print("\n=== Summary ===")
    for method in args.methods:
        method_results = [r for r in all_results if r['method'] == method]
        if method_results:
            print(f"\n{method}:")
            for r in method_results:
                print(f"  Layer {r['layer']}: {r['mean_score']:.3f}")

if __name__ == "__main__":
    main()
