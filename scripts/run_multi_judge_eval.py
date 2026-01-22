"""
Multi-judge evaluation template for Phase 2.
Evaluates persona consistency using multiple judge models to assess inter-rater reliability.
"""

import json
import argparse
from pathlib import Path
import sys
import torch
import numpy as np
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

from persona_opt.steerer import PersonaSteerer
from persona_judge.judge import PersonaFitJudge

# TODO: Import additional judge implementations
# from persona_judge.gpt4_judge import GPT4Judge
# from persona_judge.claude_judge import ClaudeJudge

def get_judges() -> Dict[str, object]:
    """
    Initialize multiple judge models.

    Returns:
        Dictionary mapping judge names to judge instances
    """
    judges = {
        'llama_judge': PersonaFitJudge()  # Current default judge
    }

    # TODO: Add additional judges
    # judges['gpt4_judge'] = GPT4Judge()
    # judges['claude_judge'] = ClaudeJudge()

    return judges

def compute_inter_rater_reliability(scores_by_judge: Dict[str, List[float]]) -> Dict:
    """
    Compute inter-rater reliability metrics.

    Args:
        scores_by_judge: Dictionary mapping judge names to lists of scores

    Returns:
        Dictionary with reliability metrics (ICC, Cronbach's alpha, etc.)
    """
    # TODO: Implement proper inter-rater reliability metrics
    # For now, compute simple correlation

    judge_names = list(scores_by_judge.keys())
    if len(judge_names) < 2:
        return {'error': 'Need at least 2 judges for reliability analysis'}

    # Compute pairwise correlations
    correlations = {}
    for i, judge1 in enumerate(judge_names):
        for judge2 in judge_names[i+1:]:
            scores1 = np.array(scores_by_judge[judge1])
            scores2 = np.array(scores_by_judge[judge2])

            corr = np.corrcoef(scores1, scores2)[0, 1]
            correlations[f"{judge1}_vs_{judge2}"] = float(corr)

    # Compute mean correlation
    mean_corr = np.mean(list(correlations.values()))

    return {
        'pairwise_correlations': correlations,
        'mean_correlation': float(mean_corr)
    }

def run_multi_judge_eval(
    persona_id: str,
    prompts_file: str,
    method: str = "proposed",
    layer: int = 22,
    num_prompts: int = 20,
    seed: int = 1
):
    """
    Run multi-judge evaluation.

    Args:
        persona_id: Persona identifier
        prompts_file: Path to prompts JSON file
        method: Steering method
        layer: Layer to apply steering
        num_prompts: Number of prompts to evaluate
        seed: Random seed
    """

    print(f"\n{'='*60}")
    print(f"Multi-Judge Evaluation")
    print(f"Method: {method}, Layer: {layer}")
    print(f"{'='*60}\n")

    # Setup
    base_dir = Path(__file__).parent.parent
    persona_dir = base_dir / "persona-opt" / persona_id

    # Initialize steerer
    steerer = PersonaSteerer(model_name="meta-llama/Meta-Llama-3-8B-Instruct")

    # Load steering vectors if needed
    vectors = {}
    if method != "base":
        vectors_file = persona_dir / "optimized_vectors.pt"
        if vectors_file.exists():
            vector_data = torch.load(vectors_file, map_location='cpu')
            vectors = {f"R{i+1}": vector_data[f'R{i+1}'] for i in range(5)}
        else:
            print(f"Warning: {vectors_file} not found, using base model")

    # Load prompts
    with open(prompts_file, 'r') as f:
        prompts = json.load(f)

    np.random.seed(seed)
    if len(prompts) > num_prompts:
        prompts = np.random.choice(prompts, num_prompts, replace=False).tolist()

    # Initialize judges
    judges = get_judges()
    print(f"Using judges: {list(judges.keys())}")

    # Generate responses
    print("\nGenerating responses...")
    responses = []
    for i, prompt in enumerate(prompts):
        if method == "base":
            response = steerer.generate_with_steering(
                prompt=prompt,
                vectors={},
                layer=layer
            )
        else:
            response = steerer.generate_with_steering(
                prompt=prompt,
                vectors=vectors,
                layer=layer
            )

        responses.append(response)

        if (i + 1) % 5 == 0:
            print(f"Generated {i+1}/{len(prompts)} responses")

    # Evaluate with each judge
    profile_file = persona_dir / "persona_profile.json"
    scores_by_judge = {}

    for judge_name, judge in judges.items():
        print(f"\nEvaluating with {judge_name}...")
        scores = []

        for i, (prompt, response) in enumerate(zip(prompts, responses)):
            score = judge.evaluate(
                persona_profile_file=str(profile_file),
                prompt=prompt,
                response=response
            )
            scores.append(score)

            if (i + 1) % 5 == 0:
                print(f"  Evaluated {i+1}/{len(prompts)} responses")

        scores_by_judge[judge_name] = scores

        print(f"  {judge_name} mean: {np.mean(scores):.3f} ± {np.std(scores):.3f}")

    # Compute inter-rater reliability
    print("\nComputing inter-rater reliability...")
    reliability = compute_inter_rater_reliability(scores_by_judge)

    # Prepare results
    results = {
        'method': method,
        'layer': layer,
        'persona_id': persona_id,
        'num_prompts': len(prompts),
        'seed': seed,
        'scores_by_judge': {
            judge_name: {
                'scores': scores,
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores))
            }
            for judge_name, scores in scores_by_judge.items()
        },
        'reliability': reliability,
        'prompts': prompts,
        'responses': responses
    }

    return results

def main():
    parser = argparse.ArgumentParser(description="Multi-judge evaluation")
    parser.add_argument('--persona-id', type=str, required=True)
    parser.add_argument('--prompts-file', type=str, required=True)
    parser.add_argument('--method', type=str, default='proposed',
                        choices=['base', 'proposed', 'meandiff', 'pca'])
    parser.add_argument('--layer', type=int, default=22)
    parser.add_argument('--num-prompts', type=int, default=20)
    parser.add_argument('--seed', type=int, default=1)

    args = parser.parse_args()

    # Run evaluation
    results = run_multi_judge_eval(
        persona_id=args.persona_id,
        prompts_file=args.prompts_file,
        method=args.method,
        layer=args.layer,
        num_prompts=args.num_prompts,
        seed=args.seed
    )

    if results is None:
        print("Evaluation failed!")
        return

    # Save results
    base_dir = Path(__file__).parent.parent
    results_dir = base_dir / "reports" / "experiments" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    output_file = results_dir / f"multi_judge_{args.method}_seed{args.seed}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {output_file}")

    # Print summary
    print("\n=== Summary ===")
    for judge_name, judge_data in results['scores_by_judge'].items():
        print(f"{judge_name}: {judge_data['mean']:.3f} ± {judge_data['std']:.3f}")

    if 'mean_correlation' in results['reliability']:
        print(f"\nMean inter-rater correlation: {results['reliability']['mean_correlation']:.3f}")

if __name__ == "__main__":
    main()
