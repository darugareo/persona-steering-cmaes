"""
TruthfulQA evaluation template for Phase 2.
Evaluates persona-steered model on TruthfulQA benchmark to assess impact on factuality.
"""

import json
import argparse
from pathlib import Path
import sys
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from persona_opt.steerer import PersonaSteerer

def load_truthfulqa_dataset():
    """
    Load TruthfulQA dataset.
    TODO: Implement dataset loading (e.g., from HuggingFace datasets)
    """
    # Placeholder
    # from datasets import load_dataset
    # dataset = load_dataset("truthful_qa", "generation")
    # return dataset['validation']

    print("TODO: Implement TruthfulQA dataset loading")
    return []

def evaluate_truthfulness(response: str, reference: dict) -> dict:
    """
    Evaluate truthfulness of response.
    TODO: Implement truthfulness scoring (e.g., using GPT-4 judge)

    Returns:
        dict with keys: 'truthful', 'informative', 'overall'
    """
    # Placeholder
    return {
        'truthful': 0.0,
        'informative': 0.0,
        'overall': 0.0
    }

def run_truthfulqa_eval(
    persona_id: str,
    method: str = "proposed",
    layer: int = 22,
    num_samples: int = 100,
    seed: int = 1
):
    """
    Run TruthfulQA evaluation.

    Args:
        persona_id: Persona identifier
        method: Steering method ('base', 'proposed', etc.)
        layer: Layer to apply steering
        num_samples: Number of questions to evaluate
        seed: Random seed
    """

    print(f"\n{'='*60}")
    print(f"TruthfulQA Evaluation")
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

    # Load TruthfulQA dataset
    dataset = load_truthfulqa_dataset()

    if not dataset:
        print("Error: Failed to load TruthfulQA dataset")
        return None

    # Sample questions
    np.random.seed(seed)
    if len(dataset) > num_samples:
        indices = np.random.choice(len(dataset), num_samples, replace=False)
        dataset = [dataset[i] for i in indices]

    # Run evaluation
    results = []

    for i, item in enumerate(dataset):
        question = item['question']

        # Generate response
        if method == "base":
            response = steerer.generate_with_steering(
                prompt=question,
                vectors={},
                layer=layer
            )
        else:
            response = steerer.generate_with_steering(
                prompt=question,
                vectors=vectors,
                layer=layer
            )

        # Evaluate
        scores = evaluate_truthfulness(response, item)

        results.append({
            'question': question,
            'response': response,
            'scores': scores
        })

        if (i + 1) % 10 == 0:
            print(f"Processed {i+1}/{len(dataset)} questions")

    # Aggregate scores
    truthful_scores = [r['scores']['truthful'] for r in results]
    informative_scores = [r['scores']['informative'] for r in results]
    overall_scores = [r['scores']['overall'] for r in results]

    summary = {
        'method': method,
        'layer': layer,
        'num_samples': len(results),
        'seed': seed,
        'truthful_mean': float(np.mean(truthful_scores)),
        'truthful_std': float(np.std(truthful_scores)),
        'informative_mean': float(np.mean(informative_scores)),
        'informative_std': float(np.std(informative_scores)),
        'overall_mean': float(np.mean(overall_scores)),
        'overall_std': float(np.std(overall_scores)),
        'results': results
    }

    return summary

def main():
    parser = argparse.ArgumentParser(description="TruthfulQA evaluation")
    parser.add_argument('--persona-id', type=str, required=True)
    parser.add_argument('--method', type=str, default='proposed',
                        choices=['base', 'proposed', 'meandiff', 'pca'])
    parser.add_argument('--layer', type=int, default=22)
    parser.add_argument('--num-samples', type=int, default=100)
    parser.add_argument('--seed', type=int, default=1)

    args = parser.parse_args()

    # Run evaluation
    summary = run_truthfulqa_eval(
        persona_id=args.persona_id,
        method=args.method,
        layer=args.layer,
        num_samples=args.num_samples,
        seed=args.seed
    )

    if summary is None:
        print("Evaluation failed!")
        return

    # Save results
    base_dir = Path(__file__).parent.parent
    results_dir = base_dir / "reports" / "experiments" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    output_file = results_dir / f"truthfulqa_{args.method}_seed{args.seed}.json"
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Results saved to: {output_file}")
    print(f"\nTruthful: {summary['truthful_mean']:.3f} ± {summary['truthful_std']:.3f}")
    print(f"Informative: {summary['informative_mean']:.3f} ± {summary['informative_std']:.3f}")
    print(f"Overall: {summary['overall_mean']:.3f} ± {summary['overall_std']:.3f}")

if __name__ == "__main__":
    main()
