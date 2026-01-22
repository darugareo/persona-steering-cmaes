#!/usr/bin/env python3
"""
Test評価: 最適化されたweightsをTest dataで評価
"""
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.fitness_comparison_optimizer import FitnessComparator

def evaluate_on_test(
    persona_id: str,
    fitness_type: str,
    optimized_weights: Dict[str, float],
    device: str = "cuda:0"
) -> Dict:
    """
    Test dataで評価

    Args:
        persona_id: ペルソナID
        fitness_type: 最適化時に使用したfitness関数
        optimized_weights: 最適化されたtrait weights
        device: GPU device

    Returns:
        評価結果 (train/test両方のスコア)
    """
    print(f"\n{'='*80}")
    print(f"Evaluating: {persona_id} ({fitness_type})")
    print(f"{'='*80}")

    # Load test data
    test_path = Path(f"personas_cc/{persona_id}/test_turns.json")
    with open(test_path) as f:
        test_data = json.load(f)
        test_turns = test_data["turns"][:10]  # Use first 10 turns

    # Initialize comparator
    comparator = FitnessComparator(
        persona_id=persona_id,
        fitness_type=fitness_type,
        device=device
    )

    # Evaluate on TRAIN (for comparison)
    print(f"\n【Train評価】")
    train_scores = {
        "bertscore": [],
        "style": [],
        "judge": [],
        "combined": []
    }
    train_generations = []

    for i, turn in enumerate(comparator.train_turns):
        # Build prompt with context
        context = turn["context"]
        input_text = turn["input"]
        ground_truth = turn["ground_truth"]

        prompt = f"""Continue this conversation naturally.

Conversation so far:
{context}

Partner: {input_text}

You:"""

        generated = comparator.generate_with_steering(prompt, optimized_weights)

        # Save generation
        train_generations.append({
            "prompt": prompt,
            "ground_truth": ground_truth,
            "generated": generated
        })

        # Calculate all fitness scores
        train_scores["bertscore"].append(
            comparator.fitness_bertscore(generated, ground_truth)
        )
        train_scores["style"].append(
            comparator.fitness_style(generated, ground_truth)
        )
        train_scores["judge"].append(
            comparator.fitness_judge(generated, ground_truth)
        )
        train_scores["combined"].append(
            comparator.fitness_combined(generated, ground_truth)
        )

        if (i + 1) % 3 == 0:
            print(f"  Progress: {i+1}/{len(comparator.train_turns)}")

    # Evaluate on TEST
    print(f"\n【Test評価】")
    test_scores = {
        "bertscore": [],
        "style": [],
        "judge": [],
        "combined": []
    }
    test_generations = []

    for i, turn in enumerate(test_turns):
        # Build prompt with context
        context = turn["context"]
        input_text = turn["input"]
        ground_truth = turn["ground_truth"]

        prompt = f"""Continue this conversation naturally.

Conversation so far:
{context}

Partner: {input_text}

You:"""

        generated = comparator.generate_with_steering(prompt, optimized_weights)

        # Save generation
        test_generations.append({
            "prompt": prompt,
            "ground_truth": ground_truth,
            "generated": generated
        })

        # Calculate all fitness scores
        test_scores["bertscore"].append(
            comparator.fitness_bertscore(generated, ground_truth)
        )
        test_scores["style"].append(
            comparator.fitness_style(generated, ground_truth)
        )
        test_scores["judge"].append(
            comparator.fitness_judge(generated, ground_truth)
        )
        test_scores["combined"].append(
            comparator.fitness_combined(generated, ground_truth)
        )

        if (i + 1) % 3 == 0:
            print(f"  Progress: {i+1}/{len(test_turns)}")

    # Calculate averages
    results = {
        "persona_id": persona_id,
        "fitness_type": fitness_type,
        "optimized_weights": optimized_weights,
        "train_scores": {
            metric: {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "min": float(np.min(scores)),
                "max": float(np.max(scores))
            }
            for metric, scores in train_scores.items()
        },
        "test_scores": {
            metric: {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "min": float(np.min(scores)),
                "max": float(np.max(scores))
            }
            for metric, scores in test_scores.items()
        },
        "generalization_gap": {
            metric: float(np.mean(train_scores[metric]) - np.mean(test_scores[metric]))
            for metric in train_scores.keys()
        },
        "train_generations": train_generations,
        "test_generations": test_generations
    }

    # Print summary
    print(f"\n{'='*80}")
    print(f"Results Summary")
    print(f"{'='*80}")
    print(f"\n{'Metric':<15} {'Train Mean':<12} {'Test Mean':<12} {'Gap':<10}")
    print(f"{'-'*50}")
    for metric in ["bertscore", "style", "judge", "combined"]:
        train_mean = results["train_scores"][metric]["mean"]
        test_mean = results["test_scores"][metric]["mean"]
        gap = results["generalization_gap"][metric]
        print(f"{metric:<15} {train_mean:<12.4f} {test_mean:<12.4f} {gap:<10.4f}")

    return results

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--persona_id", required=True)
    parser.add_argument("--fitness_type", required=True)
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()

    device = f"cuda:{args.gpu_id}"

    # Load optimized weights
    weight_file = Path(f"results/fitness_comparison/optimization_logs/{args.persona_id}_{args.fitness_type}.json")
    with open(weight_file) as f:
        opt_result = json.load(f)
        optimized_weights = opt_result["best_weights"]

    print(f"\nLoaded optimized weights from: {weight_file}")
    print(f"Best weights: {optimized_weights}")

    # Evaluate
    results = evaluate_on_test(
        persona_id=args.persona_id,
        fitness_type=args.fitness_type,
        optimized_weights=optimized_weights,
        device=device
    )

    # Save results
    output_dir = Path("results/fitness_comparison/test_evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{args.persona_id}_{args.fitness_type}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Evaluation complete!")
    print(f"  Saved to: {output_path}")

if __name__ == "__main__":
    main()
