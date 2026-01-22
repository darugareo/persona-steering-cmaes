"""
MMLU evaluation template for Phase 2.
Evaluates persona-steered model on MMLU benchmark to assess impact on general knowledge.
"""

import json
import argparse
from pathlib import Path
import sys
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from persona_opt.steerer import PersonaSteerer

def load_mmlu_dataset(subjects: list = None):
    """
    Load MMLU dataset.
    TODO: Implement dataset loading (e.g., from HuggingFace datasets)

    Args:
        subjects: List of MMLU subjects to evaluate (e.g., ['abstract_algebra', 'anatomy'])
                 If None, evaluates on all subjects.
    """
    # Placeholder
    # from datasets import load_dataset
    # dataset = load_dataset("cais/mmlu", "all")
    # return dataset['test']

    print("TODO: Implement MMLU dataset loading")
    return []

def evaluate_mmlu_accuracy(response: str, correct_answer: str, choices: list) -> bool:
    """
    Evaluate MMLU accuracy.
    Extract the predicted answer from response and compare with correct answer.

    Args:
        response: Model's generated response
        correct_answer: Correct answer (e.g., 'A', 'B', 'C', 'D')
        choices: List of answer choices

    Returns:
        bool: Whether the prediction is correct
    """
    # TODO: Implement answer extraction logic
    # For now, simple placeholder
    return False

def format_mmlu_prompt(question: str, choices: list) -> str:
    """
    Format MMLU question as a prompt.

    Args:
        question: The question text
        choices: List of answer choices

    Returns:
        Formatted prompt string
    """
    choice_labels = ['A', 'B', 'C', 'D']
    prompt = f"Question: {question}\n\n"

    for label, choice in zip(choice_labels, choices):
        prompt += f"{label}. {choice}\n"

    prompt += "\nAnswer:"
    return prompt

def run_mmlu_eval(
    persona_id: str,
    method: str = "proposed",
    layer: int = 22,
    subjects: list = None,
    num_samples_per_subject: int = 50,
    seed: int = 1
):
    """
    Run MMLU evaluation.

    Args:
        persona_id: Persona identifier
        method: Steering method ('base', 'proposed', etc.)
        layer: Layer to apply steering
        subjects: List of MMLU subjects (None = all)
        num_samples_per_subject: Number of questions per subject
        seed: Random seed
    """

    print(f"\n{'='*60}")
    print(f"MMLU Evaluation")
    print(f"Method: {method}, Layer: {layer}")
    if subjects:
        print(f"Subjects: {subjects}")
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

    # Load MMLU dataset
    dataset = load_mmlu_dataset(subjects=subjects)

    if not dataset:
        print("Error: Failed to load MMLU dataset")
        return None

    # Sample questions
    np.random.seed(seed)
    if len(dataset) > num_samples_per_subject:
        indices = np.random.choice(len(dataset), num_samples_per_subject, replace=False)
        dataset = [dataset[i] for i in indices]

    # Run evaluation
    results = []
    correct_count = 0

    for i, item in enumerate(dataset):
        question = item['question']
        choices = item['choices']
        correct_answer = item['answer']

        # Format prompt
        prompt = format_mmlu_prompt(question, choices)

        # Generate response
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

        # Evaluate
        is_correct = evaluate_mmlu_accuracy(response, correct_answer, choices)

        if is_correct:
            correct_count += 1

        results.append({
            'question': question,
            'choices': choices,
            'correct_answer': correct_answer,
            'response': response,
            'is_correct': is_correct
        })

        if (i + 1) % 10 == 0:
            print(f"Processed {i+1}/{len(dataset)} questions, Accuracy: {correct_count/(i+1):.3f}")

    # Aggregate scores
    accuracy = correct_count / len(results) if results else 0.0

    summary = {
        'method': method,
        'layer': layer,
        'subjects': subjects,
        'num_samples': len(results),
        'seed': seed,
        'accuracy': float(accuracy),
        'correct_count': correct_count,
        'results': results
    }

    return summary

def main():
    parser = argparse.ArgumentParser(description="MMLU evaluation")
    parser.add_argument('--persona-id', type=str, required=True)
    parser.add_argument('--method', type=str, default='proposed',
                        choices=['base', 'proposed', 'meandiff', 'pca'])
    parser.add_argument('--layer', type=int, default=22)
    parser.add_argument('--subjects', type=str, nargs='+', default=None,
                        help='MMLU subjects to evaluate (default: all)')
    parser.add_argument('--num-samples', type=int, default=50,
                        help='Number of samples per subject')
    parser.add_argument('--seed', type=int, default=1)

    args = parser.parse_args()

    # Run evaluation
    summary = run_mmlu_eval(
        persona_id=args.persona_id,
        method=args.method,
        layer=args.layer,
        subjects=args.subjects,
        num_samples_per_subject=args.num_samples,
        seed=args.seed
    )

    if summary is None:
        print("Evaluation failed!")
        return

    # Save results
    base_dir = Path(__file__).parent.parent
    results_dir = base_dir / "reports" / "experiments" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    output_file = results_dir / f"mmlu_{args.method}_seed{args.seed}.json"
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Results saved to: {output_file}")
    print(f"\nAccuracy: {summary['accuracy']:.3f} ({summary['correct_count']}/{summary['num_samples']})")

if __name__ == "__main__":
    main()
