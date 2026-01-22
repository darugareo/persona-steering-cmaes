#!/usr/bin/env python3
"""
CLI script for train/test split evaluation.

Usage:
    python scripts/run_train_test.py \
        --persona-id episode-184019_A \
        --prompts-file data/prompts/persona_eval_prompts.json \
        --num-prompts 10 \
        --train-ratio 0.7
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from persona_opt.evaluation.train_test import TrainTestEvaluator
from persona_opt.evaluation.utils import EvaluationConfig, load_prompts, load_persona_profile
from persona_opt.internal_steering_l3 import Llama3ActivationSteerer
from persona_opt.evaluator import PersonaAwareEvaluator


def main():
    parser = argparse.ArgumentParser(description="Train/Test Split Evaluation")

    parser.add_argument('--persona-id', type=str, required=True,
                       help='Persona identifier (e.g., episode-184019_A)')
    parser.add_argument('--prompts-file', type=str, required=True,
                       help='Path to evaluation prompts JSON file')
    parser.add_argument('--num-prompts', type=int, default=10,
                       help='Number of prompts to use')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                       help='Ratio of training prompts (default: 0.7)')
    parser.add_argument('--layer', type=int, default=None,
                       help='Layer to use (default: use optimized layer)')
    parser.add_argument('--model-name', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct',
                       help='Model name')
    parser.add_argument('--judge-model', type=str, default='gpt-4o-mini',
                       help='Judge model for evaluation')
    parser.add_argument('--optimization-dir', type=str, default='persona-opt',
                       help='Optimization results directory')
    parser.add_argument('--vectors-dir', type=str, default='data/steering_vectors_v2',
                       help='Steering vectors directory')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: reports/evaluation/train_test/{persona_id})')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')

    args = parser.parse_args()

    # Set default output dir
    if args.output_dir is None:
        args.output_dir = f'reports/evaluation/train_test/{args.persona_id}'

    print("="*80)
    print("TRAIN/TEST SPLIT EVALUATION")
    print("="*80)
    print(f"Persona: {args.persona_id}")
    print(f"Prompts file: {args.prompts_file}")
    print(f"Train ratio: {args.train_ratio}")
    print(f"Output: {args.output_dir}")
    print("="*80)

    # Load prompts
    prompts = load_prompts(args.prompts_file, max_prompts=args.num_prompts)
    print(f"\nLoaded {len(prompts)} prompts")

    # Create config
    config = EvaluationConfig(
        persona_id=args.persona_id,
        layer=args.layer if args.layer else 20,  # Will be overridden by optimized layer
        model_name=args.model_name
    )

    # Initialize steerer
    print(f"\nInitializing steerer (model: {args.model_name})...")
    steerer = Llama3ActivationSteerer(
        model_name=args.model_name,
        device=config.device
    )

    # Initialize evaluator
    print(f"Initializing evaluator (judge: {args.judge_model})...")
    persona_profile = load_persona_profile(args.persona_id)
    evaluator = PersonaAwareEvaluator(
        persona_profile=persona_profile,
        judge_model=args.judge_model
    )

    # Create evaluator
    train_test_eval = TrainTestEvaluator(
        config=config,
        steerer=steerer,
        evaluator=evaluator,
        optimization_dir=args.optimization_dir,
        vectors_dir=args.vectors_dir
    )

    # Run evaluation
    print("\nRunning evaluation...")
    results = train_test_eval.evaluate(
        prompts=prompts,
        train_ratio=args.train_ratio,
        seed=args.seed,
        output_dir=args.output_dir
    )

    # Print summary
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"Train score: {results['summary']['train_mean']}")
    print(f"Test score:  {results['summary']['test_mean']}")
    print(f"Generalization gap: {results['summary']['generalization_gap']}")
    print(f"Overfitting: {results['summary']['overfitting']}")
    print(f"\nResults saved to: {args.output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
