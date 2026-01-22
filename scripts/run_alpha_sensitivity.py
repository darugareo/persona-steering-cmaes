#!/usr/bin/env python3
"""
CLI script for alpha sensitivity evaluation.

Usage:
    python scripts/run_alpha_sensitivity.py \
        --persona-id episode-184019_A \
        --prompts-file data/prompts/persona_eval_prompts.json \
        --alpha-values 0.5 1.0 1.5 2.0 2.5 3.0
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from persona_opt.evaluation.alpha_sensitivity import AlphaSensitivityEvaluator
from persona_opt.evaluation.utils import EvaluationConfig, load_prompts, load_persona_profile
from persona_opt.internal_steering_l3 import Llama3ActivationSteerer
from persona_opt.evaluator import PersonaAwareEvaluator


def main():
    parser = argparse.ArgumentParser(description="Alpha Sensitivity Evaluation")

    parser.add_argument('--persona-id', type=str, required=True,
                       help='Persona identifier (e.g., episode-184019_A)')
    parser.add_argument('--prompts-file', type=str, required=True,
                       help='Path to evaluation prompts JSON file')
    parser.add_argument('--num-prompts', type=int, default=10,
                       help='Number of prompts to use')
    parser.add_argument('--alpha-values', type=float, nargs='+',
                       default=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
                       help='Alpha values to test (default: 0.5 1.0 1.5 2.0 2.5 3.0)')
    parser.add_argument('--model-name', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct',
                       help='Model name')
    parser.add_argument('--judge-model', type=str, default='gpt-4o-mini',
                       help='Judge model for evaluation')
    parser.add_argument('--optimization-dir', type=str, default='persona-opt',
                       help='Optimization results directory')
    parser.add_argument('--vectors-dir', type=str, default='data/steering_vectors_v2',
                       help='Steering vectors directory')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: reports/evaluation/alpha_sensitivity/{persona_id})')

    args = parser.parse_args()

    # Set default output dir
    if args.output_dir is None:
        args.output_dir = f'reports/evaluation/alpha_sensitivity/{args.persona_id}'

    print("="*80)
    print("ALPHA SENSITIVITY EVALUATION")
    print("="*80)
    print(f"Persona: {args.persona_id}")
    print(f"Alpha values: {args.alpha_values}")
    print(f"Output: {args.output_dir}")
    print("="*80)

    # Load prompts
    prompts = load_prompts(args.prompts_file, max_prompts=args.num_prompts)
    print(f"\nLoaded {len(prompts)} prompts")

    # Create config
    config = EvaluationConfig(
        persona_id=args.persona_id,
        layer=20,  # Will be overridden
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
    alpha_eval = AlphaSensitivityEvaluator(
        config=config,
        steerer=steerer,
        evaluator=evaluator,
        optimization_dir=args.optimization_dir,
        vectors_dir=args.vectors_dir
    )

    # Run evaluation
    print("\nRunning evaluation...")
    results = alpha_eval.evaluate(
        prompts=prompts,
        alpha_values=args.alpha_values,
        output_dir=args.output_dir
    )

    # Print summary
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"Optimized alpha: {results['summary']['optimized_alpha']}")
    print(f"Optimized score: {results['summary']['optimized_score']}")
    print(f"Best alpha: {results['summary']['best_alpha']}")
    print(f"Best score: {results['summary']['best_score']}")
    print(f"Score range: {results['summary']['score_range']}")
    print(f"Robust: {results['summary']['robust']}")
    print(f"\nResults saved to: {args.output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
