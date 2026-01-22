#!/usr/bin/env python3
"""
CLI script for multi-judge reliability evaluation.

Usage:
    python scripts/run_multi_judge.py \
        --persona-id episode-184019_A \
        --prompts-file data/prompts/persona_eval_prompts.json \
        --judges gpt-4o-mini gpt-4o claude-3-5-sonnet-20241022
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from persona_opt.evaluation.multi_judge import MultiJudgeEvaluator
from persona_opt.evaluation.utils import EvaluationConfig, load_prompts
from persona_opt.internal_steering_l3 import Llama3ActivationSteerer


def main():
    parser = argparse.ArgumentParser(description="Multi-Judge Reliability Evaluation")

    parser.add_argument('--persona-id', type=str, required=True,
                       help='Persona identifier (e.g., episode-184019_A)')
    parser.add_argument('--prompts-file', type=str, required=True,
                       help='Path to evaluation prompts JSON file')
    parser.add_argument('--num-prompts', type=int, default=10,
                       help='Number of prompts to use')
    parser.add_argument('--judges', type=str, nargs='+',
                       default=['gpt-4o-mini', 'gpt-4o', 'claude-3-5-sonnet-20241022'],
                       help='Judge models to compare')
    parser.add_argument('--model-name', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct',
                       help='Model name')
    parser.add_argument('--optimization-dir', type=str, default='persona-opt',
                       help='Optimization results directory')
    parser.add_argument('--vectors-dir', type=str, default='data/steering_vectors_v2',
                       help='Steering vectors directory')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: reports/evaluation/multi_judge/{persona_id})')

    args = parser.parse_args()

    # Set default output dir
    if args.output_dir is None:
        args.output_dir = f'reports/evaluation/multi_judge/{args.persona_id}'

    print("="*80)
    print("MULTI-JUDGE RELIABILITY EVALUATION")
    print("="*80)
    print(f"Persona: {args.persona_id}")
    print(f"Judges: {args.judges}")
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

    # Create evaluator
    multi_judge_eval = MultiJudgeEvaluator(
        config=config,
        steerer=steerer,
        optimization_dir=args.optimization_dir,
        vectors_dir=args.vectors_dir
    )

    # Run evaluation
    print("\nRunning evaluation...")
    results = multi_judge_eval.evaluate(
        prompts=prompts,
        judge_models=args.judges,
        output_dir=args.output_dir
    )

    # Print summary
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"Number of judges: {results['summary']['num_judges']}")
    print(f"Mean agreement (Spearman): {results['summary']['mean_agreement']}")
    print(f"Most lenient: {results['summary']['most_lenient']}")
    print(f"Most strict: {results['summary']['most_strict']}")
    print(f"Reliable: {results['summary']['reliable']}")
    print(f"\nResults saved to: {args.output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
