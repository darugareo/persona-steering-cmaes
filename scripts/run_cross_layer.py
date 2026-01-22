#!/usr/bin/env python3
"""
CLI script for cross-layer transfer evaluation.

Usage:
    python scripts/run_cross_layer.py \
        --persona-id episode-184019_A \
        --prompts-file data/prompts/persona_eval_prompts.json \
        --layers 20 21 22 23 24
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from persona_opt.evaluation.cross_layer import CrossLayerEvaluator
from persona_opt.evaluation.utils import EvaluationConfig, load_prompts, load_persona_profile
from persona_opt.internal_steering_l3 import Llama3ActivationSteerer
from persona_opt.evaluator import PersonaAwareEvaluator


def main():
    parser = argparse.ArgumentParser(description="Cross-Layer Transfer Evaluation")

    parser.add_argument('--persona-id', type=str, required=True,
                       help='Persona identifier (e.g., episode-184019_A)')
    parser.add_argument('--prompts-file', type=str, required=True,
                       help='Path to evaluation prompts JSON file')
    parser.add_argument('--num-prompts', type=int, default=10,
                       help='Number of prompts to use')
    parser.add_argument('--layers', type=int, nargs='+', default=[20, 21, 22, 23, 24],
                       help='Layers to test (default: 20 21 22 23 24)')
    parser.add_argument('--model-name', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct',
                       help='Model name')
    parser.add_argument('--judge-model', type=str, default='gpt-4o-mini',
                       help='Judge model for evaluation')
    parser.add_argument('--optimization-dir', type=str, default='persona-opt',
                       help='Optimization results directory')
    parser.add_argument('--vectors-dir', type=str, default='data/steering_vectors_v2',
                       help='Steering vectors directory')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: reports/evaluation/cross_layer/{persona_id})')

    args = parser.parse_args()

    # Set default output dir
    if args.output_dir is None:
        args.output_dir = f'reports/evaluation/cross_layer/{args.persona_id}'

    print("="*80)
    print("CROSS-LAYER TRANSFER EVALUATION")
    print("="*80)
    print(f"Persona: {args.persona_id}")
    print(f"Test layers: {args.layers}")
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
    cross_layer_eval = CrossLayerEvaluator(
        config=config,
        steerer=steerer,
        evaluator=evaluator,
        optimization_dir=args.optimization_dir,
        vectors_dir=args.vectors_dir
    )

    # Run evaluation
    print("\nRunning evaluation...")
    results = cross_layer_eval.evaluate(
        prompts=prompts,
        test_layers=args.layers,
        output_dir=args.output_dir
    )

    # Print summary
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"Optimized layer: {results['summary']['optimized_layer']}")
    print(f"Optimized score: {results['summary']['optimized_score']}")
    print(f"Best layer: {results['summary']['best_layer']}")
    print(f"Best score: {results['summary']['best_score']}")
    print(f"Transferable: {results['summary']['transferable']}")
    print(f"\nResults saved to: {args.output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
