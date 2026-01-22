#!/usr/bin/env python3
"""
CLI script for generating human evaluation data.

Usage:
    python scripts/run_human_eval_template.py \
        --persona-id episode-184019_A \
        --prompts-file data/prompts/persona_eval_prompts.json \
        --num-samples 20 \
        --include-persona-sample
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from persona_opt.evaluation.human_eval import HumanEvalGenerator
from persona_opt.evaluation.utils import EvaluationConfig, load_prompts
from persona_opt.internal_steering_l3 import Llama3ActivationSteerer


def main():
    parser = argparse.ArgumentParser(description="Human Evaluation Data Generator")

    parser.add_argument('--persona-id', type=str, required=True,
                       help='Persona identifier (e.g., episode-184019_A)')
    parser.add_argument('--prompts-file', type=str, required=True,
                       help='Path to evaluation prompts JSON file')
    parser.add_argument('--num-samples', type=int, default=20,
                       help='Number of samples to generate')
    parser.add_argument('--include-persona-sample', action='store_true',
                       help='Include real persona samples (3-way comparison)')
    parser.add_argument('--model-name', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct',
                       help='Model name')
    parser.add_argument('--optimization-dir', type=str, default='persona-opt',
                       help='Optimization results directory')
    parser.add_argument('--vectors-dir', type=str, default='data/steering_vectors_v2',
                       help='Steering vectors directory')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: reports/evaluation/human_eval/{persona_id})')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')

    args = parser.parse_args()

    # Set default output dir
    if args.output_dir is None:
        args.output_dir = f'reports/evaluation/human_eval/{args.persona_id}'

    print("="*80)
    print("HUMAN EVALUATION DATA GENERATOR")
    print("="*80)
    print(f"Persona: {args.persona_id}")
    print(f"Samples: {args.num_samples}")
    print(f"Include persona sample: {args.include_persona_sample}")
    print(f"Output: {args.output_dir}")
    print("="*80)

    # Load prompts
    prompts = load_prompts(args.prompts_file)
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

    # Create generator
    human_eval_gen = HumanEvalGenerator(
        config=config,
        steerer=steerer,
        optimization_dir=args.optimization_dir,
        vectors_dir=args.vectors_dir
    )

    # Generate evaluation set
    print("\nGenerating evaluation set...")
    results = human_eval_gen.generate_evaluation_set(
        prompts=prompts,
        num_samples=args.num_samples,
        include_persona_sample=args.include_persona_sample,
        output_dir=args.output_dir,
        seed=args.seed
    )

    # Print summary
    print("\n" + "="*80)
    print("GENERATION COMPLETE")
    print("="*80)
    print(f"Generated {len(results['evaluation_items'])} evaluation items")
    print(f"Comparison type: {'3-way (with persona)' if args.include_persona_sample else '2-way'}")
    print(f"\nOutput files:")
    print(f"  - evaluation_data.json (full data)")
    print(f"  - human_evaluation.csv (for evaluators)")
    print(f"  - answer_key.csv (true labels)")
    print(f"  - instructions.md (evaluation guide)")
    print(f"\nLocation: {args.output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
