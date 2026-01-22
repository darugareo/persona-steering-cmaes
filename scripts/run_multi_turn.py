#!/usr/bin/env python3
"""
CLI script for multi-turn persona stability evaluation.

Usage:
    python scripts/run_multi_turn.py \
        --persona-id episode-184019_A \
        --prompts-file data/prompts/persona_eval_prompts.json \
        --num-turns 5
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from persona_opt.evaluation.multi_turn import MultiTurnEvaluator
from persona_opt.evaluation.utils import EvaluationConfig, load_prompts, load_persona_profile
from persona_opt.internal_steering_l3 import Llama3ActivationSteerer
from persona_opt.evaluator import PersonaAwareEvaluator


def main():
    parser = argparse.ArgumentParser(description="Multi-Turn Persona Stability Evaluation")

    parser.add_argument('--persona-id', type=str, required=True,
                       help='Persona identifier (e.g., episode-184019_A)')
    parser.add_argument('--prompts-file', type=str, required=True,
                       help='Path to evaluation prompts JSON file')
    parser.add_argument('--num-conversations', type=int, default=5,
                       help='Number of conversations to run')
    parser.add_argument('--num-turns', type=int, default=5,
                       help='Number of turns per conversation')
    parser.add_argument('--model-name', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct',
                       help='Model name')
    parser.add_argument('--judge-model', type=str, default='gpt-4o-mini',
                       help='Judge model for evaluation')
    parser.add_argument('--optimization-dir', type=str, default='persona-opt',
                       help='Optimization results directory')
    parser.add_argument('--vectors-dir', type=str, default='data/steering_vectors_v2',
                       help='Steering vectors directory')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: reports/evaluation/multi_turn/{persona_id})')

    args = parser.parse_args()

    # Set default output dir
    if args.output_dir is None:
        args.output_dir = f'reports/evaluation/multi_turn/{args.persona_id}'

    print("="*80)
    print("MULTI-TURN PERSONA STABILITY EVALUATION")
    print("="*80)
    print(f"Persona: {args.persona_id}")
    print(f"Conversations: {args.num_conversations}")
    print(f"Turns per conversation: {args.num_turns}")
    print(f"Output: {args.output_dir}")
    print("="*80)

    # Load prompts
    prompts = load_prompts(args.prompts_file, max_prompts=args.num_conversations)
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
    multi_turn_eval = MultiTurnEvaluator(
        config=config,
        steerer=steerer,
        evaluator=evaluator,
        optimization_dir=args.optimization_dir,
        vectors_dir=args.vectors_dir
    )

    # Run evaluation
    print("\nRunning evaluation...")
    results = multi_turn_eval.evaluate(
        initial_prompts=prompts,
        num_turns=args.num_turns,
        output_dir=args.output_dir
    )

    # Print summary
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"Number of turns: {results['summary']['num_turns']}")
    print(f"First turn score: {results['summary']['first_turn_score']}")
    print(f"Last turn score: {results['summary']['last_turn_score']}")
    print(f"Drift: {results['summary']['drift']}")
    print(f"Stable: {results['summary']['stable']}")
    print(f"\nResults saved to: {args.output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
