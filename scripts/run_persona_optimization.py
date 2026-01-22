"""
Run CMA-ES optimization for persona trait weights.

Usage:
    python scripts/run_persona_optimization.py \
      --persona-id episode-184019_A \
      --layer 20 \
      --max-iterations 30 \
      --num-prompts 10
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from persona_opt.cmaes_persona_optimizer import optimize_persona_weights


def main():
    parser = argparse.ArgumentParser(
        description="Optimize persona trait weights using CMA-ES"
    )

    parser.add_argument(
        "--persona-id",
        type=str,
        required=True,
        help="Persona ID (e.g., episode-184019_A)"
    )

    parser.add_argument(
        "--layer",
        type=int,
        default=20,
        help="Layer to apply steering (default: 20)"
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=2.0,
        help="Steering strength (default: 2.0)"
    )

    parser.add_argument(
        "--max-iterations",
        type=int,
        default=30,
        help="Maximum CMA-ES iterations (default: 30)"
    )

    parser.add_argument(
        "--population-size",
        type=int,
        default=None,
        help="CMA-ES population size (default: auto)"
    )

    parser.add_argument(
        "--sigma0",
        type=float,
        default=1.0,
        help="Initial CMA-ES sigma (default: 1.0)"
    )

    parser.add_argument(
        "--num-prompts",
        type=int,
        default=None,
        help="Number of evaluation prompts to use (default: all available)"
    )

    parser.add_argument(
        "--save-dir",
        type=str,
        default="optimization_results",
        help="Directory to save results (default: optimization_results)"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Model name (default: Meta-Llama-3-8B-Instruct)"
    )

    parser.add_argument(
        "--trait-vector-dir",
        type=str,
        default="data/steering_vectors_v2",
        help="Directory containing trait vectors"
    )

    parser.add_argument(
        "--persona-dir",
        type=str,
        default="personas",
        help="Directory containing persona profiles"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("CMA-ES Persona Optimization")
    print("=" * 80)
    print(f"Persona ID: {args.persona_id}")
    print(f"Model: {args.model}")
    print(f"Layer: {args.layer}")
    print(f"Alpha: {args.alpha}")
    print(f"Max iterations: {args.max_iterations}")
    if args.population_size:
        print(f"Population size: {args.population_size}")
    if args.num_prompts:
        print(f"Num prompts: {args.num_prompts}")
    print(f"Save directory: {args.save_dir}")
    print("=" * 80)

    # Load evaluation prompts if num_prompts specified
    eval_prompts = None
    if args.num_prompts:
        import json
        eval_prompts_file = Path("data/eval_prompts/persona_eval_prompts_v1.json")
        if eval_prompts_file.exists():
            with open(eval_prompts_file, 'r') as f:
                data = json.load(f)
                all_prompts = []
                for prompt_data in data["prompts"]:
                    all_prompts.append(prompt_data["text"])
                eval_prompts = all_prompts[:args.num_prompts]
                print(f"Loaded {len(eval_prompts)} evaluation prompts")

    # Run optimization
    try:
        results = optimize_persona_weights(
            persona_id=args.persona_id,
            model_name=args.model,
            layer=args.layer,
            trait_vector_dir=args.trait_vector_dir,
            persona_dir=args.persona_dir,
            eval_prompts=eval_prompts,
            alpha=args.alpha,
            max_iterations=args.max_iterations,
            save_dir=args.save_dir,
            population_size=args.population_size,
            sigma0=args.sigma0
        )

        print("\n✅ Optimization completed successfully!")
        print(f"\nBest weights:")
        for trait, weight in results["best_weights"].items():
            print(f"  {trait}: {weight:.4f}")
        print(f"\nBest persona fit score: {results['best_score']:.4f}")
        print(f"Total evaluations: {results['num_evaluations']}")

    except Exception as e:
        print(f"\n❌ Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
