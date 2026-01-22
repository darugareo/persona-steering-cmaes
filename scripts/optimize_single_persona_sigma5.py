#!/usr/bin/env python3
"""
σ=5.0で1ペルソナを最適化

Usage:
    python scripts/optimize_single_persona_sigma5.py \
        --persona_id episode-184019_A \
        --device cuda:0 \
        --output_dir optimization_results_sigma5
"""

import json
import numpy as np
from pathlib import Path
import sys
import argparse
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from persona_opt.cmaes_persona_optimizer import CMAESPersonaOptimizer


def main():
    parser = argparse.ArgumentParser(description="Optimize single persona with σ=5.0")
    parser.add_argument("--persona_id", required=True, help="Persona ID")
    parser.add_argument("--device", default="cuda:0", help="Device (cuda:0, cuda:1, etc.)")
    parser.add_argument("--output_dir", default="optimization_results_sigma5", help="Output directory")
    parser.add_argument("--sigma", type=float, default=5.0, help="CMA-ES initial step size")
    parser.add_argument("--alpha", type=float, default=2.0, help="Steering strength")
    parser.add_argument("--max_iterations", type=int, default=20, help="Max CMA-ES iterations")
    parser.add_argument("--population_size", type=int, default=10, help="Population size")

    args = parser.parse_args()

    print(f"\n{'='*80}")
    print(f"Optimizing: {args.persona_id}")
    print(f"  Device: {args.device}")
    print(f"  σ (sigma): {args.sigma}")
    print(f"  α (alpha): {args.alpha}")
    print(f"{'='*80}\n")

    # Load eval prompts
    test_turns_path = Path(f"personas_cc/{args.persona_id}/test_turns_selected.json")
    if not test_turns_path.exists():
        test_turns_path = Path(f"personas_cc/{args.persona_id}/test_turns.json")

    with open(test_turns_path) as f:
        data = json.load(f)
        eval_turns = data["turns"][:10]

    eval_prompts = []
    for turn in eval_turns:
        context = turn["context"]
        input_text = turn["input"]
        prompt = f"""{context}
{input_text}

Response:"""
        eval_prompts.append(prompt)

    print(f"✓ Loaded {len(eval_prompts)} evaluation turns")

    # Initialize optimizer
    optimizer = CMAESPersonaOptimizer(
        persona_id=args.persona_id,
        layer=20,
        trait_vector_dir="data/steering_vectors_v2",
        persona_dir="personas",
        eval_prompts=eval_prompts[:5],  # Use 5 prompts for optimization
        alpha=args.alpha
    )

    # Run optimization
    output_dir = Path(args.output_dir) / args.persona_id

    start_time = datetime.now()
    result = optimizer.optimize(
        sigma0=args.sigma,
        max_iterations=args.max_iterations,
        population_size=args.population_size,
        save_dir=str(output_dir)
    )
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    # Calculate L2 norm
    weights = result["best_weights"]
    l2_norm = np.sqrt(sum(w**2 for w in weights.values()))

    print(f"\n{'='*80}")
    print(f"✅ Optimization Complete: {args.persona_id}")
    print(f"{'='*80}")
    print(f"  Best weights: {weights}")
    print(f"  L2 norm: {l2_norm:.3f}")
    print(f"  Best score: {result['best_score']:.4f}")
    print(f"  Duration: {duration/60:.1f} minutes")
    print(f"{'='*80}\n")

    # Save summary
    summary = {
        "persona_id": args.persona_id,
        "sigma": args.sigma,
        "alpha": args.alpha,
        "best_weights": weights,
        "l2_norm": float(l2_norm),
        "best_score": float(result["best_score"]),
        "iterations": result.get("iterations", args.max_iterations),
        "duration_seconds": duration,
        "timestamp": datetime.now().isoformat()
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"📁 Saved to: {output_dir}/summary.json")


if __name__ == "__main__":
    main()
