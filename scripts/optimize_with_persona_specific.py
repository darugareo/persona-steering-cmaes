#!/usr/bin/env python3
"""
ペルソナ特有ターンで再最適化

設定: σ=5.0, α=2.0
評価: train_turns_persona_specific.json を使用
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from persona_opt.cmaes_persona_optimizer import CMAESPersonaOptimizer


def main():
    parser = argparse.ArgumentParser(
        description="Optimize with persona-specific turns only"
    )
    parser.add_argument("--persona_id", required=True, help="Persona ID")
    parser.add_argument("--device", default="cuda:0", help="Device")
    parser.add_argument("--output_dir", default="optimization_results_persona_specific",
                        help="Output directory")
    parser.add_argument("--sigma", type=float, default=5.0, help="CMA-ES sigma")
    parser.add_argument("--alpha", type=float, default=2.0, help="Steering alpha")
    parser.add_argument("--max_iterations", type=int, default=50, help="Max iterations")
    parser.add_argument("--population_size", type=int, default=8, help="Population size")

    args = parser.parse_args()

    print(f"\n{'='*80}")
    print(f"ペルソナ特有ターンでの最適化: {args.persona_id}")
    print(f"{'='*80}")
    print(f"  σ (sigma): {args.sigma}")
    print(f"  α (alpha): {args.alpha}")
    print(f"  Max iterations: {args.max_iterations}")
    print(f"  Population size: {args.population_size}")
    print(f"{'='*80}\n")

    # ペルソナ特有ターンの読み込み
    train_file = Path(f"personas_cc/{args.persona_id}/train_turns_persona_specific.json")

    if not train_file.exists():
        print(f"❌ Error: {train_file} not found")
        print(f"\n先にターン選定を実行してください:")
        print(f"  python scripts/select_persona_specific_turns.py")
        sys.exit(1)

    with open(train_file) as f:
        data = json.load(f)

    turns = data["turns"]
    selected_count = len(turns)

    print(f"✅ Loaded {selected_count} persona-specific turns from train set")
    print(f"   (Original: {data['total_turns']} turns, Selection rate: {data['selection_rate']*100:.1f}%)\n")

    if selected_count < 5:
        print(f"⚠️ Warning: Only {selected_count} turns available. This may not be sufficient for optimization.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)

    # eval_promptsの準備
    eval_prompts = []
    for turn in turns:
        # contextがあれば含める
        context = turn.get('context', '')
        user_input = turn['user']

        if context:
            prompt = f"Context: {context}\n\nUser: {user_input}\n\nYou:"
        else:
            prompt = f"User: {user_input}\n\nYou:"

        eval_prompts.append(prompt)

    print(f"📋 Prepared {len(eval_prompts)} evaluation prompts\n")

    # 最適化実行
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    optimizer = CMAESPersonaOptimizer(
        persona_id=args.persona_id,
        layer=20,
        alpha=args.alpha,
        device=args.device,
        max_iterations=args.max_iterations,
        population_size=args.population_size,
        sigma0=args.sigma,
        save_dir=str(output_dir)
    )

    print(f"🚀 Starting optimization...\n")

    results = optimizer.optimize(eval_prompts=eval_prompts)

    print(f"\n{'='*80}")
    print(f"✅ Optimization Complete")
    print(f"{'='*80}")
    print(f"  Best score: {results['best_score']:.4f}")
    print(f"  L2 norm: {results['l2_norm']:.4f}")
    print(f"  Iterations: {results['num_iterations']}")
    print(f"  Best weights:")
    for trait, weight in results['best_weights'].items():
        print(f"    {trait}: {weight:>8.3f}")
    print(f"\n📁 Results saved to: {output_dir}/{args.persona_id}/")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
