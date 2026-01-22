#!/usr/bin/env python3
"""
Optimize 7 new personas using existing R1-R5 trait vectors and CMA-ES.

Matches Phase 1-C configuration:
- Trait vectors: R1-R5 from data/steering_vectors_v2/
- Source model: Llama-3-8B-Instruct
- Layer: 20
- Alpha: 2.0
- CMA-ES: 100 iterations, sigma=0.3
"""

import json
import sys
import time
from pathlib import Path
from datetime import datetime

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from persona_opt.run_cma_es import CMAESOptimizer

# New personas to optimize
NEW_PERSONAS = [
    "episode-5289_A",
    "episode-29600_A",
    "episode-88279_B",
    "episode-132247_A",
    "episode-166805_A",
    "episode-196697_B",
    "episode-225888_A",
]

# Configuration (matches existing 3 personas)
CONFIG = {
    "n_dims": 5,  # R1-R5
    "pop_size": 8,
    "n_parents": 4,
    "sigma0": 0.3,
    "bounds": (-10.0, 10.0),
    "generator_model": "hf//data02/nakata/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2",
    "judge_model": "gpt-4o-mini",
    "layer": 20,
    "alpha": 2.0,
    "use_ospace": False,  # Use semantic R1-R5 space
    "max_iterations": 15,
}


def create_eval_prompts(persona_id: str, output_dir: Path):
    """Create eval prompts file for persona."""
    # Use standard evaluation prompts
    prompts = [
        {"id": 1, "text": "効率的な学習方法について教えてください"},
        {"id": 2, "text": "チーム開発で重要なことは何ですか？"},
        {"id": 3, "text": "新しい技術を学ぶ際のアプローチを教えてください"},
        {"id": 4, "text": "コードレビューのベストプラクティスは？"},
        {"id": 5, "text": "プロジェクト管理のコツを教えてください"},
    ]

    output_file = output_dir / "eval_prompts.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({"prompts": prompts}, f, indent=2, ensure_ascii=False)

    print(f"  Created eval prompts: {output_file}")
    return output_file


def optimize_persona(persona_id: str, persona_idx: int, total: int):
    """Optimize a single persona using CMA-ES."""
    print("\n" + "="*80)
    print(f"OPTIMIZING PERSONA [{persona_idx}/{total}]: {persona_id}")
    print("="*80)

    # Create output directory
    output_dir = Path(f"persona-opt/{persona_id}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create eval prompts
    prompts_file = create_eval_prompts(persona_id, output_dir)

    # Check if optimization already exists
    best_weights_file = output_dir / "best_weights.json"
    if best_weights_file.exists():
        print(f"  ⚠️  Optimization already exists: {best_weights_file}")
        response = input("  Overwrite? [y/N]: ")
        if response.lower() != 'y':
            print("  Skipping...")
            return False

    # Initialize optimizer
    print(f"\nInitializing CMA-ES optimizer...")
    print(f"  Dimensions: {CONFIG['n_dims']}")
    print(f"  Population: {CONFIG['pop_size']}")
    print(f"  Sigma: {CONFIG['sigma0']}")
    print(f"  Max iterations: {CONFIG['max_iterations']}")

    start_time = time.time()

    try:
        optimizer = CMAESOptimizer(
            n_dims=CONFIG['n_dims'],
            pop_size=CONFIG['pop_size'],
            n_parents=CONFIG['n_parents'],
            sigma0=CONFIG['sigma0'],
            bounds=CONFIG['bounds'],
            generator_model=CONFIG['generator_model'],
            judge_model=CONFIG['judge_model'],
            log_dir=f"logs/cma_es/{persona_id}",
            prompts_file=str(prompts_file),
            use_ospace=CONFIG['use_ospace'],
        )

        # Run optimization
        print(f"\nRunning CMA-ES optimization (max {CONFIG['max_iterations']} iterations)...")
        print("This will take approximately 6-8 hours per persona.")
        print("Press Ctrl+C to stop early (best weights will be saved)")

        best_solution, best_score = optimizer.optimize(n_generations=CONFIG['max_iterations'])

        elapsed = time.time() - start_time
        print(f"\n✓ Optimization complete!")
        print(f"  Best score: {best_score:.4f}")
        print(f"  Elapsed time: {elapsed/3600:.2f} hours")

        # Save results
        result = {
            "persona_id": persona_id,
            "layer": CONFIG['layer'],
            "alpha": CONFIG['alpha'],
            "weights": best_solution.tolist(),
            "trait_names": ["R1", "R2", "R3", "R4", "R5"],
            "score": float(best_score),
            "timestamp": datetime.now().strftime("%Y-%m-%d"),
            "model": CONFIG['generator_model'],
            "optimization_config": {
                "max_iterations": CONFIG['max_iterations'],
                "pop_size": CONFIG['pop_size'],
                "sigma0": CONFIG['sigma0'],
            }
        }

        with open(best_weights_file, 'w') as f:
            json.dump(result, f, indent=2)

        print(f"  ✓ Saved to: {best_weights_file}")
        return True

    except KeyboardInterrupt:
        print("\n\n⚠️  Optimization interrupted by user")
        # Save partial results if available
        if hasattr(optimizer, 'best_solution') and optimizer.best_solution is not None:
            result = {
                "persona_id": persona_id,
                "layer": CONFIG['layer'],
                "alpha": CONFIG['alpha'],
                "weights": optimizer.best_solution.tolist(),
                "trait_names": ["R1", "R2", "R3", "R4", "R5"],
                "score": float(optimizer.best_score),
                "timestamp": datetime.now().strftime("%Y-%m-%d"),
                "model": CONFIG['generator_model'],
                "status": "interrupted",
            }

            with open(best_weights_file, 'w') as f:
                json.dump(result, f, indent=2)

            print(f"  ✓ Saved partial results to: {best_weights_file}")

        return False

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Optimize all 7 new personas."""
    print("="*80)
    print("7 PERSONAS OPTIMIZATION (CMA-ES)")
    print("="*80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Personas: {len(NEW_PERSONAS)}")
    print(f"Est. total time: {len(NEW_PERSONAS) * 7} hours (~{len(NEW_PERSONAS) * 7 / 24:.1f} days)")
    print("="*80)

    # Check prerequisites
    trait_vectors_dir = Path("data/steering_vectors_v2")
    if not trait_vectors_dir.exists():
        print(f"\n✗ ERROR: Trait vectors directory not found: {trait_vectors_dir}")
        print("Run trait vector extraction first!")
        return 1

    for trait in ["R1", "R2", "R3", "R4", "R5"]:
        trait_dir = trait_vectors_dir / trait
        if not trait_dir.exists():
            print(f"\n✗ ERROR: Trait {trait} not found: {trait_dir}")
            return 1

    print("\n✓ All trait vectors (R1-R5) found")

    # Start optimization
    print(f"\n⚠️  This will take approximately {len(NEW_PERSONAS) * 7} hours to complete.")
    print("Starting optimization...")

    # Optimize each persona
    results = []
    for i, persona_id in enumerate(NEW_PERSONAS, start=1):
        success = optimize_persona(persona_id, i, len(NEW_PERSONAS))
        results.append((persona_id, success))

        if not success:
            print(f"\n⚠️  Persona {persona_id} optimization failed or was skipped")

    # Summary
    print("\n" + "="*80)
    print("OPTIMIZATION SUMMARY")
    print("="*80)

    successful = sum(1 for _, success in results if success)
    print(f"\nCompleted: {successful}/{len(NEW_PERSONAS)}")

    for persona_id, success in results:
        status = "✓" if success else "✗"
        print(f"  {status} {persona_id}")

    if successful == len(NEW_PERSONAS):
        print("\n✓ ALL PERSONAS OPTIMIZED SUCCESSFULLY!")
        print("\nNext step: Run cross-model generation")
        print("  python experiments/run_phase1b_generation.py")
    else:
        print(f"\n⚠️  {len(NEW_PERSONAS) - successful} persona(s) failed")

    return 0 if successful == len(NEW_PERSONAS) else 1


if __name__ == "__main__":
    sys.exit(main())
