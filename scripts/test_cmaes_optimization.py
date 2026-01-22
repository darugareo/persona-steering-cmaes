"""
Quick test of CMA-ES optimization with minimal iterations.

Tests the complete optimization pipeline with 3 iterations and 3 prompts.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from persona_opt.cmaes_persona_optimizer import CMAESPersonaOptimizer


def test_optimization():
    """Test optimization with minimal settings."""

    # Test prompts (minimal set)
    test_prompts = [
        "A friend is going through a difficult time. What do you say?",
        "Someone asks about your weekend. What do you tell them?",
        "You had an interesting experience today. What do you share?"
    ]

    # Initialize optimizer
    print("Initializing optimizer...")
    optimizer = CMAESPersonaOptimizer(
        persona_id="episode-184019_A",
        layer=20,
        alpha=2.0,
        eval_prompts=test_prompts
    )

    print(f"\n✓ Loaded {len(optimizer.trait_vectors)} trait vectors")
    print(f"✓ Using {len(test_prompts)} evaluation prompts")
    print(f"✓ Persona: {optimizer.persona_id}")
    print(f"✓ Layer: {optimizer.layer}")

    # Run optimization (minimal iterations for testing)
    print("\nRunning test optimization (3 iterations)...")
    results = optimizer.optimize(
        max_iterations=3,
        sigma0=1.0,
        save_dir="test_optimization_results"
    )

    print("\n" + "=" * 60)
    print("TEST COMPLETED")
    print("=" * 60)
    print(f"Best score: {results['best_score']:.4f}")
    print(f"Total evaluations: {results['num_evaluations']}")
    print("\nBest weights:")
    for trait, w in results['best_weights'].items():
        print(f"  {trait}: {w:.4f}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("CMA-ES Optimization Test")
    print("=" * 60)
    print("This is a quick test with minimal settings:")
    print("  - 3 iterations")
    print("  - 3 evaluation prompts")
    print("  - 1 persona")
    print("=" * 60)

    try:
        results = test_optimization()
        print("\n✅ Test passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
