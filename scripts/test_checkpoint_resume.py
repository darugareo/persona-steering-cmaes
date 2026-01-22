#!/usr/bin/env python3
"""
Test checkpoint resume functionality for CMA-ES optimizer.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from persona_opt.cmaes_persona_optimizer import optimize_persona_weights


def test_checkpoint_resume():
    """Test checkpoint save and resume functionality."""

    persona_id = "episode-118328_B"
    save_dir = "optimization_results_checkpoint_test"

    print("=" * 80)
    print("TEST 1: Run optimization for 2 iterations")
    print("=" * 80)

    # First run - keep checkpoint to simulate interrupted optimization
    try:
        results = optimize_persona_weights(
            persona_id=persona_id,
            max_iterations=2,
            save_dir=save_dir,
            population_size=4,  # Small population for quick testing
            keep_checkpoint=True  # Keep checkpoint to test resume
        )
        print(f"\n✅ First run completed: {results['num_iterations']} iterations")
    except Exception as e:
        print(f"❌ First run failed: {e}")
        return False

    # Check checkpoint file exists
    checkpoint_file = Path(save_dir) / f"{persona_id}_checkpoint.pkl"
    print(f"\nCheckpoint exists after first run: {checkpoint_file.exists()}")

    print("\n" + "=" * 80)
    print("TEST 2: Resume from checkpoint and run 2 more iterations")
    print("=" * 80)

    # Second run - should resume from checkpoint
    try:
        results = optimize_persona_weights(
            persona_id=persona_id,
            max_iterations=4,  # Total 4 iterations
            save_dir=save_dir,
            population_size=4
        )
        print(f"\n✅ Second run completed: {results['num_iterations']} iterations")
    except Exception as e:
        print(f"❌ Second run failed: {e}")
        return False

    # Checkpoint should be removed after successful completion
    print(f"\nCheckpoint exists after second run: {checkpoint_file.exists()}")

    print("\n" + "=" * 80)
    print("CHECKPOINT RESUME TEST COMPLETE")
    print("=" * 80)
    print(f"Final iterations: {results['num_iterations']}")
    print(f"Final score: {results['best_score']:.4f}")

    return True


if __name__ == "__main__":
    success = test_checkpoint_resume()
    sys.exit(0 if success else 1)
