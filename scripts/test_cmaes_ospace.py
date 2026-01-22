#!/usr/bin/env python3
"""
Test CMA-ES with O-space optimization (1 generation)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from persona_opt.run_cma_es import CMAESOptimizer

print("="*80)
print("CMA-ES O-SPACE TEST (1 Generation)")
print("="*80)

# Initialize optimizer with O-space
print("\nInitializing CMA-ES optimizer...")
print("  - Mode: O-space (orthogonal trait space)")
print("  - Dimensions: 6 (O1-O6)")
print("  - Population: 4")
print("  - Parents: 2")
print("  - Generations: 1 (test)")

optimizer = CMAESOptimizer(
    n_dims=6,
    pop_size=4,
    n_parents=2,
    sigma0=0.15,
    bounds=(-3.0, 3.0),
    generator_model="mock",  # Use mock for testing
    judge_model="mock",
    log_dir="logs/test_ospace",
    use_ospace=True
)

print(f"\n✓ Optimizer initialized")
print(f"  - Using O-space: {optimizer.use_ospace}")
print(f"  - Log directory: {optimizer.log_dir}")

# Run 1 generation
print("\nRunning 1 generation...")
results = optimizer.optimize(
    n_generations=1,
    tau=0.8,
    verbose=True
)

print("\n" + "="*80)
print("RESULTS")
print("="*80)
print(f"Best fitness: {results['best_fitness']:.4f}")
print(f"Best traits (semantic): {results['best_traits']}")
print(f"History: {len(results['history'])} generations")

print("\n✓ CMA-ES O-space test completed successfully!")
print(f"  Check logs in: {optimizer.log_dir}")
print("="*80)
