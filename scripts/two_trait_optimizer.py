#!/usr/bin/env python3
"""
2-Trait Steering Optimizer (R2 + R4 only)

Based on analysis showing:
- R4: p < 0.001 (highly significant, +3.049 difference)
- R2: p = 0.0012 (significant, +2.344 difference)
- R1, R5: no significant effect

Hypothesis: Removing ineffective traits will lead to larger weights
and better steering effect.
"""

import torch
import numpy as np
import cma
import json
from pathlib import Path
from typing import Dict, List
import sys
import argparse
import difflib

sys.path.insert(0, str(Path(__file__).parent.parent))
from persona_opt.internal_steering_l3 import Llama3ActivationSteerer

class TwoTraitOptimizer:
    """CMA-ES optimizer for 2-trait steering (R2 + R4 only)"""

    def __init__(
        self,
        persona_id: str,
        model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        layer: int = 20,
        alpha: float = 2.0,
        device: str = "cuda:0"
    ):
        self.persona_id = persona_id
        self.layer = layer
        self.alpha = alpha
        self.device = device

        print(f"\n{'='*80}")
        print(f"2-Trait Steering Optimizer: {persona_id}")
        print(f"Using only R2 and R4 (statistically significant traits)")
        print(f"{'='*80}\n")

        # Load persona data
        profile_path = Path(f"personas_cc/{persona_id}/profile.json")
        train_path = Path(f"personas_cc/{persona_id}/train_turns.json")

        with open(profile_path) as f:
            self.profile = json.load(f)

        with open(train_path) as f:
            train_data = json.load(f)
            self.train_turns = train_data["turns"][:10]  # Use 10 training turns

        print(f"Loaded {len(self.train_turns)} training turns")

        # Load model
        print(f"Loading model: {model_name}")
        self.steerer = Llama3ActivationSteerer(
            model_name=model_name,
            target_layer=layer,
            device=device
        )

        # Load ONLY R2 and R4 trait vectors
        self.trait_vectors = self._load_two_trait_vectors()

        # Optimization log
        self.optimization_log = {
            "persona_id": persona_id,
            "traits_used": ["R2", "R4"],
            "layer": layer,
            "alpha": alpha,
            "generations": []
        }

    def _load_two_trait_vectors(self) -> Dict[str, torch.Tensor]:
        """Load ONLY R2 and R4 trait vectors"""
        trait_dir = Path("data/steering_vectors_v2_inverted")
        vectors = {}

        for trait in ["R2", "R4"]:
            vector_path = trait_dir / trait / "layer20_svd.pt"
            if not vector_path.exists():
                raise FileNotFoundError(f"Trait vector not found: {vector_path}")

            vector = torch.load(vector_path, map_location='cpu', weights_only=False)
            vectors[trait] = vector.to(self.device)
            print(f"✓ Loaded {trait}: shape={vector.shape}")

        return vectors

    def _calculate_style_similarity(self, generated: List[str], ground_truth: List[str]) -> float:
        """
        Calculate simple style similarity using sequence matching.

        Returns a score between 0 and 1, where 1 means perfect similarity.
        """
        if len(generated) != len(ground_truth):
            raise ValueError("Generated and ground truth lists must have same length")

        similarities = []
        for gen, gt in zip(generated, ground_truth):
            # Use SequenceMatcher for similarity ratio
            ratio = difflib.SequenceMatcher(None, gen.lower(), gt.lower()).ratio()
            similarities.append(ratio)

        return float(np.mean(similarities))

    def _calculate_fitness(self, weights: np.ndarray) -> float:
        """
        Calculate fitness for given 2D weights [w_R2, w_R4]

        Uses Style Similarity as it showed best generalization
        in previous experiments.
        """
        # Convert to dict
        weight_dict = {
            "R2": float(weights[0]),
            "R4": float(weights[1])
        }

        # Generate responses with steering
        responses = []

        for turn in self.train_turns:
            context = turn["context"]
            input_text = turn["input"]

            # Build prompt
            prompt = f"""Continue this conversation naturally.

Conversation so far:
{context}

Partner: {input_text}

You:"""

            # Apply steering
            scaled_weights = {k: v * self.alpha for k, v in weight_dict.items()}
            self.steerer.register_hooks(
                multi_trait_vectors=self.trait_vectors,
                trait_weights=scaled_weights
            )

            # Generate
            response = self.steerer.generate(
                prompt=prompt,
                max_new_tokens=100,
                temperature=0.7
            )

            responses.append(response)

            # Remove hooks
            self.steerer.remove_hooks()

        # Calculate style similarity
        ground_truths = [turn["ground_truth"] for turn in self.train_turns]

        fitness = self._calculate_style_similarity(
            generated=responses,
            ground_truth=ground_truths
        )

        return fitness

    def optimize(
        self,
        initial_mean: List[float] = [0.0, 0.0],
        initial_sigma: float = 2.0,
        popsize: int = 8,
        max_generations: int = 20
    ) -> Dict:
        """
        Run CMA-ES optimization for 2 dimensions (R2, R4)

        Args:
            initial_mean: Starting point [w_R2, w_R4]
            initial_sigma: Initial step size (larger = wider exploration)
            popsize: Population size per generation
            max_generations: Maximum number of generations
        """
        print(f"\nStarting CMA-ES optimization:")
        print(f"  Dimensions: 2 (R2, R4 only)")
        print(f"  Initial mean: {initial_mean}")
        print(f"  Initial sigma: {initial_sigma}")
        print(f"  Population size: {popsize}")
        print(f"  Max generations: {max_generations}")
        print()

        # Initialize CMA-ES
        es = cma.CMAEvolutionStrategy(
            initial_mean,
            initial_sigma,
            {
                'popsize': popsize,
                'bounds': [-10, 10],  # Wider bounds to allow larger weights
                'verbose': -1
            }
        )

        best_fitness = -float('inf')
        best_weights = None
        generation = 0

        while not es.stop() and generation < max_generations:
            generation += 1
            print(f"Generation {generation}/{max_generations}:")

            # Generate population
            solutions = es.ask()

            # Evaluate fitness (negative because CMA-ES minimizes)
            fitnesses = []
            for i, weights in enumerate(solutions):
                fitness = self._calculate_fitness(weights)
                fitnesses.append(-fitness)  # Negate for minimization

                print(f"  Individual {i+1}/{popsize}: "
                      f"R2={weights[0]:+.3f}, R4={weights[1]:+.3f} "
                      f"→ fitness={fitness:.4f}")

                # Track best
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_weights = weights.copy()

            # Update CMA-ES
            es.tell(solutions, fitnesses)

            # Log generation
            self.optimization_log["generations"].append({
                "generation": generation,
                "best_fitness": float(best_fitness),
                "best_weights": {
                    "R2": float(best_weights[0]),
                    "R4": float(best_weights[1])
                },
                "mean_fitness": float(-np.mean(fitnesses)),
                "population_mean": es.result.xfavorite.tolist()
            })

            print(f"  Best so far: R2={best_weights[0]:+.3f}, R4={best_weights[1]:+.3f}, "
                  f"fitness={best_fitness:.4f}")
            print()

        # Final result
        final_weights = {
            "R2": float(best_weights[0]),
            "R4": float(best_weights[1])
        }

        # Calculate L2 norm for comparison with 5-trait
        l2_norm = np.linalg.norm(best_weights)

        print(f"\n{'='*80}")
        print(f"Optimization complete!")
        print(f"  Best fitness: {best_fitness:.4f}")
        print(f"  Best weights: R2={final_weights['R2']:+.3f}, R4={final_weights['R4']:+.3f}")
        print(f"  L2 norm: {l2_norm:.3f}")
        print(f"  Generations: {generation}")
        print(f"{'='*80}\n")

        self.optimization_log["final_result"] = {
            "best_fitness": float(best_fitness),
            "best_weights": final_weights,
            "l2_norm": float(l2_norm),
            "total_generations": generation
        }

        return {
            "best_weights": final_weights,
            "best_fitness": best_fitness,
            "l2_norm": l2_norm,
            "log": self.optimization_log
        }

    def save_results(self, results: Dict, output_dir: Path):
        """Save optimization results"""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save weights
        weights_file = output_dir / f"{self.persona_id}_weights.json"
        with open(weights_file, "w") as f:
            json.dump(results["best_weights"], f, indent=2)

        # Save log
        log_file = output_dir / f"{self.persona_id}_log.json"
        with open(log_file, "w") as f:
            json.dump(results["log"], f, indent=2)

        print(f"✓ Saved results:")
        print(f"  Weights: {weights_file}")
        print(f"  Log: {log_file}")

def main():
    parser = argparse.ArgumentParser(description="2-Trait Steering Optimizer")
    parser.add_argument("--persona_id", required=True, help="Persona ID")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
    parser.add_argument("--max_generations", type=int, default=20, help="Max CMA-ES generations")
    parser.add_argument("--popsize", type=int, default=8, help="Population size")
    parser.add_argument("--initial_sigma", type=float, default=2.0, help="Initial CMA-ES sigma")

    args = parser.parse_args()

    device = f"cuda:{args.gpu_id}"

    # Create optimizer
    optimizer = TwoTraitOptimizer(
        persona_id=args.persona_id,
        device=device
    )

    # Run optimization
    results = optimizer.optimize(
        initial_sigma=args.initial_sigma,
        popsize=args.popsize,
        max_generations=args.max_generations
    )

    # Save results
    output_dir = Path("optimization_results_2trait") / args.persona_id
    optimizer.save_results(results, output_dir)

    print("\n✅ 2-Trait optimization complete!")

if __name__ == "__main__":
    main()
