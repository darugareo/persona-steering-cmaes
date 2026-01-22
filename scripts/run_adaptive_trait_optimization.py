#!/usr/bin/env python3
"""
Adaptive Trait Optimization: ペルソナごとに推奨されたTraitのみで最適化

trait_recommendations.jsonから各ペルソナの推奨Traitを読み込み、
そのTraitのみで最適化を実行する。
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


class AdaptiveTraitOptimizer:
    """ペルソナごとに推奨されたTraitのみで最適化"""

    def __init__(
        self,
        persona_id: str,
        selected_traits: List[str],
        model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        layer: int = 20,
        alpha: float = 2.0,
        device: str = "cuda:0",
        warm_start_weights: Dict[str, float] = None
    ):
        self.persona_id = persona_id
        self.selected_traits = selected_traits
        self.layer = layer
        self.alpha = alpha
        self.device = device

        print(f"\n{'='*80}")
        print(f"Adaptive Trait Optimizer: {persona_id}")
        print(f"Selected Traits: {', '.join(selected_traits)}")
        print(f"{'='*80}\n")

        # Load persona data
        profile_path = Path(f"personas_cc/{persona_id}/profile.json")
        train_path = Path(f"personas_cc/{persona_id}/train_turns.json")

        with open(profile_path) as f:
            self.profile = json.load(f)

        with open(train_path) as f:
            train_data = json.load(f)
            self.train_turns = train_data["turns"][:10]

        print(f"Loaded {len(self.train_turns)} training turns")

        # Load model
        print(f"Loading model: {model_name}")
        self.steerer = Llama3ActivationSteerer(
            model_name=model_name,
            target_layer=layer,
            device=device
        )

        # Load ONLY selected trait vectors
        self.trait_vectors = self._load_selected_trait_vectors()

        # Warm start weights (optional)
        self.warm_start_weights = warm_start_weights

        # Optimization log
        self.optimization_log = {
            "persona_id": persona_id,
            "traits_used": selected_traits,
            "layer": layer,
            "alpha": alpha,
            "warm_start": warm_start_weights is not None,
            "generations": []
        }

    def _load_selected_trait_vectors(self) -> Dict[str, torch.Tensor]:
        """Load ONLY selected trait vectors"""
        trait_dir = Path("data/steering_vectors_v2_inverted")
        vectors = {}

        for trait in self.selected_traits:
            vector_path = trait_dir / trait / "layer20_svd.pt"
            if not vector_path.exists():
                raise FileNotFoundError(f"Trait vector not found: {vector_path}")

            vector = torch.load(vector_path, map_location='cpu', weights_only=False)
            vectors[trait] = vector.to(self.device)
            print(f"✓ Loaded {trait}: shape={vector.shape}")

        return vectors

    def _calculate_style_similarity(self, generated: List[str], ground_truth: List[str]) -> float:
        """Calculate style similarity using sequence matching"""
        if len(generated) != len(ground_truth):
            raise ValueError("Generated and ground truth lists must have same length")

        similarities = []
        for gen, gt in zip(generated, ground_truth):
            ratio = difflib.SequenceMatcher(None, gen.lower(), gt.lower()).ratio()
            similarities.append(ratio)

        return float(np.mean(similarities))

    def _calculate_fitness(self, weights: np.ndarray) -> float:
        """
        Calculate fitness for given weights

        Args:
            weights: Array of weights corresponding to selected_traits
        """
        # Convert to dict
        weight_dict = {
            trait: float(weights[i])
            for i, trait in enumerate(self.selected_traits)
        }

        # Generate responses with steering
        responses = []

        for turn in self.train_turns:
            context = turn["context"]
            input_text = turn["input"]

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
        initial_sigma: float = 2.0,
        popsize: int = 8,
        max_generations: int = 20
    ) -> Dict:
        """
        Run CMA-ES optimization for selected traits

        Args:
            initial_sigma: Initial step size
            popsize: Population size per generation
            max_generations: Maximum number of generations
        """
        n_dims = len(self.selected_traits)

        # Set initial mean
        if self.warm_start_weights:
            # Warm start from 5-Trait optimized weights
            initial_mean = [
                self.warm_start_weights.get(trait, 0.0)
                for trait in self.selected_traits
            ]
            print(f"\n🔥 Warm start enabled")
            print(f"   Initial weights: {dict(zip(self.selected_traits, initial_mean))}")
        else:
            # Cold start from zero
            initial_mean = [0.0] * n_dims

        print(f"\nStarting CMA-ES optimization:")
        print(f"  Dimensions: {n_dims} ({', '.join(self.selected_traits)})")
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
                'bounds': [-10, 10],
                'verbose': -1
            }
        )

        best_fitness = -float('inf')
        best_weights = None
        generation = 0

        while not es.stop() and generation < max_generations:
            generation += 1
            print(f"Generation {generation}/{max_generations}:")

            solutions = es.ask()
            fitnesses = []

            for i, weights in enumerate(solutions):
                fitness = self._calculate_fitness(weights)
                fitnesses.append(-fitness)  # Negate for minimization

                weight_str = ", ".join(
                    f"{trait}={weights[j]:+.3f}"
                    for j, trait in enumerate(self.selected_traits)
                )
                print(f"  Individual {i+1}/{popsize}: {weight_str} → fitness={fitness:.4f}")

                if fitness > best_fitness:
                    best_fitness = fitness
                    best_weights = weights.copy()

            es.tell(solutions, fitnesses)

            # Log generation
            best_weight_dict = {
                trait: float(best_weights[i])
                for i, trait in enumerate(self.selected_traits)
            }

            self.optimization_log["generations"].append({
                "generation": generation,
                "best_fitness": float(best_fitness),
                "best_weights": best_weight_dict,
                "mean_fitness": float(-np.mean(fitnesses)),
                "population_mean": es.result.xfavorite.tolist()
            })

            best_weight_str = ", ".join(
                f"{trait}={best_weights[i]:+.3f}"
                for i, trait in enumerate(self.selected_traits)
            )
            print(f"  Best so far: {best_weight_str}, fitness={best_fitness:.4f}")
            print()

        # Final result
        final_weights = {
            trait: float(best_weights[i])
            for i, trait in enumerate(self.selected_traits)
        }

        l2_norm = np.linalg.norm(best_weights)

        print(f"\n{'='*80}")
        print(f"Optimization complete!")
        print(f"  Best fitness: {best_fitness:.4f}")
        print(f"  Best weights: {final_weights}")
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

        weights_file = output_dir / f"{self.persona_id}_weights.json"
        with open(weights_file, "w") as f:
            json.dump(results["best_weights"], f, indent=2)

        log_file = output_dir / f"{self.persona_id}_log.json"
        with open(log_file, "w") as f:
            json.dump(results["log"], f, indent=2)

        print(f"✓ Saved results:")
        print(f"  Weights: {weights_file}")
        print(f"  Log: {log_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Adaptive Trait Optimization: ペルソナごとに推奨Traitで最適化"
    )
    parser.add_argument("--persona_id", required=True, help="Persona ID")
    parser.add_argument(
        "--recommendations",
        default="trait_recommendations_threshold1.5.json",
        help="Trait recommendations JSON file"
    )
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
    parser.add_argument("--max_generations", type=int, default=20, help="Max CMA-ES generations")
    parser.add_argument("--popsize", type=int, default=8, help="Population size")
    parser.add_argument("--initial_sigma", type=float, default=2.0, help="Initial CMA-ES sigma")
    parser.add_argument(
        "--warm_start",
        action="store_true",
        help="Warm start from 5-Trait optimized weights"
    )
    parser.add_argument(
        "--output_dir",
        default="optimization_results_adaptive",
        help="Output directory"
    )

    args = parser.parse_args()

    device = f"cuda:{args.gpu_id}"

    # Load recommendations
    with open(args.recommendations) as f:
        recommendations = json.load(f)

    if args.persona_id not in recommendations["recommendations"]:
        print(f"❌ Persona {args.persona_id} not found in recommendations")
        sys.exit(1)

    persona_rec = recommendations["recommendations"][args.persona_id]

    selected_traits = persona_rec["selected_traits"]

    if not selected_traits:
        print(f"⚠️  Persona {args.persona_id} has no selected traits (all weights < threshold)")
        print(f"   Skipping optimization.")
        sys.exit(0)

    # Warm start weights (optional)
    warm_start_weights = None
    if args.warm_start:
        warm_start_weights = persona_rec["all_weights"]

    # Create optimizer
    optimizer = AdaptiveTraitOptimizer(
        persona_id=args.persona_id,
        selected_traits=selected_traits,
        device=device,
        warm_start_weights=warm_start_weights
    )

    # Run optimization
    results = optimizer.optimize(
        initial_sigma=args.initial_sigma,
        popsize=args.popsize,
        max_generations=args.max_generations
    )

    # Save results
    output_dir = Path(args.output_dir) / args.persona_id
    optimizer.save_results(results, output_dir)

    print("\n✅ Adaptive trait optimization complete!")


if __name__ == "__main__":
    main()
