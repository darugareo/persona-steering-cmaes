"""
Baseline 6: Grid Search
Limited grid search over trait weight space.
Combinatorial explosion controlled by limiting grid points.
"""

from typing import List, Dict, Any, Optional
import torch
import numpy as np
from itertools import product
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from persona_opt.evaluation.utils import load_steering_vectors
from persona_opt.evaluator import PersonaAwareEvaluator


def build_combined_steering_vector(weights: List[float], trait_vectors: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Build combined steering vector from weights and trait vectors."""
    combined = torch.zeros_like(list(trait_vectors.values())[0])
    traits = ["R1", "R2", "R3", "R4", "R5"]

    for i, trait in enumerate(traits):
        if trait in trait_vectors:
            combined += weights[i] * trait_vectors[trait]

    return combined


class GridSearchMethod:
    """
    Grid search for optimal trait weights.
    Limited to avoid combinatorial explosion.
    """

    def __init__(
        self,
        steerer,
        persona_id: str,
        layer: int = 20,
        alpha: float = 2.0,
        grid_points: int = 3,
        grid_range: tuple = (-2.0, 2.0),
        max_combinations: int = 100,
        eval_prompts: Optional[List[str]] = None,
        persona_profile: Optional[Dict] = None,
        judge_model: str = "gpt-4o-mini",
        vectors_dir: str = "data/steering_vectors_v2",
        seed: int = 42,
        **kwargs
    ):
        """
        Args:
            steerer: Llama3ActivationSteerer instance
            persona_id: Persona identifier
            layer: Target layer for steering
            alpha: Steering strength
            grid_points: Number of grid points per dimension (e.g., 3 = [-2, 0, 2])
            grid_range: (min, max) for grid
            max_combinations: Maximum combinations to evaluate (limit explosion)
            eval_prompts: Prompts for evaluation
            persona_profile: Preloaded persona profile
            judge_model: LLM judge model name
            vectors_dir: Directory containing trait vectors
            seed: Random seed
            **kwargs: Additional configuration
        """
        self.steerer = steerer
        self.persona_id = persona_id
        self.layer = layer
        self.alpha = alpha
        self.grid_points = grid_points
        self.grid_range = grid_range
        self.max_combinations = max_combinations
        self.judge_model = judge_model
        self.vectors_dir = vectors_dir
        self.seed = seed
        self.method_name = "grid_search"

        # Set random seed for sampling
        np.random.seed(seed)

        # Load trait vectors
        self.trait_vectors = self._load_trait_vectors()

        # Run grid search if eval prompts provided
        if eval_prompts is not None:
            self.best_weights, self.best_score = self._run_grid_search(
                eval_prompts, persona_profile
            )
            self.steering_vector = build_combined_steering_vector(
                self.best_weights, self.trait_vectors
            )
        else:
            # Initialize with zeros if no optimization
            self.best_weights = [0.0] * 5
            self.best_score = 0.0
            self.steering_vector = torch.zeros(self.steerer.model.config.hidden_size)

    def _load_trait_vectors(self) -> Dict[str, torch.Tensor]:
        """Load trait vectors for the specified layer."""
        trait_vectors = {}
        traits = ["R1", "R2", "R3", "R4", "R5"]

        for trait in traits:
            vector_file = Path(self.vectors_dir) / trait / f"layer{self.layer}_svd.pt"
            vector_data = torch.load(vector_file, map_location='cpu')

            # Handle different formats
            if isinstance(vector_data, torch.Tensor):
                trait_vectors[trait] = vector_data
            elif isinstance(vector_data, dict) and 'vector' in vector_data:
                trait_vectors[trait] = vector_data['vector']
            else:
                raise ValueError(f"Unexpected vector format in {vector_file}")

        return trait_vectors

    def _generate_grid(self) -> List[List[float]]:
        """
        Generate grid points, with sampling if too many combinations.

        Returns:
            List of weight vectors
        """
        # Create grid points for one dimension
        grid_values = np.linspace(
            self.grid_range[0],
            self.grid_range[1],
            self.grid_points
        )

        # Generate all combinations (5 traits)
        all_combinations = list(product(grid_values, repeat=5))

        total_combinations = len(all_combinations)
        print(f"[GridSearch] Total grid combinations: {total_combinations}")

        # Sample if too many
        if total_combinations > self.max_combinations:
            print(f"[GridSearch] Sampling {self.max_combinations} combinations...")
            indices = np.random.choice(
                total_combinations,
                self.max_combinations,
                replace=False
            )
            sampled_combinations = [all_combinations[i] for i in indices]
            return [list(combo) for combo in sampled_combinations]
        else:
            return [list(combo) for combo in all_combinations]

    def _run_grid_search(
        self,
        eval_prompts: List[str],
        persona_profile: Optional[Dict]
    ) -> tuple:
        """
        Run grid search optimization.

        Returns:
            (best_weights, best_score)
        """
        print(f"[GridSearch] Starting grid search...")
        print(f"[GridSearch] Grid points: {self.grid_points}, Range: {self.grid_range}")

        # Generate grid
        grid = self._generate_grid()
        n_combinations = len(grid)

        print(f"[GridSearch] Evaluating {n_combinations} combinations...")

        # Initialize evaluator
        evaluator = PersonaAwareEvaluator(
            persona_id=self.persona_id,
            persona_profile=persona_profile,
            judge_model=self.judge_model
        )

        best_weights = None
        best_score = -float('inf')

        for idx, weights in enumerate(grid):
            # Build steering vector
            steering_vector = build_combined_steering_vector(
                weights, self.trait_vectors
            )

            # Generate responses
            baseline_responses = self.steerer.batch_generate(eval_prompts)

            steered_responses = self.steerer.batch_generate(
                eval_prompts,
                steering_vector=steering_vector,
                layer=self.layer,
                alpha=self.alpha
            )

            # Evaluate
            results = evaluator.batch_evaluate(
                prompts=eval_prompts,
                baseline_responses=baseline_responses,
                steered_responses=steered_responses
            )

            # Compute mean score
            scores = [r['persona_fit'] for r in results]
            mean_score = np.mean(scores)

            # Update best
            if mean_score > best_score:
                best_score = mean_score
                best_weights = weights
                print(f"[GridSearch] Combination {idx+1}/{n_combinations}: New best score = {mean_score:.2f}")

        print(f"[GridSearch] Optimization complete. Best score: {best_score:.2f}")
        print(f"[GridSearch] Best weights: {best_weights}")

        return best_weights, best_score

    def generate(self, prompt: str, **generation_kwargs) -> str:
        """
        Generate response with optimized steering.

        Args:
            prompt: Input prompt
            **generation_kwargs: Generation parameters

        Returns:
            Generated response
        """
        # Apply steering
        self.steerer.register_hooks(
            steering_vector=self.steering_vector,
            alpha=self.alpha
        )

        # Generate
        response = self.steerer.generate(prompt, **generation_kwargs)

        return response

    def batch_generate(self, prompts: List[str], **generation_kwargs) -> List[str]:
        """
        Generate multiple responses with steering.

        Args:
            prompts: List of input prompts
            **generation_kwargs: Generation parameters

        Returns:
            List of generated responses
        """
        return self.steerer.batch_generate(
            prompts,
            steering_vector=self.steering_vector,
            layer=self.layer,
            alpha=self.alpha,
            **generation_kwargs
        )

    def get_config(self) -> Dict[str, Any]:
        """Get method configuration."""
        return {
            'method': self.method_name,
            'persona_id': self.persona_id,
            'layer': self.layer,
            'alpha': self.alpha,
            'grid_points': self.grid_points,
            'grid_range': self.grid_range,
            'max_combinations': self.max_combinations,
            'best_weights': self.best_weights if hasattr(self, 'best_weights') else None,
            'best_score': float(self.best_score) if hasattr(self, 'best_score') else None,
            'description': 'Grid search over trait weights',
            'steering': 'activation_based',
        }

    def __repr__(self):
        return f"GridSearchMethod(persona_id={self.persona_id}, grid={self.grid_points})"
