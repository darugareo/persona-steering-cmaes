"""
Baseline 5: Random Search
Random search over trait weight space.
Fair comparison with same evaluation budget as CMA-ES.
"""

from typing import List, Dict, Any, Optional
import torch
import numpy as np
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


class RandomSearchMethod:
    """
    Random search for optimal trait weights.
    """

    def __init__(
        self,
        steerer,
        persona_id: str,
        layer: int = 20,
        alpha: float = 2.0,
        n_iterations: int = 100,
        search_range: tuple = (-10.0, 10.0),
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
            n_iterations: Number of random samples (match CMA-ES budget)
            search_range: (min, max) for weight sampling
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
        self.n_iterations = n_iterations
        self.search_range = search_range
        self.judge_model = judge_model
        self.vectors_dir = vectors_dir
        self.seed = seed
        self.method_name = "random_search"

        # Set random seed
        np.random.seed(seed)

        # Load trait vectors
        self.trait_vectors = self._load_trait_vectors()

        # Run random search if eval prompts provided
        if eval_prompts is not None:
            self.best_weights, self.best_score = self._run_random_search(
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

    def _run_random_search(
        self,
        eval_prompts: List[str],
        persona_profile: Optional[Dict]
    ) -> tuple:
        """
        Run random search optimization.

        Returns:
            (best_weights, best_score)
        """
        print(f"[RandomSearch] Starting random search with {self.n_iterations} iterations...")

        # Initialize evaluator
        evaluator = PersonaAwareEvaluator(
            persona_id=self.persona_id,
            persona_profile=persona_profile,
            judge_model=self.judge_model
        )

        best_weights = None
        best_score = -float('inf')
        history = []

        for iteration in range(self.n_iterations):
            # Sample random weights
            weights = np.random.uniform(
                self.search_range[0],
                self.search_range[1],
                size=5
            ).tolist()

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
                print(f"[RandomSearch] Iteration {iteration+1}/{self.n_iterations}: New best score = {mean_score:.2f}")

            history.append({
                'iteration': iteration,
                'weights': weights,
                'score': mean_score
            })

        print(f"[RandomSearch] Optimization complete. Best score: {best_score:.2f}")
        print(f"[RandomSearch] Best weights: {best_weights}")

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
            'n_iterations': self.n_iterations,
            'search_range': self.search_range,
            'best_weights': self.best_weights,
            'best_score': float(self.best_score),
            'description': 'Random search over trait weights',
            'steering': 'activation_based',
        }

    def __repr__(self):
        return f"RandomSearchMethod(persona_id={self.persona_id}, n_iter={self.n_iterations})"
