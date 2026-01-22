"""
CMA-ES Persona Optimizer

Optimizes trait weights for persona reproduction using CMA-ES.
Uses persona-aware judge for evaluation.
"""

import torch
import numpy as np
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cma
from datetime import datetime

from persona_opt.internal_steering_l3 import Llama3ActivationSteerer
from persona_opt.persona_judge_evaluator import evaluate_with_persona_judge


class CMAESPersonaOptimizer:
    """Optimizes trait weights for persona using CMA-ES."""

    def __init__(
        self,
        persona_id: str,
        model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        layer: int = 20,
        trait_vector_dir: str = "data/steering_vectors_v2",
        persona_dir: str = "personas",
        eval_prompts: Optional[List[str]] = None,
        alpha: float = 2.0,
        baseline_generation_params: Optional[Dict] = None,
        steered_generation_params: Optional[Dict] = None
    ):
        """
        Initialize CMA-ES optimizer for persona.

        Args:
            persona_id: Persona identifier (e.g., "episode-184019_A")
            model_name: HuggingFace model name
            layer: Layer to apply steering (default: 20)
            trait_vector_dir: Directory containing SVD trait vectors
            persona_dir: Directory containing persona profiles
            eval_prompts: List of prompts for evaluation
            alpha: Steering strength
            baseline_generation_params: Generation params for baseline
            steered_generation_params: Generation params for steered responses
        """
        self.persona_id = persona_id
        self.model_name = model_name
        self.layer = layer
        self.trait_vector_dir = Path(trait_vector_dir)
        self.persona_dir = Path(persona_dir)
        self.alpha = alpha

        # Load trait vectors
        self.trait_names = ["R1", "R2", "R3", "R4", "R5"]
        self.trait_vectors = self._load_trait_vectors()

        # Evaluation prompts
        self.eval_prompts = eval_prompts or self._load_default_eval_prompts()

        # Initialize steering
        self.steerer = Llama3ActivationSteerer(
            model_name=model_name,
            target_layer=layer
        )

        # Generation parameters
        self.baseline_params = baseline_generation_params or {
            "max_new_tokens": 150,
            "do_sample": False
        }
        self.steered_params = steered_generation_params or {
            "max_new_tokens": 150,
            "do_sample": False
        }

        # Optimization history
        self.history = {
            "iterations": [],
            "best_weights": [],
            "best_scores": [],
            "all_weights": [],
            "all_scores": []
        }

    def _load_trait_vectors(self) -> Dict[str, torch.Tensor]:
        """Load SVD trait vectors for target layer."""
        vectors = {}
        for trait in self.trait_names:
            vector_file = self.trait_vector_dir / trait / f"layer{self.layer}_svd.pt"
            if not vector_file.exists():
                raise FileNotFoundError(f"Trait vector not found: {vector_file}")
            vectors[trait] = torch.load(vector_file)
        return vectors

    def _load_default_eval_prompts(self) -> List[str]:
        """Load default evaluation prompts."""
        eval_prompts_file = Path("data/eval_prompts/persona_eval_prompts_v1.json")
        if eval_prompts_file.exists():
            with open(eval_prompts_file, 'r') as f:
                data = json.load(f)
                # Take first 15 prompts
                all_prompts = []
                for prompt_data in data["prompts"]:
                    all_prompts.append(prompt_data["text"])
                return all_prompts[:15]
        else:
            # Fallback default prompts
            return [
                "A friend is going through a difficult time. What do you say?",
                "Someone asks for your advice on a decision. How do you respond?",
                "You had an interesting experience today. What do you share?",
                "A neighbor mentions they need help with something. What's your response?",
                "Someone asks about your weekend. What do you tell them?"
            ]

    def _save_checkpoint(self, es: cma.CMAEvolutionStrategy, iteration: int, save_dir: Path):
        """Save optimization checkpoint."""
        checkpoint = {
            'es_state': es.pickle_dumps(),
            'iteration': iteration,
            'history': self.history,
            'persona_id': self.persona_id
        }
        checkpoint_file = save_dir / f"{self.persona_id}_checkpoint.pkl"
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint, f)
        print(f"  💾 Checkpoint saved: {checkpoint_file}")

    def _load_checkpoint(self, save_dir: Path) -> Optional[Tuple[cma.CMAEvolutionStrategy, int]]:
        """Load optimization checkpoint if exists."""
        checkpoint_file = save_dir / f"{self.persona_id}_checkpoint.pkl"
        if not checkpoint_file.exists():
            return None

        with open(checkpoint_file, 'rb') as f:
            checkpoint = pickle.load(f)

        # Restore CMA-ES state using pickle.loads on the bytes string
        es = pickle.loads(checkpoint['es_state'])
        iteration = checkpoint['iteration']
        self.history = checkpoint['history']

        print(f"  🔄 Resuming from checkpoint: iteration {iteration}")
        return es, iteration

    def _build_steering_vector(self, weights: np.ndarray) -> torch.Tensor:
        """
        Build combined steering vector from trait weights.

        Args:
            weights: Array of 5 weights for R1-R5

        Returns:
            Combined steering vector
        """
        combined = torch.zeros_like(self.trait_vectors["R1"])

        for i, trait in enumerate(self.trait_names):
            combined += weights[i] * self.trait_vectors[trait]

        return combined

    def _generate_responses(
        self,
        prompts: List[str],
        steering_vector: Optional[torch.Tensor] = None
    ) -> List[str]:
        """
        Generate responses with optional steering.

        Args:
            prompts: List of prompts
            steering_vector: Optional steering vector

        Returns:
            List of generated responses
        """
        if steering_vector is not None:
            # Steered generation
            self.steerer.register_hooks(
                steering_vector=steering_vector,
                alpha=self.alpha
            )
            params = self.steered_params
        else:
            # Baseline generation
            params = self.baseline_params

        responses = []
        for prompt in prompts:
            response = self.steerer.generate(prompt, **params)
            responses.append(response)

        if steering_vector is not None:
            self.steerer.remove_hooks()

        return responses

    def _evaluate_responses(
        self,
        prompts: List[str],
        baseline_responses: List[str],
        steered_responses: List[str]
    ) -> float:
        """
        Evaluate steered responses against baseline using persona-aware judge.

        Args:
            prompts: Evaluation prompts
            baseline_responses: Baseline responses
            steered_responses: Steered responses

        Returns:
            Mean persona fit score for steered responses
        """
        scores = []

        for i, (prompt, baseline, steered) in enumerate(zip(prompts, baseline_responses, steered_responses)):
            try:
                result = evaluate_with_persona_judge(
                    persona_id=self.persona_id,
                    prompt=prompt,
                    response_a=baseline,
                    response_b=steered,
                    trait_name="Overall Persona Fit",
                    trait_direction="matches persona style and values",
                    base_dir=str(self.persona_dir)
                )
                scores.append(result["persona_fit_score_b"])
            except Exception as e:
                print(f"Warning: Evaluation failed for prompt {i+1}/{len(prompts)}: {type(e).__name__}")
                scores.append(2.5)  # Neutral score on failure

        return np.mean(scores)

    def objective_function(self, weights: np.ndarray) -> float:
        """
        CMA-ES objective function.

        Args:
            weights: Trait weights [w1, w2, w3, w4, w5]

        Returns:
            Negative mean persona fit score (for minimization)
        """
        # Build steering vector
        steering_vec = self._build_steering_vector(weights)

        # Generate baseline responses (cached on first call)
        if not hasattr(self, '_baseline_responses'):
            print("Generating baseline responses...")
            self._baseline_responses = self._generate_responses(self.eval_prompts)

        # Generate steered responses
        steered_responses = self._generate_responses(
            self.eval_prompts,
            steering_vector=steering_vec
        )

        # Evaluate
        mean_score = self._evaluate_responses(
            self.eval_prompts,
            self._baseline_responses,
            steered_responses
        )

        # Store in history
        self.history["all_weights"].append(weights.tolist())
        self.history["all_scores"].append(mean_score)

        # Return negative for minimization
        return -mean_score

    def optimize(
        self,
        initial_weights: Optional[np.ndarray] = None,
        sigma0: float = 1.0,
        max_iterations: int = 50,
        population_size: Optional[int] = None,
        tolx: float = 1e-4,
        save_dir: Optional[str] = None,
        keep_checkpoint: bool = False
    ) -> Dict:
        """
        Run CMA-ES optimization.

        Args:
            initial_weights: Initial weights (default: zeros)
            sigma0: Initial standard deviation
            max_iterations: Maximum number of iterations
            population_size: CMA-ES population size (default: auto)
            tolx: Tolerance for convergence
            save_dir: Directory to save results
            keep_checkpoint: Keep checkpoint file after completion (default: False)

        Returns:
            Optimization results dictionary
        """
        # Initial weights
        if initial_weights is None:
            initial_weights = np.zeros(5)

        # Prepare save directory
        save_path = Path(save_dir) if save_dir else None
        if save_path:
            save_path.mkdir(parents=True, exist_ok=True)

        # CMA-ES options
        opts = {
            'maxiter': max_iterations,
            'tolx': tolx,
            'verbose': 1,
            'verb_disp': 1
        }
        if population_size is not None:
            opts['popsize'] = population_size

        print("=" * 80)
        print(f"CMA-ES Optimization: {self.persona_id}")
        print("=" * 80)
        print(f"Model: {self.model_name}")
        print(f"Layer: {self.layer}")
        print(f"Alpha: {self.alpha}")
        print(f"Eval prompts: {len(self.eval_prompts)}")
        print(f"Initial weights: {initial_weights}")
        print(f"Sigma0: {sigma0}")
        print(f"Max iterations: {max_iterations}")
        print("=" * 80)

        # Try to load checkpoint
        checkpoint_data = self._load_checkpoint(save_path) if save_path else None

        if checkpoint_data:
            es, iteration = checkpoint_data
            print(f"✅ Resumed from iteration {iteration}")
        else:
            # Run CMA-ES from scratch
            es = cma.CMAEvolutionStrategy(initial_weights, sigma0, opts)
            iteration = 0
            print("🆕 Starting new optimization")
        while not es.stop():
            solutions = es.ask()
            fitness_values = [self.objective_function(x) for x in solutions]
            es.tell(solutions, fitness_values)

            # Log iteration
            iteration += 1
            best_fitness = -es.result.fbest  # Convert back to positive score
            best_weights = es.result.xbest

            self.history["iterations"].append(iteration)
            self.history["best_weights"].append(best_weights.tolist())
            self.history["best_scores"].append(best_fitness)

            print(f"\nIteration {iteration}:")
            print(f"  Best score: {best_fitness:.4f}")
            print(f"  Best weights: {best_weights}")

            es.disp()

            # Save checkpoint
            if save_path:
                self._save_checkpoint(es, iteration, save_path)

        # Final results
        best_weights = es.result.xbest
        best_score = -es.result.fbest

        results = {
            "persona_id": self.persona_id,
            "model": self.model_name,
            "layer": self.layer,
            "alpha": self.alpha,
            "best_weights": {
                trait: float(w) for trait, w in zip(self.trait_names, best_weights)
            },
            "best_score": float(best_score),
            "num_iterations": iteration,
            "num_evaluations": len(self.history["all_scores"]),
            "optimization_history": self.history,
            "timestamp": datetime.now().isoformat()
        }

        # Save results
        if save_path:
            results_file = save_path / f"{self.persona_id}_optimization.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n✅ Results saved to: {results_file}")

            # Save best weights as separate file
            weights_file = save_path / f"{self.persona_id}_best_weights.json"
            with open(weights_file, 'w') as f:
                json.dump(results["best_weights"], f, indent=2)
            print(f"✅ Best weights saved to: {weights_file}")

            # Remove checkpoint file after successful completion (unless keep_checkpoint=True)
            if not keep_checkpoint:
                checkpoint_file = save_path / f"{self.persona_id}_checkpoint.pkl"
                if checkpoint_file.exists():
                    checkpoint_file.unlink()
                    print(f"🗑️  Checkpoint removed: {checkpoint_file}")
            else:
                print(f"💾 Checkpoint kept for testing: {save_path / f'{self.persona_id}_checkpoint.pkl'}")

        print("\n" + "=" * 80)
        print("OPTIMIZATION COMPLETE")
        print("=" * 80)
        print(f"Best persona fit score: {best_score:.4f}")
        print(f"Best weights:")
        for trait, w in results["best_weights"].items():
            print(f"  {trait}: {w:.4f}")
        print("=" * 80)

        return results


def optimize_persona_weights(
    persona_id: str,
    layer: int = 20,
    max_iterations: int = 50,
    population_size: Optional[int] = None,
    sigma0: float = 1.0,
    save_dir: str = "optimization_results",
    keep_checkpoint: bool = False,
    **kwargs
) -> Dict:
    """
    Convenience function to optimize persona weights.

    Args:
        persona_id: Persona identifier
        layer: Layer to apply steering
        max_iterations: Maximum CMA-ES iterations
        population_size: CMA-ES population size
        sigma0: Initial CMA-ES sigma
        save_dir: Directory to save results
        keep_checkpoint: Keep checkpoint file after completion (default: False)
        **kwargs: Additional arguments for CMAESPersonaOptimizer

    Returns:
        Optimization results
    """
    optimizer = CMAESPersonaOptimizer(
        persona_id=persona_id,
        layer=layer,
        **kwargs
    )

    results = optimizer.optimize(
        max_iterations=max_iterations,
        population_size=population_size,
        sigma0=sigma0,
        save_dir=save_dir,
        keep_checkpoint=keep_checkpoint
    )

    return results
