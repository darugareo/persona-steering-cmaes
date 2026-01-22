"""
CMA-ES Optimization Loop
Optimize trait vectors using Covariance Matrix Adaptation Evolution Strategy
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

from persona_opt.generator import PersonaGenerator
from persona_opt.judge import LLMJudge

# Import O-space conversion
try:
    from persona_opt.steering_space_v4 import orthogonal_to_semantic, get_trait_names
    OSPACE_AVAILABLE = True
except ImportError:
    OSPACE_AVAILABLE = False
    print("Warning: steering_space_v4 not available. Running in semantic space mode.")


class CMAESOptimizer:
    """CMA-ES optimizer for trait vectors"""

    def __init__(
        self,
        n_dims: int = 6,
        pop_size: int = 8,
        n_parents: int = 4,
        sigma0: float = 0.10,
        bounds: Tuple[float, float] = (-3.0, 3.0),
        generator_model: str = "mock",
        judge_model: str = "mock",
        log_dir: str = "logs",
        prompts_file: Optional[str] = None,
        use_ospace: bool = True
    ):
        """
        Initialize CMA-ES optimizer

        Args:
            n_dims: Number of trait dimensions (6 for O-space: O1-O6, or 5 for semantic R1-R5)
            pop_size: Population size per generation
            n_parents: Number of parents for recombination
            sigma0: Initial step size
            bounds: Trait value bounds (wider for O-space: ±3.0)
            generator_model: Generator model type
            judge_model: Judge model type
            log_dir: Directory for logs
            prompts_file: Prompts file path
            use_ospace: If True, optimize in O-space (requires orthogonal_basis_v4.npy)
        """
        self.n_dims = n_dims
        self.pop_size = pop_size
        self.n_parents = n_parents
        self.sigma0 = sigma0
        self.bounds = bounds
        self.generator_model = generator_model
        self.judge_model = judge_model
        self.use_ospace = use_ospace and OSPACE_AVAILABLE

        # Initialize CMA-ES parameters
        self.mean = np.zeros(n_dims)
        self.sigma = sigma0
        self.C = np.eye(n_dims)  # Covariance matrix
        self.pc = np.zeros(n_dims)  # Evolution path for C
        self.ps = np.zeros(n_dims)  # Evolution path for sigma

        # Strategy parameters (simplified)
        self.cc = 4 / (n_dims + 4)
        self.cs = 4 / (n_dims + 4)
        self.c1 = 2 / ((n_dims + 1.3) ** 2 + n_parents)
        self.cmu = min(1 - self.c1, 2 * (n_parents - 2 + 1 / n_parents) / ((n_dims + 2) ** 2 + n_parents))
        self.damps = 1 + 2 * max(0, np.sqrt((n_parents - 1) / (n_dims + 1)) - 1) + self.cs

        # Weights for recombination
        self.weights = np.log(n_parents + 0.5) - np.log(np.arange(1, n_parents + 1))
        self.weights /= self.weights.sum()
        self.mu_eff = 1 / (self.weights ** 2).sum()

        # Initialize components
        self.generator = PersonaGenerator(model=generator_model)
        self.judge = LLMJudge(model=judge_model)

        # Logging setup
        space_type = "ospace" if self.use_ospace else "semantic"
        self.log_dir = Path(log_dir) / f"run_{space_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Log optimization mode
        mode_file = self.log_dir / "optimization_mode.txt"
        with open(mode_file, 'w') as f:
            f.write(f"Optimization Space: {'O-space (Orthogonal)' if self.use_ospace else 'Semantic Space'}\n")
            f.write(f"Dimensions: {n_dims}\n")
            f.write(f"Bounds: {bounds}\n")
            f.write(f"Population: {pop_size}\n")
            f.write(f"Parents: {n_parents}\n")
            f.write(f"Sigma0: {sigma0}\n")

        # Test prompts (small set for evaluation)
        self.prompts_file = prompts_file
        self.test_prompts = self._load_test_prompts()

    def _load_test_prompts(self) -> List[str]:
        """Load test prompts for evaluation"""
        if self.prompts_file:
            # Load from JSON file
            with open(self.prompts_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Extract text from prompts
            if 'prompts' in data:
                return [p['text'] for p in data['prompts']]
            else:
                raise ValueError(f"Invalid prompts file format: {self.prompts_file}")

        # Fallback: hardcoded examples
        return [
            "Pythonでファイルを読み込む方法を教えてください",
            "機械学習プロジェクトを始めるには何から始めるべきですか？",
            "バグが見つかったときの対処法を教えてください",
            "コードレビューで注意すべき点は何ですか？",
            "データベース設計の基本原則を説明してください"
        ]

    def _clip_to_bounds(self, x: np.ndarray) -> np.ndarray:
        """Clip trait vector to valid bounds"""
        return np.clip(x, self.bounds[0], self.bounds[1])

    def _evaluate_candidate(
        self,
        candidate: np.ndarray,
        baseline_responses: List[str]
    ) -> Tuple[float, float, List[Dict]]:
        """
        Evaluate a single candidate trait vector

        Args:
            candidate: Trait vector to evaluate (O-space if use_ospace=True, else semantic)
            baseline_responses: Baseline responses for comparison

        Returns:
            Tuple of (win_rate, tie_rate, judgments)
        """
        # Convert candidate vector to trait dict
        if self.use_ospace:
            # candidate is in O-space → convert to semantic for generator
            semantic_vector = orthogonal_to_semantic(candidate)
            trait_names = get_trait_names()
            trait_dict = {name: float(val) for name, val in zip(trait_names, semantic_vector)}
        else:
            # candidate is already in semantic space
            trait_dict = {f'R{i+1}': float(candidate[i]) for i in range(len(candidate))}

        # Generate responses with this trait vector
        candidate_responses = []
        for prompt in self.test_prompts:
            response = self.generator.generate_response(prompt, trait_dict, model=self.generator_model)
            candidate_responses.append(response)

        # Evaluate against baseline
        win_rate, tie_rate, judgments = self.judge.evaluate_batch(
            self.test_prompts,
            baseline_responses,
            candidate_responses,
            trait_dict
        )

        return win_rate, tie_rate, judgments

    def _sample_population(self) -> np.ndarray:
        """Sample population from current distribution"""
        population = []
        for _ in range(self.pop_size):
            # Sample from multivariate normal
            z = np.random.randn(self.n_dims)
            x = self.mean + self.sigma * (self.C @ z)
            x_clipped = self._clip_to_bounds(x)
            population.append(x_clipped)
        return np.array(population)

    def _update_distribution(self, population: np.ndarray, fitness: np.ndarray):
        """Update CMA-ES distribution parameters"""
        # Sort by fitness (descending)
        idx_sorted = np.argsort(-fitness)
        selected = population[idx_sorted[:self.n_parents]]

        # Update mean
        mean_old = self.mean.copy()
        self.mean = selected.T @ self.weights

        # Update evolution paths
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mu_eff) * (self.mean - mean_old) / self.sigma
        hsig = (np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs) ** (2 * (len(fitness) + 1))) <
                (1.4 + 2 / (self.n_dims + 1)) * np.sqrt(self.n_dims))

        self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mu_eff) * (self.mean - mean_old) / self.sigma

        # Update covariance matrix
        artmp = (selected - mean_old) / self.sigma
        self.C = ((1 - self.c1 - self.cmu) * self.C +
                  self.c1 * np.outer(self.pc, self.pc) +
                  self.cmu * artmp.T @ np.diag(self.weights) @ artmp)

        # Update step size
        self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / np.sqrt(self.n_dims) - 1))

    def optimize(
        self,
        n_generations: int = 10,
        tau: float = 0.8,
        verbose: bool = True,
        early_stop_window: int = 3,
        early_stop_threshold: float = 0.01
    ) -> Dict:
        """
        Run CMA-ES optimization

        Args:
            n_generations: Number of generations to run
            tau: Win rate tie-breaking parameter (TIE counts as 0.5 * tau)
            verbose: Print progress
            early_stop_window: Number of generations to check for convergence (default: 3)
            early_stop_threshold: Min improvement threshold to continue (default: 0.01)

        Returns:
            Optimization results dictionary
        """
        if verbose:
            print(f"Starting CMA-ES optimization")
            print(f"Generations: {n_generations}, Pop size: {self.pop_size}")
            print(f"Log dir: {self.log_dir}")

        # Generate baseline responses (neutral traits)
        baseline_traits = {f'R{i+1}': 0.0 for i in range(self.n_dims)}
        baseline_responses = [
            self.generator.generate_response(p, baseline_traits, model=self.generator_model)
            for p in self.test_prompts
        ]

        # Optimization loop
        best_fitness = -np.inf
        best_candidate = None
        history = []

        for gen in range(n_generations):
            gen_start_time = time.time()

            # Sample population
            population = self._sample_population()

            # Evaluate each candidate
            fitness_list = []
            gen_results = []

            for i, candidate in enumerate(population):
                win_rate, tie_rate, judgments = self._evaluate_candidate(candidate, baseline_responses)

                # Fitness = win_rate + tie_rate * tau
                fitness = win_rate + tie_rate * tau
                fitness_list.append(fitness)

                result = {
                    "generation": gen,
                    "candidate_id": i,
                    "traits": {f'R{j+1}': float(candidate[j]) for j in range(len(candidate))},
                    "win_rate": win_rate,
                    "tie_rate": tie_rate,
                    "fitness": fitness,
                    "judgments": judgments
                }
                gen_results.append(result)

                if fitness > best_fitness:
                    best_fitness = fitness
                    best_candidate = candidate.copy()

            fitness_array = np.array(fitness_list)

            # Update CMA-ES distribution
            self._update_distribution(population, fitness_array)

            # Log generation results
            self._log_generation(gen, gen_results)

            gen_time = time.time() - gen_start_time

            if verbose:
                mean_fitness = fitness_array.mean()
                max_fitness = fitness_array.max()
                print(f"Gen {gen:3d}: mean_fit={mean_fitness:.3f}, max_fit={max_fitness:.3f}, "
                      f"sigma={self.sigma:.4f}, time={gen_time:.1f}s")

            history.append({
                "generation": gen,
                "mean_fitness": float(mean_fitness),
                "max_fitness": float(max_fitness),
                "best_fitness": float(best_fitness),
                "sigma": float(self.sigma),
                "time": gen_time
            })

            # Early stopping check
            if gen >= early_stop_window:
                recent_means = [h["mean_fitness"] for h in history[-early_stop_window:]]
                fitness_improvement = recent_means[-1] - recent_means[0]

                if fitness_improvement < early_stop_threshold:
                    if verbose:
                        print(f"\n⚠️  Early stopping at generation {gen}")
                        print(f"   Recent improvement: {fitness_improvement:.4f} < threshold {early_stop_threshold:.4f}")
                        print(f"   Recent mean fitness: {recent_means}")
                    break

        # Save final results
        final_results = {
            "best_traits": {f'R{i+1}': float(best_candidate[i]) for i in range(len(best_candidate))},
            "best_fitness": float(best_fitness),
            "history": history,
            "config": {
                "n_generations": n_generations,
                "actual_generations": len(history),
                "pop_size": self.pop_size,
                "n_parents": self.n_parents,
                "sigma0": self.sigma0,
                "tau": tau,
                "early_stop_window": early_stop_window,
                "early_stop_threshold": early_stop_threshold
            }
        }

        with open(self.log_dir / "final_results.json", 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)

        if verbose:
            print(f"\nOptimization complete!")
            print(f"Generations run: {len(history)}/{n_generations}")
            print(f"Best fitness: {best_fitness:.3f}")
            print(f"Best traits: {final_results['best_traits']}")

        # Store best solution for later access
        self.best_solution = best_candidate
        self.best_score = best_fitness

        return best_candidate, best_fitness

    def _log_generation(self, gen: int, results: List[Dict]):
        """Log generation results to JSONL"""
        log_file = self.log_dir / f"gen{gen:03d}.jsonl"
        with open(log_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description="Run CMA-ES trait optimization")
    parser.add_argument('--gens', type=int, default=10, help='Number of generations')
    parser.add_argument('--pop', type=int, default=8, help='Population size')
    parser.add_argument('--parents', type=int, default=4, help='Number of parents')
    parser.add_argument('--sigma0', type=float, default=0.10, help='Initial step size')
    parser.add_argument('--tau', type=float, default=0.80, help='TIE weight parameter')
    parser.add_argument('--max_new_tokens', type=int, default=256, help='Max tokens (not used in mock)')
    parser.add_argument('--generator_model', type=str, default='mock', help='Generator model')
    parser.add_argument('--judge_model', type=str, default='mock', help='Judge model')
    parser.add_argument('--prompts_file', type=str, default=None, help='Path to prompts JSON file')
    parser.add_argument('--use_ospace', type=lambda x: x.lower() == 'true', default=True,
                        help='Use O-space optimization (default: True)')

    args = parser.parse_args()

    # Determine dimensions and bounds based on space type
    if args.use_ospace:
        n_dims = 6  # O1-O6
        bounds = (-3.0, 3.0)
    else:
        n_dims = 5  # R1-R5
        bounds = (-1.0, 1.0)

    optimizer = CMAESOptimizer(
        n_dims=n_dims,
        pop_size=args.pop,
        n_parents=args.parents,
        sigma0=args.sigma0,
        bounds=bounds,
        generator_model=args.generator_model,
        judge_model=args.judge_model,
        prompts_file=args.prompts_file,
        use_ospace=args.use_ospace
    )

    results = optimizer.optimize(
        n_generations=args.gens,
        tau=args.tau,
        verbose=True
    )


if __name__ == "__main__":
    main()
