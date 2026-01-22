"""
Alpha Sensitivity Evaluation for persona steering.

Tests robustness across different steering strengths (alpha values).
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch

from .utils import (
    load_optimization_results,
    load_steering_vectors,
    build_combined_steering_vector,
    save_evaluation_results,
    EvaluationConfig
)


class AlphaSensitivityEvaluator:
    """Evaluates persona steering across different alpha values."""

    def __init__(
        self,
        config: EvaluationConfig,
        steerer,
        evaluator,
        optimization_dir: str = "persona-opt",
        vectors_dir: str = "data/steering_vectors_v2"
    ):
        """
        Initialize evaluator.

        Args:
            config: Evaluation configuration
            steerer: Llama3ActivationSteerer instance
            evaluator: PersonaAwareEvaluator instance
            optimization_dir: Directory with optimization results
            vectors_dir: Directory with steering vectors
        """
        self.config = config
        self.steerer = steerer
        self.evaluator = evaluator
        self.optimization_dir = optimization_dir
        self.vectors_dir = vectors_dir

        # Load optimization results
        self.opt_results = load_optimization_results(config.persona_id, optimization_dir)
        self.weights = self.opt_results['weights']
        self.optimized_alpha = self.opt_results.get('alpha', 2.0)
        self.optimized_layer = self.opt_results.get('layer', config.layer)

    def evaluate(
        self,
        prompts: List[str],
        alpha_values: List[float] = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        output_dir: Optional[str] = None
    ) -> Dict:
        """
        Run alpha sensitivity evaluation.

        Args:
            prompts: List of evaluation prompts
            alpha_values: Alpha values to test
            output_dir: Output directory for results

        Returns:
            Results dictionary
        """
        print(f"[AlphaSensitivity] Evaluating {self.config.persona_id}")
        print(f"[AlphaSensitivity] Testing alpha values: {alpha_values}")
        print(f"[AlphaSensitivity] Optimized alpha: {self.optimized_alpha}")

        # Load steering vectors for optimized layer
        trait_vectors = {}
        traits = ["R1", "R2", "R3", "R4", "R5"]
        for trait in traits:
            vector_file = Path(self.vectors_dir) / trait / f"layer{self.optimized_layer}_svd.pt"
            vector_data = torch.load(vector_file, map_location='cpu')
            # Handle different formats - check tensor first
            if isinstance(vector_data, torch.Tensor):
                trait_vectors[trait] = vector_data
            elif isinstance(vector_data, dict) and 'vector' in vector_data:
                trait_vectors[trait] = vector_data['vector']
            else:
                raise ValueError(f"Unexpected vector format in {vector_file}")

        # Build combined steering vector
        steering_vector = build_combined_steering_vector(self.weights, trait_vectors)

        # Generate baseline responses once
        print(f"[AlphaSensitivity] Generating baseline responses...")
        baseline_responses = self.steerer.batch_generate(prompts)

        # Evaluate each alpha
        alpha_results = {}
        for alpha in alpha_values:
            print(f"\n[AlphaSensitivity] Testing alpha = {alpha}...")

            # Generate steered responses
            steered_responses = self.steerer.batch_generate(
                prompts,
                steering_vector=steering_vector,
                layer=self.optimized_layer,
                alpha=alpha
            )

            # Evaluate
            eval_results = self.evaluator.batch_evaluate(
                prompts=prompts,
                baseline_responses=baseline_responses,
                steered_responses=steered_responses
            )

            # Compute statistics
            scores = [r['persona_fit'] for r in eval_results]
            win_rate = sum(1 for r in eval_results if r['winner'] == 'steered') / len(eval_results)

            alpha_results[alpha] = {
                'mean_score': float(np.mean(scores)),
                'std_score': float(np.std(scores)),
                'win_rate': float(win_rate),
                'scores': scores
            }

            print(f"  Alpha {alpha}: Score = {alpha_results[alpha]['mean_score']:.2f}, Win Rate = {win_rate:.2%}")

        # Find best alpha
        best_alpha = max(alpha_results.keys(), key=lambda a: alpha_results[a]['mean_score'])
        best_score = alpha_results[best_alpha]['mean_score']

        # Compute sensitivity metrics
        scores_list = [alpha_results[a]['mean_score'] for a in alpha_values]
        score_variance = float(np.var(scores_list))
        score_range = float(max(scores_list) - min(scores_list))

        results = {
            'evaluation_type': 'alpha_sensitivity',
            'persona_id': self.config.persona_id,
            'layer': self.optimized_layer,
            'optimized_alpha': self.optimized_alpha,
            'timestamp': datetime.now().isoformat(),
            'weights': self.weights,
            'alpha_values': alpha_values,
            'alpha_results': alpha_results,
            'sensitivity_metrics': {
                'score_variance': score_variance,
                'score_range': score_range,
                'stable': score_range < 0.5
            },
            'summary': {
                'optimized_alpha': self.optimized_alpha,
                'optimized_score': f"{alpha_results.get(self.optimized_alpha, {}).get('mean_score', 0):.2f}",
                'best_alpha': best_alpha,
                'best_score': f"{best_score:.2f}",
                'score_range': f"{score_range:.2f}",
                'robust': "Yes" if score_range < 0.5 else "No"
            }
        }

        # Generate plots
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            fig_path = self._plot_results(results, output_path)
            save_evaluation_results(results, output_dir, {'Alpha Sensitivity': str(fig_path)})

        return results

    def _plot_results(self, results: Dict, output_dir: Path) -> Path:
        """Generate plot of alpha sensitivity."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        alphas = results['alpha_values']
        alpha_results = results['alpha_results']

        # Plot 1: Persona Fit Score vs Alpha
        scores = [alpha_results[a]['mean_score'] for a in alphas]
        stds = [alpha_results[a]['std_score'] for a in alphas]

        ax1.plot(alphas, scores, marker='o', linewidth=2, markersize=8, color='#3498db')
        ax1.fill_between(alphas,
                         [s - std for s, std in zip(scores, stds)],
                         [s + std for s, std in zip(scores, stds)],
                         alpha=0.2, color='#3498db')
        ax1.axvline(x=results['optimized_alpha'], color='red', linestyle='--',
                   label=f'Optimized α = {results["optimized_alpha"]}', alpha=0.7)
        ax1.axhline(y=5.0, color='green', linestyle=':', alpha=0.5, label='Perfect Score')
        ax1.set_xlabel('Alpha (Steering Strength)', fontsize=12)
        ax1.set_ylabel('Persona Fit Score', fontsize=12)
        ax1.set_title('Alpha Sensitivity: Persona Fit', fontsize=14, fontweight='bold')
        ax1.set_ylim(0, 5.5)
        ax1.grid(alpha=0.3)
        ax1.legend()

        # Plot 2: Win Rate vs Alpha
        win_rates = [alpha_results[a]['win_rate'] * 100 for a in alphas]

        ax2.plot(alphas, win_rates, marker='s', linewidth=2, markersize=8, color='#e74c3c')
        ax2.axvline(x=results['optimized_alpha'], color='red', linestyle='--',
                   label=f'Optimized α = {results["optimized_alpha"]}', alpha=0.7)
        ax2.axhline(y=50, color='gray', linestyle=':', alpha=0.5, label='Baseline')
        ax2.set_xlabel('Alpha (Steering Strength)', fontsize=12)
        ax2.set_ylabel('Win Rate (%)', fontsize=12)
        ax2.set_title('Alpha Sensitivity: Win Rate', fontsize=14, fontweight='bold')
        ax2.set_ylim(0, 110)
        ax2.grid(alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        fig_path = output_dir / 'plot.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()

        return fig_path
