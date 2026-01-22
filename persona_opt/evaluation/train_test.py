"""
Train/Test Split Evaluation for persona steering.

Evaluates generalization by splitting prompts into train/test sets.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

from .utils import (
    load_optimization_results,
    load_steering_vectors,
    build_combined_steering_vector,
    load_prompts,
    train_test_split,
    save_evaluation_results,
    EvaluationConfig
)


class TrainTestEvaluator:
    """Evaluates persona steering with train/test split."""

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
        self.alpha = self.opt_results.get('alpha', 2.0)
        self.optimized_layer = self.opt_results.get('layer', config.layer)

    def evaluate(
        self,
        prompts: List[str],
        train_ratio: float = 0.7,
        seed: int = 42,
        output_dir: Optional[str] = None
    ) -> Dict:
        """
        Run train/test split evaluation.

        Args:
            prompts: List of evaluation prompts
            train_ratio: Ratio of training prompts
            seed: Random seed
            output_dir: Output directory for results

        Returns:
            Results dictionary
        """
        print(f"[TrainTest] Evaluating {self.config.persona_id}")
        print(f"[TrainTest] Total prompts: {len(prompts)}")
        print(f"[TrainTest] Train ratio: {train_ratio}")

        # Split prompts
        train_prompts, test_prompts = train_test_split(prompts, train_ratio, seed)
        print(f"[TrainTest] Train: {len(train_prompts)}, Test: {len(test_prompts)}")

        # Load steering vectors for optimized layer
        trait_vectors = {}
        traits = ["R1", "R2", "R3", "R4", "R5"]
        for trait in traits:
            vector_file = Path(self.vectors_dir) / trait / f"layer{self.optimized_layer}_svd.pt"
            import torch
            vector_data = torch.load(vector_file, map_location='cpu')
            # Handle different formats
            if isinstance(vector_data, torch.Tensor):
                trait_vectors[trait] = vector_data
            elif isinstance(vector_data, dict) and 'vector' in vector_data:
                trait_vectors[trait] = vector_data['vector']
            else:
                raise ValueError(f"Unknown vector format in {vector_file}")

        # Build combined steering vector
        steering_vector = build_combined_steering_vector(self.weights, trait_vectors)

        # Generate baseline responses
        print(f"[TrainTest] Generating baseline responses...")
        baseline_train = self.steerer.batch_generate(train_prompts)
        baseline_test = self.steerer.batch_generate(test_prompts)

        # Generate steered responses
        print(f"[TrainTest] Generating steered responses...")
        steered_train = self.steerer.batch_generate(
            train_prompts,
            steering_vector=steering_vector,
            layer=self.optimized_layer,
            alpha=self.alpha
        )
        steered_test = self.steerer.batch_generate(
            test_prompts,
            steering_vector=steering_vector,
            layer=self.optimized_layer,
            alpha=self.alpha
        )

        # Evaluate train set
        print(f"[TrainTest] Evaluating train set...")
        train_results = self.evaluator.batch_evaluate(
            prompts=train_prompts,
            baseline_responses=baseline_train,
            steered_responses=steered_train
        )

        # Evaluate test set
        print(f"[TrainTest] Evaluating test set...")
        test_results = self.evaluator.batch_evaluate(
            prompts=test_prompts,
            baseline_responses=baseline_test,
            steered_responses=steered_test
        )

        # Compute statistics
        train_scores = [r['persona_fit'] for r in train_results]
        test_scores = [r['persona_fit'] for r in test_results]

        train_mean = np.mean(train_scores)
        test_mean = np.mean(test_scores)
        train_std = np.std(train_scores)
        test_std = np.std(test_scores)

        generalization_gap = train_mean - test_mean

        results = {
            'evaluation_type': 'train_test_split',
            'persona_id': self.config.persona_id,
            'layer': self.optimized_layer,
            'alpha': self.alpha,
            'timestamp': datetime.now().isoformat(),
            'config': {
                'train_ratio': train_ratio,
                'seed': seed,
                'n_train': len(train_prompts),
                'n_test': len(test_prompts)
            },
            'weights': self.weights,
            'train': {
                'mean_score': float(train_mean),
                'std_score': float(train_std),
                'scores': train_scores
            },
            'test': {
                'mean_score': float(test_mean),
                'std_score': float(test_std),
                'scores': test_scores
            },
            'summary': {
                'train_mean': f"{train_mean:.2f}",
                'test_mean': f"{test_mean:.2f}",
                'generalization_gap': f"{generalization_gap:.3f}",
                'overfitting': "Yes" if generalization_gap > 0.5 else "No"
            }
        }

        # Generate plots
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            fig_path = self._plot_results(results, output_path)
            save_evaluation_results(results, output_dir, {'Train vs Test': str(fig_path)})

        return results

    def _plot_results(self, results: Dict, output_dir: Path) -> Path:
        """Generate bar plot comparing train and test scores."""
        fig, ax = plt.subplots(figsize=(8, 6))

        categories = ['Train', 'Test']
        means = [results['train']['mean_score'], results['test']['mean_score']]
        stds = [results['train']['std_score'], results['test']['std_score']]

        x = np.arange(len(categories))
        bars = ax.bar(x, means, yerr=stds, capsize=5, color=['#3498db', '#e74c3c'], alpha=0.7)

        ax.set_ylabel('Persona Fit Score', fontsize=12)
        ax.set_title(f'Train/Test Evaluation: {results["persona_id"]}', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=12)
        ax.set_ylim(0, 5.5)
        ax.axhline(y=5.0, color='green', linestyle='--', alpha=0.5, label='Perfect Score')
        ax.grid(axis='y', alpha=0.3)
        ax.legend()

        # Add value labels
        for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
            ax.text(bar.get_x() + bar.get_width()/2, mean + std + 0.1,
                   f'{mean:.2f}±{std:.2f}',
                   ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        fig_path = output_dir / 'plot.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()

        return fig_path
