"""
Cross-Layer Transfer Evaluation for persona steering.

Tests whether optimized weights transfer across different layers.
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


class CrossLayerEvaluator:
    """Evaluates persona steering across different layers."""

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
        test_layers: List[int] = [20, 21, 22, 23, 24],
        output_dir: Optional[str] = None
    ) -> Dict:
        """
        Run cross-layer evaluation.

        Args:
            prompts: List of evaluation prompts
            test_layers: Layers to test
            output_dir: Output directory for results

        Returns:
            Results dictionary
        """
        print(f"[CrossLayer] Evaluating {self.config.persona_id}")
        print(f"[CrossLayer] Testing layers: {test_layers}")
        print(f"[CrossLayer] Optimized weights from layer {self.optimized_layer}")

        # Load all steering vectors
        all_vectors = load_steering_vectors(test_layers, vectors_dir=self.vectors_dir)

        # Generate baseline responses once
        print(f"[CrossLayer] Generating baseline responses...")
        baseline_responses = self.steerer.batch_generate(prompts)

        # Evaluate each layer
        layer_results = {}
        for layer in test_layers:
            print(f"\n[CrossLayer] Testing layer {layer}...")

            # Build steering vector for this layer
            trait_vectors = {trait: all_vectors[trait][layer] for trait in ["R1", "R2", "R3", "R4", "R5"]}
            steering_vector = build_combined_steering_vector(self.weights, trait_vectors)

            # Generate steered responses
            steered_responses = self.steerer.batch_generate(
                prompts,
                steering_vector=steering_vector,
                layer=layer,
                alpha=self.alpha
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

            layer_results[layer] = {
                'mean_score': float(np.mean(scores)),
                'std_score': float(np.std(scores)),
                'win_rate': float(win_rate),
                'scores': scores
            }

            print(f"  Layer {layer}: Score = {layer_results[layer]['mean_score']:.2f}, Win Rate = {win_rate:.2%}")

        # Find best layer
        best_layer = max(layer_results.keys(), key=lambda l: layer_results[l]['mean_score'])
        best_score = layer_results[best_layer]['mean_score']

        results = {
            'evaluation_type': 'cross_layer_transfer',
            'persona_id': self.config.persona_id,
            'optimized_layer': self.optimized_layer,
            'alpha': self.alpha,
            'timestamp': datetime.now().isoformat(),
            'weights': self.weights,
            'test_layers': test_layers,
            'layer_results': layer_results,
            'summary': {
                'optimized_layer': self.optimized_layer,
                'optimized_score': f"{layer_results.get(self.optimized_layer, {}).get('mean_score', 0):.2f}",
                'best_layer': best_layer,
                'best_score': f"{best_score:.2f}",
                'transferable': "Yes" if abs(best_layer - self.optimized_layer) > 1 else "Limited"
            }
        }

        # Generate plots
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            fig_path = self._plot_results(results, output_path)
            save_evaluation_results(results, output_dir, {'Cross-Layer Performance': str(fig_path)})

        return results

    def _plot_results(self, results: Dict, output_dir: Path) -> Path:
        """Generate line plot of layer performance."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        layers = results['test_layers']
        layer_results = results['layer_results']

        # Plot 1: Persona Fit Score
        scores = [layer_results[l]['mean_score'] for l in layers]
        stds = [layer_results[l]['std_score'] for l in layers]

        ax1.plot(layers, scores, marker='o', linewidth=2, markersize=8, color='#3498db')
        ax1.fill_between(layers,
                         [s - std for s, std in zip(scores, stds)],
                         [s + std for s, std in zip(scores, stds)],
                         alpha=0.2, color='#3498db')
        ax1.axvline(x=results['optimized_layer'], color='red', linestyle='--',
                   label=f'Optimized Layer ({results["optimized_layer"]})', alpha=0.7)
        ax1.set_xlabel('Layer', fontsize=12)
        ax1.set_ylabel('Persona Fit Score', fontsize=12)
        ax1.set_title('Cross-Layer Persona Fit', fontsize=14, fontweight='bold')
        ax1.set_ylim(0, 5.5)
        ax1.grid(alpha=0.3)
        ax1.legend()

        # Plot 2: Win Rate
        win_rates = [layer_results[l]['win_rate'] * 100 for l in layers]

        ax2.plot(layers, win_rates, marker='s', linewidth=2, markersize=8, color='#e74c3c')
        ax2.axvline(x=results['optimized_layer'], color='red', linestyle='--',
                   label=f'Optimized Layer ({results["optimized_layer"]})', alpha=0.7)
        ax2.axhline(y=50, color='gray', linestyle=':', alpha=0.5, label='Baseline')
        ax2.set_xlabel('Layer', fontsize=12)
        ax2.set_ylabel('Win Rate (%)', fontsize=12)
        ax2.set_title('Cross-Layer Win Rate', fontsize=14, fontweight='bold')
        ax2.set_ylim(0, 110)
        ax2.grid(alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        fig_path = output_dir / 'plot.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()

        return fig_path
