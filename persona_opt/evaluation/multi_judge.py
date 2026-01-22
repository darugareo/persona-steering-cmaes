"""
Multi-Judge Reliability Evaluation.

Tests inter-judge agreement across different LLM judges.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr, pearsonr

from .utils import (
    load_optimization_results,
    build_combined_steering_vector,
    save_evaluation_results,
    EvaluationConfig
)


class MultiJudgeEvaluator:
    """Evaluates persona steering with multiple judge models."""

    def __init__(
        self,
        config: EvaluationConfig,
        steerer,
        optimization_dir: str = "persona-opt",
        vectors_dir: str = "data/steering_vectors_v2"
    ):
        """
        Initialize evaluator.

        Args:
            config: Evaluation configuration
            steerer: Llama3ActivationSteerer instance
            optimization_dir: Directory with optimization results
            vectors_dir: Directory with steering vectors
        """
        self.config = config
        self.steerer = steerer
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
        judge_models: List[str] = ["gpt-4o-mini", "gpt-4o", "claude-3-5-sonnet-20241022"],
        output_dir: Optional[str] = None
    ) -> Dict:
        """
        Run multi-judge evaluation.

        Args:
            prompts: List of evaluation prompts
            judge_models: List of judge model names
            output_dir: Output directory for results

        Returns:
            Results dictionary
        """
        print(f"[MultiJudge] Evaluating {self.config.persona_id}")
        print(f"[MultiJudge] Judge models: {judge_models}")
        print(f"[MultiJudge] Prompts: {len(prompts)}")

        # Load steering vectors
        trait_vectors = {}
        traits = ["R1", "R2", "R3", "R4", "R5"]
        for trait in traits:
            vector_file = Path(self.vectors_dir) / trait / f"layer{self.optimized_layer}_svd.pt"
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

        # Generate baseline and steered responses once
        print(f"[MultiJudge] Generating baseline responses...")
        baseline_responses = self.steerer.batch_generate(prompts)

        print(f"[MultiJudge] Generating steered responses...")
        steered_responses = self.steerer.batch_generate(
            prompts,
            steering_vector=steering_vector,
            layer=self.optimized_layer,
            alpha=self.alpha
        )

        # Evaluate with each judge
        judge_results = {}
        for judge_model in judge_models:
            print(f"\n[MultiJudge] Evaluating with {judge_model}...")

            # Create evaluator for this judge
            from persona_opt.evaluator import PersonaAwareEvaluator
            from persona_opt.evaluation.utils import load_persona_profile

            persona_profile = load_persona_profile(self.config.persona_id)
            evaluator = PersonaAwareEvaluator(
                persona_profile=persona_profile,
                judge_model=judge_model
            )

            # Evaluate
            eval_results = evaluator.batch_evaluate(
                prompts=prompts,
                baseline_responses=baseline_responses,
                steered_responses=steered_responses
            )

            # Compute statistics
            scores = [r['persona_fit'] for r in eval_results]
            win_rate = sum(1 for r in eval_results if r['winner'] == 'steered') / len(eval_results)

            judge_results[judge_model] = {
                'mean_score': float(np.mean(scores)),
                'std_score': float(np.std(scores)),
                'win_rate': float(win_rate),
                'scores': scores,
                'judgments': eval_results
            }

            print(f"  {judge_model}: Score = {judge_results[judge_model]['mean_score']:.2f}, Win Rate = {win_rate:.2%}")

        # Compute inter-judge agreement
        agreement_metrics = self._compute_agreement(judge_results, judge_models)

        # Find most lenient and strict judges
        mean_scores = {j: judge_results[j]['mean_score'] for j in judge_models}
        most_lenient = max(mean_scores, key=mean_scores.get)
        most_strict = min(mean_scores, key=mean_scores.get)

        results = {
            'evaluation_type': 'multi_judge_reliability',
            'persona_id': self.config.persona_id,
            'layer': self.optimized_layer,
            'alpha': self.alpha,
            'timestamp': datetime.now().isoformat(),
            'weights': self.weights,
            'judge_models': judge_models,
            'judge_results': judge_results,
            'agreement_metrics': agreement_metrics,
            'summary': {
                'num_judges': len(judge_models),
                'mean_agreement': f"{agreement_metrics['mean_spearman']:.3f}",
                'most_lenient': most_lenient,
                'most_strict': most_strict,
                'reliable': "Yes" if agreement_metrics['mean_spearman'] > 0.7 else "Moderate"
            }
        }

        # Generate plots
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            fig_path = self._plot_results(results, output_path)
            save_evaluation_results(results, output_dir, {'Judge Comparison': str(fig_path)})

        return results

    def _compute_agreement(self, judge_results: Dict, judge_models: List[str]) -> Dict:
        """Compute inter-judge agreement metrics."""
        # Pairwise correlations
        pairwise_spearman = {}
        pairwise_pearson = {}

        for i, judge1 in enumerate(judge_models):
            for j, judge2 in enumerate(judge_models[i+1:], start=i+1):
                scores1 = judge_results[judge1]['scores']
                scores2 = judge_results[judge2]['scores']

                spearman_corr, spearman_p = spearmanr(scores1, scores2)
                pearson_corr, pearson_p = pearsonr(scores1, scores2)

                pair_key = f"{judge1}_vs_{judge2}"
                pairwise_spearman[pair_key] = {
                    'correlation': float(spearman_corr),
                    'p_value': float(spearman_p)
                }
                pairwise_pearson[pair_key] = {
                    'correlation': float(pearson_corr),
                    'p_value': float(pearson_p)
                }

        # Mean correlations
        mean_spearman = np.mean([v['correlation'] for v in pairwise_spearman.values()])
        mean_pearson = np.mean([v['correlation'] for v in pairwise_pearson.values()])

        return {
            'pairwise_spearman': pairwise_spearman,
            'pairwise_pearson': pairwise_pearson,
            'mean_spearman': float(mean_spearman),
            'mean_pearson': float(mean_pearson)
        }

    def _plot_results(self, results: Dict, output_dir: Path) -> Path:
        """Generate plots comparing judges."""
        judge_models = results['judge_models']
        judge_results = results['judge_results']

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Mean scores comparison
        ax1 = axes[0, 0]
        means = [judge_results[j]['mean_score'] for j in judge_models]
        stds = [judge_results[j]['std_score'] for j in judge_models]
        x = np.arange(len(judge_models))

        bars = ax1.bar(x, means, yerr=stds, capsize=5, color='#3498db', alpha=0.7)
        ax1.set_ylabel('Mean Persona Fit Score', fontsize=11)
        ax1.set_title('Judge Comparison: Mean Scores', fontsize=12, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels([j.split('-')[0] for j in judge_models], fontsize=10)
        ax1.set_ylim(0, 5.5)
        ax1.axhline(y=5.0, color='green', linestyle='--', alpha=0.5)
        ax1.grid(axis='y', alpha=0.3)

        # Plot 2: Win rates comparison
        ax2 = axes[0, 1]
        win_rates = [judge_results[j]['win_rate'] * 100 for j in judge_models]

        bars = ax2.bar(x, win_rates, color='#e74c3c', alpha=0.7)
        ax2.set_ylabel('Win Rate (%)', fontsize=11)
        ax2.set_title('Judge Comparison: Win Rates', fontsize=12, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels([j.split('-')[0] for j in judge_models], fontsize=10)
        ax2.set_ylim(0, 110)
        ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
        ax2.grid(axis='y', alpha=0.3)

        # Plot 3: Score distributions
        ax3 = axes[1, 0]
        score_data = [judge_results[j]['scores'] for j in judge_models]
        bp = ax3.boxplot(score_data, labels=[j.split('-')[0] for j in judge_models], patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('#95a5a6')
            patch.set_alpha(0.7)
        ax3.set_ylabel('Persona Fit Score', fontsize=11)
        ax3.set_title('Score Distribution by Judge', fontsize=12, fontweight='bold')
        ax3.set_ylim(0, 5.5)
        ax3.grid(axis='y', alpha=0.3)

        # Plot 4: Inter-judge correlation heatmap
        ax4 = axes[1, 1]
        n_judges = len(judge_models)
        corr_matrix = np.zeros((n_judges, n_judges))

        for i in range(n_judges):
            for j in range(n_judges):
                if i == j:
                    corr_matrix[i, j] = 1.0
                else:
                    judge1, judge2 = judge_models[i], judge_models[j]
                    pair_key = f"{judge1}_vs_{judge2}" if i < j else f"{judge2}_vs_{judge1}"
                    if pair_key in results['agreement_metrics']['pairwise_spearman']:
                        corr_matrix[i, j] = results['agreement_metrics']['pairwise_spearman'][pair_key]['correlation']
                    else:
                        corr_matrix[i, j] = corr_matrix[j, i]

        im = ax4.imshow(corr_matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
        ax4.set_xticks(np.arange(n_judges))
        ax4.set_yticks(np.arange(n_judges))
        ax4.set_xticklabels([j.split('-')[0] for j in judge_models], fontsize=9)
        ax4.set_yticklabels([j.split('-')[0] for j in judge_models], fontsize=9)
        ax4.set_title('Inter-Judge Spearman Correlation', fontsize=12, fontweight='bold')

        # Add correlation values
        for i in range(n_judges):
            for j in range(n_judges):
                text = ax4.text(j, i, f'{corr_matrix[i, j]:.2f}',
                              ha="center", va="center", color="black", fontsize=9)

        plt.colorbar(im, ax=ax4)

        plt.tight_layout()
        fig_path = output_dir / 'plot.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()

        return fig_path
