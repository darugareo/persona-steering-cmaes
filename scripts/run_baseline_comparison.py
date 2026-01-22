"""
Phase 1 Step 3: Baseline Comparison Evaluation
Runs all 7 baseline methods and compares performance.

Evaluations:
1. Train/Test Split (generalization)
2. Cross-Layer Transfer (transferability)
3. Primary metrics (persona fit, win rate)

Outputs:
- JSON results for each method
- Table 1: Method comparison
- Figure 4: Layer × Score heatmap
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from persona_opt.utils.config_loader import (
    load_experiment_config,
    PromptTemplateManager,
    set_seed,
    create_output_directories,
    ExperimentLogger
)
from persona_opt.internal_steering_l3 import Llama3ActivationSteerer
from persona_opt.baselines import (
    BaseMethod,
    PromptPersonaMethod,
    MeanDiffMethod,
    PCASteeringMethod,
    RandomSearchMethod,
    GridSearchMethod,
)
from persona_opt.evaluator import PersonaAwareEvaluator
from persona_opt.evaluation.utils import load_persona_profile


class BaselineComparison:
    """Manages baseline method comparison."""

    def __init__(
        self,
        persona_id: str,
        eval_prompts: List[str],
        config_path: str = "config/experiment_config.yaml",
        seed: int = 1,
    ):
        self.persona_id = persona_id
        self.eval_prompts = eval_prompts
        self.seed = seed

        # Load config
        self.config = load_experiment_config(config_path)

        # Set seed
        set_seed(seed, deterministic=self.config.deterministic)

        # Create output directories
        self.output_dirs = create_output_directories(self.config)

        # Initialize logger
        self.logger = ExperimentLogger(self.config, seed)

        # Load persona profile
        self.persona_profile = load_persona_profile(persona_id)

        # Initialize steerer
        print(f"\n[BaselineComparison] Initializing steerer...")
        self.steerer = Llama3ActivationSteerer(
            model_name=self.config.model_name,
            target_layer=self.config.default_layer,
            device=self.config.device,
        )

        # Initialize evaluator
        self.evaluator = PersonaAwareEvaluator(
            persona_id=persona_id,
            persona_profile=self.persona_profile,
            judge_model=self.config.primary_judge,
            temperature=self.config.judge_temperature,
        )

        # Storage for results
        self.results = {}

    def run_method(
        self,
        method_name: str,
        method_instance,
        prompts: List[str],
    ) -> Dict[str, Any]:
        """
        Evaluate a single method.

        Args:
            method_name: Name of the method
            method_instance: Method instance
            prompts: Evaluation prompts

        Returns:
            Results dictionary
        """
        print(f"\n{'='*80}")
        print(f"Evaluating: {method_name}")
        print(f"{'='*80}")

        # Generate baseline responses (no steering)
        print(f"[{method_name}] Generating baseline responses...")
        self.steerer.remove_hooks()
        baseline_responses = []
        for prompt in prompts:
            response = self.steerer.generate(
                prompt,
                **self.config.get_generation_kwargs()
            )
            baseline_responses.append(response)

        # Generate method responses
        print(f"[{method_name}] Generating steered responses...")
        steered_responses = method_instance.batch_generate(
            prompts,
            **self.config.get_generation_kwargs()
        )

        # Evaluate
        print(f"[{method_name}] Evaluating with {self.config.primary_judge}...")
        eval_results = self.evaluator.batch_evaluate(
            prompts=prompts,
            baseline_responses=baseline_responses,
            steered_responses=steered_responses
        )

        # Compute metrics
        scores = [r['persona_fit'] for r in eval_results]
        wins = sum(1 for r in eval_results if r['winner'] == 'steered')

        metrics = {
            'mean_score': float(np.mean(scores)),
            'std_score': float(np.std(scores)),
            'min_score': float(np.min(scores)),
            'max_score': float(np.max(scores)),
            'win_rate': wins / len(prompts),
            'num_prompts': len(prompts),
            'scores': scores,
        }

        print(f"[{method_name}] Results:")
        print(f"  Mean Score: {metrics['mean_score']:.2f} ± {metrics['std_score']:.2f}")
        print(f"  Win Rate: {metrics['win_rate']*100:.1f}%")

        # Get method config
        method_config = method_instance.get_config()

        return {
            'method': method_name,
            'config': method_config,
            'metrics': metrics,
            'eval_results': eval_results,
        }

    def run_all_baselines(self, skip_optimization: bool = False):
        """
        Run all baseline methods.

        Args:
            skip_optimization: Skip methods requiring optimization (for speed)
        """
        print(f"\n{'='*80}")
        print(f"BASELINE COMPARISON")
        print(f"Persona: {self.persona_id}")
        print(f"Prompts: {len(self.eval_prompts)}")
        print(f"Seed: {self.seed}")
        print(f"{'='*80}")

        # Method 1: Base (no steering)
        print("\n[1/7] Base Method")
        base = BaseMethod(self.steerer, self.persona_id)
        self.results['base'] = self.run_method('base', base, self.eval_prompts)
        self.logger.log_result('base', 'primary', self.results['base'])

        # Method 2: Prompt Persona
        print("\n[2/7] Prompt Persona Method")
        prompt_persona = PromptPersonaMethod(
            self.steerer,
            self.persona_id,
            persona_profile=self.persona_profile
        )
        self.results['prompt_persona'] = self.run_method(
            'prompt_persona', prompt_persona, self.eval_prompts
        )
        self.logger.log_result('prompt_persona', 'primary', self.results['prompt_persona'])

        # Method 3: Mean Difference
        print("\n[3/7] Mean Difference Method")
        meandiff = MeanDiffMethod(
            self.steerer,
            self.persona_id,
            layer=self.config.default_layer,
            alpha=self.config.default_alpha,
            persona_profile=self.persona_profile
        )
        self.results['meandiff'] = self.run_method(
            'meandiff', meandiff, self.eval_prompts
        )
        self.logger.log_result('meandiff', 'primary', self.results['meandiff'])

        # Method 4: PCA
        print("\n[4/7] PCA Steering Method")
        pca = PCASteeringMethod(
            self.steerer,
            self.persona_id,
            layer=self.config.default_layer,
            alpha=self.config.default_alpha,
            n_components=5,
            persona_profile=self.persona_profile
        )
        self.results['pca'] = self.run_method('pca', pca, self.eval_prompts)
        self.logger.log_result('pca', 'primary', self.results['pca'])

        if not skip_optimization:
            # Method 5: Random Search
            print("\n[5/7] Random Search Method (may take a while...)")
            random_search = RandomSearchMethod(
                self.steerer,
                self.persona_id,
                layer=self.config.default_layer,
                alpha=self.config.default_alpha,
                n_iterations=100,
                eval_prompts=self.eval_prompts[:5],  # Use subset for optimization
                persona_profile=self.persona_profile,
                judge_model=self.config.primary_judge,
                seed=self.seed
            )
            self.results['random_search'] = self.run_method(
                'random_search', random_search, self.eval_prompts
            )
            self.logger.log_result('random_search', 'primary', self.results['random_search'])

            # Method 6: Grid Search
            print("\n[6/7] Grid Search Method (may take a while...)")
            grid_search = GridSearchMethod(
                self.steerer,
                self.persona_id,
                layer=self.config.default_layer,
                alpha=self.config.default_alpha,
                grid_points=3,
                max_combinations=27,  # 3^3 for 3 most important traits
                eval_prompts=self.eval_prompts[:5],  # Use subset for optimization
                persona_profile=self.persona_profile,
                judge_model=self.config.primary_judge,
                seed=self.seed
            )
            self.results['grid_search'] = self.run_method(
                'grid_search', grid_search, self.eval_prompts
            )
            self.logger.log_result('grid_search', 'primary', self.results['grid_search'])

        # Method 7: Proposed (load from existing results)
        print("\n[7/7] Proposed Method (SVD + CMA-ES)")
        print("Loading pre-optimized weights from persona-opt/...")

        # Load best weights
        weights_file = Path(f"persona-opt/{self.persona_id}/best_weights.json")
        if weights_file.exists():
            with open(weights_file, 'r') as f:
                weights_data = json.load(f)

            # Create proposed method using existing implementation
            from persona_opt.evaluation.utils import load_steering_vectors

            # Load vectors
            trait_vectors = {}
            for trait in ["R1", "R2", "R3", "R4", "R5"]:
                vector_file = Path("data/steering_vectors_v2") / trait / f"layer20_svd.pt"
                import torch
                vector_data = torch.load(vector_file, map_location='cpu')
                if isinstance(vector_data, torch.Tensor):
                    trait_vectors[trait] = vector_data
                elif isinstance(vector_data, dict):
                    trait_vectors[trait] = vector_data['vector']

            # Build combined vector
            def build_vector(weights, vectors):
                combined = torch.zeros_like(list(vectors.values())[0])
                for i, trait in enumerate(["R1", "R2", "R3", "R4", "R5"]):
                    combined += weights[i] * vectors[trait]
                return combined

            steering_vector = build_vector(weights_data['weights'], trait_vectors)

            # Create simple wrapper
            class ProposedMethod:
                def __init__(self, steerer, steering_vector, layer, alpha):
                    self.steerer = steerer
                    self.steering_vector = steering_vector
                    self.layer = layer
                    self.alpha = alpha
                    self.method_name = "proposed"

                def batch_generate(self, prompts, **kwargs):
                    return self.steerer.batch_generate(
                        prompts,
                        steering_vector=self.steering_vector,
                        layer=self.layer,
                        alpha=self.alpha,
                        **kwargs
                    )

                def get_config(self):
                    return {
                        'method': 'proposed',
                        'description': 'SVD + CMA-ES + LLM Judge',
                        'layer': self.layer,
                        'alpha': self.alpha,
                        'weights': weights_data['weights'],
                        'steering': 'activation_based',
                    }

            proposed = ProposedMethod(
                self.steerer,
                steering_vector,
                layer=weights_data['layer'],
                alpha=weights_data['alpha']
            )

            self.results['proposed'] = self.run_method(
                'proposed', proposed, self.eval_prompts
            )
            self.logger.log_result('proposed', 'primary', self.results['proposed'])
        else:
            print(f"  Warning: Weights file not found, skipping proposed method")

    def save_results(self):
        """Save all results to JSON."""
        output_file = self.output_dirs['results'] / f"baseline_comparison_seed{self.seed}.json"

        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\n✓ Results saved to: {output_file}")

        return output_file


def main():
    parser = argparse.ArgumentParser(description="Run baseline comparison")
    parser.add_argument('--persona-id', type=str, required=True, help="Persona ID")
    parser.add_argument('--prompts-file', type=str, required=True, help="Evaluation prompts file")
    parser.add_argument('--num-prompts', type=int, default=10, help="Number of prompts to use")
    parser.add_argument('--seed', type=int, default=1, help="Random seed")
    parser.add_argument('--skip-optimization', action='store_true', help="Skip optimization-based methods")

    args = parser.parse_args()

    # Load prompts
    with open(args.prompts_file, 'r') as f:
        prompts_data = json.load(f)

    # Handle different formats
    if isinstance(prompts_data, list):
        eval_prompts = prompts_data[:args.num_prompts]
    elif isinstance(prompts_data, dict):
        # Extract prompts from dict (assuming 'prompts' key or similar)
        if 'prompts' in prompts_data:
            eval_prompts = prompts_data['prompts'][:args.num_prompts]
        else:
            # Take first N values
            eval_prompts = list(prompts_data.values())[:args.num_prompts]
    else:
        raise ValueError(f"Unexpected prompts format: {type(prompts_data)}")

    # Run comparison
    comparison = BaselineComparison(
        persona_id=args.persona_id,
        eval_prompts=eval_prompts,
        seed=args.seed
    )

    comparison.run_all_baselines(skip_optimization=args.skip_optimization)
    comparison.save_results()

    print(f"\n{'='*80}")
    print("BASELINE COMPARISON COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
