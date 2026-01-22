"""
Fast Baseline Comparison - Skips Random Search and Grid Search
Phase 1 Step 3: Quick baseline comparison for seeds 2 & 3
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
)
from persona_opt.evaluator import PersonaAwareEvaluator
from persona_opt.evaluation.utils import load_persona_profile


class FastBaselineComparison:
    """Fast baseline comparison - skips Random/Grid Search."""

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
        set_seed(seed, deterministic=self.config.deterministic)

        # Create output directories
        self.output_dirs = create_output_directories(self.config)

        # Initialize logger
        self.logger = ExperimentLogger(self.config, seed)

        # Load persona profile
        self.persona_profile = load_persona_profile(persona_id)

        # Initialize steerer
        print(f"\n[FastBaselineComparison] Initializing steerer...")
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

        self.results = {}

    def run_method(
        self,
        method_name: str,
        method_instance,
        prompts: List[str],
    ) -> Dict[str, Any]:
        """Evaluate a single method."""
        print(f"\n{'='*80}")
        print(f"Evaluating: {method_name}")
        print(f"{'='*80}")

        # Generate baseline responses
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

        # Get method config for metadata
        method_config = method_instance.get_config()
        layer = method_config.get('layer', self.config.default_layer)
        alpha = method_config.get('alpha', self.config.default_alpha)
        weights = method_config.get('weights', None)

        # Create evaluator with method-specific metadata
        evaluator_with_meta = PersonaAwareEvaluator(
            persona_id=self.persona_id,
            persona_profile=self.persona_profile,
            judge_model=self.config.primary_judge,
            temperature=self.config.judge_temperature,
            method_name=method_name,
            seed=self.seed,
            layer=layer,
            alpha=alpha,
            weights=weights
        )

        # Evaluate
        print(f"[{method_name}] Evaluating with {self.config.primary_judge}...")
        eval_results = evaluator_with_meta.batch_evaluate(
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
            'scores': scores
        }

        result = {
            'method': method_name,
            'config': method_instance.get_config(),
            'metrics': metrics,
            'eval_results': eval_results
        }

        print(f"[{method_name}] Results:")
        print(f"  Mean Score: {metrics['mean_score']:.2f} ± {metrics['std_score']:.2f}")
        print(f"  Win Rate: {metrics['win_rate']:.1%}")

        return result

    def run_all(self, num_prompts: int = 10) -> Dict[str, Any]:
        """Run all baseline methods (excluding Random/Grid Search)."""

        # Sample prompts
        np.random.seed(self.seed)
        prompts = np.random.choice(
            self.eval_prompts,
            min(num_prompts, len(self.eval_prompts)),
            replace=False
        ).tolist()

        print(f"\n{'='*80}")
        print(f"FAST BASELINE COMPARISON")
        print(f"Persona: {self.persona_id}")
        print(f"Prompts: {len(prompts)}")
        print(f"Seed: {self.seed}")
        print(f"Methods: Base, Prompt Persona, MeanDiff, PCA, Proposed")
        print(f"{'='*80}\n")

        # 1. Base Method
        print(f"\n[1/5] Base Method\n")
        base_method = BaseMethod(
            steerer=self.steerer,
            persona_id=self.persona_id
        )
        self.results['base'] = self.run_method('base', base_method, prompts)

        # 2. Prompt Persona Method
        print(f"\n[2/5] Prompt Persona Method\n")
        prompt_method = PromptPersonaMethod(
            steerer=self.steerer,
            persona_id=self.persona_id,
            persona_profile=self.persona_profile
        )
        self.results['prompt_persona'] = self.run_method('prompt_persona', prompt_method, prompts)

        # 3. MeanDiff Method
        print(f"\n[3/5] Mean Difference Method")
        meandiff_method = MeanDiffMethod(
            steerer=self.steerer,
            persona_id=self.persona_id,
            layer=self.config.default_layer,
            alpha=self.config.default_alpha
        )
        self.results['meandiff'] = self.run_method('meandiff', meandiff_method, prompts)

        # 4. PCA Method
        print(f"\n[4/5] PCA Steering Method")
        pca_method = PCASteeringMethod(
            steerer=self.steerer,
            persona_id=self.persona_id,
            layer=self.config.default_layer,
            alpha=self.config.default_alpha
        )
        self.results['pca'] = self.run_method('pca', pca_method, prompts)

        # 5. Proposed Method (from optimized weights)
        print(f"\n[5/5] Proposed Method (SVD + CMA-ES)")
        from persona_opt.baselines.proposed import ProposedMethod
        proposed_method = ProposedMethod(
            steerer=self.steerer,
            persona_id=self.persona_id,
            layer=self.config.default_layer,
            alpha=self.config.default_alpha
        )
        self.results['proposed'] = self.run_method('proposed', proposed_method, prompts)

        return self.results

    def save_results(self, output_path: Path):
        """Save results to JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\n✓ Results saved to: {output_path}")


def run_single_persona(persona_id: str, prompts_file: str, num_prompts: int, seed: int):
    """Run baseline comparison for a single persona."""
    # Load prompts
    with open(prompts_file, 'r') as f:
        data = json.load(f)
        if isinstance(data, dict) and 'prompts' in data:
            prompts = [p['text'] if isinstance(p, dict) else p for p in data['prompts']]
        else:
            prompts = data

    # Run comparison
    comparison = FastBaselineComparison(
        persona_id=persona_id,
        eval_prompts=prompts,
        seed=seed
    )

    results = comparison.run_all(num_prompts=num_prompts)

    # Save results with persona-specific path
    output_path = Path(f"reports/{persona_id}/phase1/baseline_comparison_seed{seed}.json")
    comparison.save_results(output_path)

    # Print summary
    print(f"\n{'='*80}")
    print(f"SUMMARY - {persona_id} (Seed {seed})")
    print(f"{'='*80}\n")

    for method_name, result in results.items():
        metrics = result['metrics']
        print(f"{method_name:20s}: {metrics['mean_score']:.2f} ± {metrics['std_score']:.2f} (Win Rate: {metrics['win_rate']:.1%})")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--persona-id', type=str, help='Single persona ID')
    parser.add_argument('--persona-list', type=str, help='Path to personas.yaml for batch processing')
    parser.add_argument('--prompts-file', type=str, required=True)
    parser.add_argument('--num-prompts', type=int, default=20)
    parser.add_argument('--seed', type=int, default=1)

    args = parser.parse_args()

    # Get persona list
    from persona_opt.utils.persona_list_loader import get_persona_list_from_args
    persona_list = get_persona_list_from_args(args)

    print(f"\n{'='*80}")
    print(f"Fast Baseline Comparison - Multi-Persona Pipeline")
    print(f"Personas: {', '.join(persona_list)}")
    print(f"Seed: {args.seed}")
    print(f"{'='*80}\n")

    # Run for each persona
    all_results = {}
    for persona_id in persona_list:
        print(f"\n\n{'#'*80}")
        print(f"# Processing: {persona_id}")
        print(f"{'#'*80}\n")

        results = run_single_persona(
            persona_id=persona_id,
            prompts_file=args.prompts_file,
            num_prompts=args.num_prompts,
            seed=args.seed
        )

        all_results[persona_id] = results

    print(f"\n{'='*80}")
    print(f"All Personas Complete!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
