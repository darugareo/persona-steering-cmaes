"""
Phase 2 - Multi-Judge Evaluation
Evaluates Base, Prompt Persona, and Proposed methods using multiple judges (gpt-4o-mini, gpt-4o)
to assess inter-rater reliability and validate judge consistency.
"""

import json
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from persona_opt.utils.config_loader import (
    load_experiment_config,
    set_seed,
)
from persona_opt.internal_steering_l3 import Llama3ActivationSteerer
from persona_opt.baselines import (
    BaseMethod,
    PromptPersonaMethod,
    ProposedMethod
)
from persona_opt.evaluation.utils import load_persona_profile
from persona_opt.evaluator import PersonaAwareEvaluator


class MultiJudgeEvaluator:
    """Multi-judge evaluation for persona steering methods."""

    def __init__(
        self,
        persona_id: str,
        seed: int = 1,
        num_prompts: int = 20,
        judge_models: List[str] = None,
        config_path: str = "config/experiment_config.yaml"
    ):
        self.persona_id = persona_id
        self.seed = seed
        self.num_prompts = num_prompts

        # Default to gpt-4o-mini and gpt-4o
        self.judge_models = judge_models or ["gpt-4o-mini", "gpt-4o"]

        # Load config and set seed
        self.config = load_experiment_config(config_path)
        set_seed(seed, deterministic=self.config.deterministic)

        # Load persona profile
        self.persona_profile = load_persona_profile(persona_id)

        # Initialize steerer
        print(f"\n[Multi-Judge] Initializing steerer...")
        self.steerer = Llama3ActivationSteerer(
            model_name=self.config.model_name,
            target_layer=self.config.default_layer,
            device=self.config.device,
        )

        # Load test prompts
        self.prompts = self._load_test_prompts()

        self.results = {}

    def _load_test_prompts(self) -> List[str]:
        """Load test prompts from Phase 1 eval_prompts.json."""
        # Load from persona-opt eval_prompts.json
        prompts_file = Path(f"persona-opt/{self.persona_id}/eval_prompts.json")

        if prompts_file.exists():
            with open(prompts_file, 'r') as f:
                data = json.load(f)
                # Can be either list of strings or dict with 'prompts' key
                if isinstance(data, list):
                    prompts = data
                elif isinstance(data, dict):
                    prompts = data.get('prompts', [])
                else:
                    prompts = []
        else:
            raise FileNotFoundError(f"No eval_prompts.json found for {self.persona_id}")

        # Sample prompts
        if len(prompts) > self.num_prompts:
            np.random.seed(self.seed)
            prompts = list(np.random.choice(prompts, self.num_prompts, replace=False))

        print(f"[Multi-Judge] Loaded {len(prompts)} test prompts")
        return prompts

    def evaluate_method_with_judges(
        self,
        method_name: str,
        method_instance
    ) -> Dict:
        """Evaluate a method using multiple judges."""

        print(f"\n{'='*80}")
        print(f"Evaluating: {method_name}")
        print(f"{'='*80}")

        # Generate responses
        print(f"Generating responses for {len(self.prompts)} prompts...")
        responses = []

        for i, prompt in enumerate(self.prompts):
            self.steerer.remove_hooks()

            if method_name == "base":
                response = self.steerer.generate(
                    prompt,
                    **self.config.get_generation_kwargs()
                )
            else:
                response = method_instance.generate(
                    prompt,
                    **self.config.get_generation_kwargs()
                )

            responses.append(response)

            if (i + 1) % 5 == 0:
                print(f"  Generated {i+1}/{len(self.prompts)}")

        # Evaluate with each judge
        scores_by_judge = {}
        detailed_results_by_judge = {}

        for judge_model in self.judge_models:
            print(f"\nEvaluating with {judge_model}...")

            # Create evaluator for this judge
            evaluator = PersonaAwareEvaluator(
                persona_id=self.persona_id,
                persona_profile=self.persona_profile,
                judge_model=judge_model,
                temperature=0.3,  # Low temp for consistency
                method_name=method_name,
                seed=self.seed,
                experiment="multi_judge"
            )

            scores = []
            detailed_results = []

            for i, (prompt, response) in enumerate(zip(self.prompts, responses)):
                # Generate baseline response
                self.steerer.remove_hooks()
                baseline_response = self.steerer.generate(
                    prompt,
                    **self.config.get_generation_kwargs()
                )

                # Evaluate
                result = evaluator.evaluate_with_persona_judge(
                    prompt=prompt,
                    baseline_response=baseline_response,
                    steered_response=response
                )

                score = result.get('persona_fit', 3.0)  # persona_fit is the score for steered response
                scores.append(score)

                detailed_results.append({
                    'prompt': prompt,
                    'response': response,
                    'baseline_response': baseline_response,
                    'score': score,
                    'explanation': result.get('explanation', '')
                })

                if (i + 1) % 5 == 0:
                    print(f"  Evaluated {i+1}/{len(self.prompts)} | Mean: {np.mean(scores):.3f}")

            scores_by_judge[judge_model] = scores
            detailed_results_by_judge[judge_model] = detailed_results

            mean_score = float(np.mean(scores))
            std_score = float(np.std(scores))
            print(f"\n  [{judge_model}] Mean: {mean_score:.3f} ± {std_score:.3f}")

        # Compute inter-rater reliability
        reliability = self._compute_reliability(scores_by_judge)

        result = {
            'method': method_name,
            'num_prompts': len(self.prompts),
            'seed': self.seed,
            'scores_by_judge': scores_by_judge,
            'detailed_results_by_judge': detailed_results_by_judge,
            'reliability': reliability,
            'judge_models': self.judge_models
        }

        return result

    def _compute_reliability(self, scores_by_judge: Dict[str, List[float]]) -> Dict:
        """Compute inter-rater reliability metrics."""

        judge_names = list(scores_by_judge.keys())

        if len(judge_names) < 2:
            return {'error': 'Need at least 2 judges'}

        # Compute pairwise correlations
        correlations = {}
        for i, judge1 in enumerate(judge_names):
            for judge2 in judge_names[i+1:]:
                scores1 = np.array(scores_by_judge[judge1])
                scores2 = np.array(scores_by_judge[judge2])

                # Pearson correlation
                corr = float(np.corrcoef(scores1, scores2)[0, 1])
                correlations[f"{judge1}_vs_{judge2}"] = corr

        # Mean correlation
        mean_corr = float(np.mean(list(correlations.values())))

        # Mean absolute difference
        mad_values = []
        for i, judge1 in enumerate(judge_names):
            for judge2 in judge_names[i+1:]:
                scores1 = np.array(scores_by_judge[judge1])
                scores2 = np.array(scores_by_judge[judge2])
                mad = float(np.mean(np.abs(scores1 - scores2)))
                mad_values.append(mad)

        mean_mad = float(np.mean(mad_values)) if mad_values else 0.0

        return {
            'pairwise_correlations': correlations,
            'mean_correlation': mean_corr,
            'mean_absolute_difference': mean_mad
        }

    def run_all_methods(self):
        """Run evaluation on all methods with all judges."""

        # 1. Base Method
        print(f"\n[1/3] Base Method (No Steering)")
        base_method = BaseMethod(
            steerer=self.steerer,
            persona_id=self.persona_id
        )
        self.results['base'] = self.evaluate_method_with_judges('base', base_method)

        # 2. Prompt Persona Method
        print(f"\n[2/3] Prompt Persona Method")
        prompt_method = PromptPersonaMethod(
            steerer=self.steerer,
            persona_id=self.persona_id,
            persona_profile=self.persona_profile
        )
        self.results['prompt_persona'] = self.evaluate_method_with_judges('prompt_persona', prompt_method)

        # 3. Proposed Method
        print(f"\n[3/3] Proposed Method (SVD + CMA-ES)")
        proposed_method = ProposedMethod(
            steerer=self.steerer,
            persona_id=self.persona_id,
            layer=self.config.default_layer,
            alpha=self.config.default_alpha
        )
        self.results['proposed'] = self.evaluate_method_with_judges('proposed', proposed_method)

        return self.results

    def save_results(self):
        """Save results to JSON and Markdown."""

        # Create output directory (persona-specific)
        output_dir = Path(f"reports/{self.persona_id}/phase2/multi_judge")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save full JSON
        json_path = output_dir / f"multi_judge_seed{self.seed}.json"

        # Prepare simplified JSON (remove detailed per-prompt data for size)
        simplified_results = {}
        for method, result in self.results.items():
            simplified_results[method] = {
                'method': result['method'],
                'num_prompts': result['num_prompts'],
                'seed': result['seed'],
                'judge_models': result['judge_models'],
                'scores_by_judge': {
                    judge: {
                        'mean': float(np.mean(scores)),
                        'std': float(np.std(scores)),
                        'scores': scores
                    }
                    for judge, scores in result['scores_by_judge'].items()
                },
                'reliability': result['reliability']
            }

        with open(json_path, 'w') as f:
            json.dump(simplified_results, f, indent=2)
        print(f"\n✓ Saved results: {json_path}")

        # Generate markdown summary
        md_lines = [
            f"# Multi-Judge Evaluation Results (Seed {self.seed})",
            "",
            f"**Judges**: {', '.join(self.judge_models)}",
            f"**Prompts**: {self.num_prompts}",
            "",
            "## Mean Scores by Method and Judge",
            "",
            "| Method | " + " | ".join(self.judge_models) + " | Mean |",
            "|--------|" + "|".join(["------"] * len(self.judge_models)) + "|------|"
        ]

        # Add rows for each method
        for method_name in ['base', 'prompt_persona', 'proposed']:
            result = self.results[method_name]
            scores_by_judge = result['scores_by_judge']

            row_values = [method_name.replace('_', ' ').title()]

            judge_means = []
            for judge in self.judge_models:
                scores = scores_by_judge[judge]
                mean = np.mean(scores)
                judge_means.append(mean)
                row_values.append(f"{mean:.3f}")

            overall_mean = np.mean(judge_means)
            row_values.append(f"{overall_mean:.3f}")

            md_lines.append("| " + " | ".join(row_values) + " |")

        md_lines.extend(["", "## Inter-Rater Reliability", ""])

        # Add reliability metrics
        for method_name in ['base', 'prompt_persona', 'proposed']:
            result = self.results[method_name]
            reliability = result['reliability']

            md_lines.append(f"### {method_name.replace('_', ' ').title()}")
            md_lines.append("")

            if 'pairwise_correlations' in reliability:
                for pair, corr in reliability['pairwise_correlations'].items():
                    md_lines.append(f"- **{pair}**: r = {corr:.3f}")

            if 'mean_correlation' in reliability:
                md_lines.append(f"- **Mean correlation**: r = {reliability['mean_correlation']:.3f}")

            if 'mean_absolute_difference' in reliability:
                md_lines.append(f"- **Mean absolute difference**: {reliability['mean_absolute_difference']:.3f}")

            md_lines.append("")

        # Add interpretation
        md_lines.extend(["## Interpretation", ""])

        # Check if judges agree
        proposed_reliability = self.results['proposed']['reliability']
        if 'mean_correlation' in proposed_reliability:
            corr = proposed_reliability['mean_correlation']
            if corr > 0.8:
                md_lines.append(f"✓ **High inter-rater reliability** (r = {corr:.3f}) - judges strongly agree")
            elif corr > 0.6:
                md_lines.append(f"✓ **Moderate inter-rater reliability** (r = {corr:.3f}) - judges moderately agree")
            else:
                md_lines.append(f"⚠ **Low inter-rater reliability** (r = {corr:.3f}) - judges show inconsistency")

        md_path = output_dir / f"multi_judge_seed{self.seed}.md"
        with open(md_path, 'w') as f:
            f.write('\n'.join(md_lines))
        print(f"✓ Saved summary: {md_path}")


def run_single_persona(persona_id: str, seed: int, num_prompts: int, judges: List[str]):
    """Run multi-judge evaluation for a single persona."""
    print(f"\n{'='*80}")
    print(f"Phase 2: Multi-Judge Evaluation - {persona_id}")
    print(f"Seed: {seed}")
    print(f"Judges: {', '.join(judges)}")
    print(f"Prompts: {num_prompts}")
    print(f"{'='*80}\n")

    evaluator = MultiJudgeEvaluator(
        persona_id=persona_id,
        seed=seed,
        num_prompts=num_prompts,
        judge_models=judges
    )

    results = evaluator.run_all_methods()
    evaluator.save_results()

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--persona-id', type=str, help='Single persona ID')
    parser.add_argument('--persona-list', type=str, help='Path to personas.yaml')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--num-prompts', type=int, default=20,
                       help='Number of prompts to evaluate (default: 20)')
    parser.add_argument('--judges', type=str, nargs='+', default=['gpt-4o-mini', 'gpt-4o'],
                       help='Judge models to use (default: gpt-4o-mini gpt-4o)')

    args = parser.parse_args()

    # Get persona list
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from persona_opt.utils.persona_list_loader import get_persona_list_from_args
    persona_list = get_persona_list_from_args(args)

    print(f"\n{'='*80}")
    print(f"Multi-Judge Multi-Persona Pipeline")
    print(f"Personas: {', '.join(persona_list)}")
    print(f"{'='*80}\n")

    # Run for each persona
    for persona_id in persona_list:
        print(f"\n{'#'*80}")
        print(f"# Processing: {persona_id}")
        print(f"{'#'*80}\n")

        run_single_persona(
            persona_id=persona_id,
            seed=args.seed,
            num_prompts=args.num_prompts,
            judges=args.judges
        )

    print(f"\n{'='*80}")
    print("All Personas Complete!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
