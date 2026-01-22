"""
Ablation Study - Phase 1 Analysis
Analyzes contributions of different components in the Proposed method.

Ablation configurations:
1. Proposed (Full: SVD + CMA-ES)
2. w/o SVD (MeanDiff + CMA-ES weights)
3. w/o CMA-ES (SVD + equal weights)
4. Single Trait: R1-only, R2-only, R3-only, R4-only, R5-only
"""

import json
import argparse
import numpy as np
from pathlib import Path
import sys
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from persona_opt.utils.config_loader import (
    load_experiment_config,
    set_seed,
)
from persona_opt.internal_steering_l3 import Llama3ActivationSteerer
from persona_opt.baselines import ProposedMethod, MeanDiffMethod
from persona_opt.evaluator import PersonaAwareEvaluator
from persona_opt.evaluation.utils import load_persona_profile


class AblationMethod:
    """Ablation method wrapper."""

    def __init__(
        self,
        steerer,
        persona_id: str,
        ablation_type: str,
        layer: int = 20,
        alpha: float = 2.0,
        trait: str = None
    ):
        self.steerer = steerer
        self.persona_id = persona_id
        self.ablation_type = ablation_type
        self.layer = layer
        self.alpha = alpha
        self.trait = trait

        # Load vectors
        self._load_vectors()

    def _load_vectors(self):
        """Load vectors based on ablation type."""
        persona_dir = Path(f"persona-opt/{self.persona_id}")

        if self.ablation_type == "proposed":
            # Full method: SVD + CMA-ES (use ProposedMethod)
            self.method = ProposedMethod(
                steerer=self.steerer,
                persona_id=self.persona_id,
                layer=self.layer,
                alpha=self.alpha
            )

        elif self.ablation_type == "wo_svd":
            # Use MeanDiff method (no SVD) with CMA-ES weights
            from persona_opt.baselines.meandiff import MeanDiffMethod

            # Compute MeanDiff vector
            meandiff_method = MeanDiffMethod(
                steerer=self.steerer,
                persona_id=self.persona_id,
                layer=self.layer,
                alpha=1.0  # Will be scaled by combined weights later
            )

            # Get the MeanDiff steering vector
            meandiff_vector = meandiff_method.steering_vector

            # Load optimized CMA-ES weights
            weights_file = persona_dir / "best_weights.json"
            with open(weights_file, 'r') as f:
                weights_data = json.load(f)
                weights = weights_data.get('weights', [1.0] * 5)

            # Apply CMA-ES weight scaling
            # Note: MeanDiff is a single vector, so we use mean of weights as scaling factor
            weight_scale = sum(abs(w) for w in weights) / len(weights)
            self.steering_vector = meandiff_vector * weight_scale

            print(f"[w/o SVD] Using MeanDiff vector with CMA-ES weight scaling: {weight_scale:.3f}")

        elif self.ablation_type == "wo_cmaes":
            # Use SVD vectors with equal weights (no CMA-ES)
            vectors = {}
            for trait in ["R1", "R2", "R3", "R4", "R5"]:
                vector_file = Path(f"data/steering_vectors_v2/{trait}/layer{self.layer}_svd.pt")
                vector_data = torch.load(vector_file, map_location='cpu')
                if isinstance(vector_data, torch.Tensor):
                    vectors[trait] = vector_data
                elif isinstance(vector_data, dict):
                    vectors[trait] = vector_data['vector']

            # Equal weights
            weights = [1.0] * 5

            # Build combined vector
            self.steering_vector = sum(
                w * vectors[f"R{i+1}"]
                for i, w in enumerate(weights)
            )
            self.steering_vector = self.steering_vector / torch.norm(self.steering_vector)

        elif self.ablation_type.startswith("single_"):
            # Single trait steering
            trait_id = self.trait  # e.g., "R1"
            vector_file = Path(f"data/steering_vectors_v2/{trait_id}/layer{self.layer}_svd.pt")
            vector_data = torch.load(vector_file, map_location='cpu')

            if isinstance(vector_data, torch.Tensor):
                self.steering_vector = vector_data
            elif isinstance(vector_data, dict):
                self.steering_vector = vector_data['vector']

        else:
            raise ValueError(f"Unknown ablation type: {self.ablation_type}")

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response with ablation configuration."""
        if self.ablation_type == "proposed":
            # Use ProposedMethod directly
            return self.method.generate(prompt, **kwargs)
        else:
            # Use custom steering vector
            self.steerer.register_hooks(
                steering_vector=self.steering_vector,
                alpha=self.alpha
            )
            response = self.steerer.generate(prompt, **kwargs)
            return response

    def get_config(self):
        """Return config for metadata."""
        return {
            'ablation_type': self.ablation_type,
            'layer': self.layer,
            'alpha': self.alpha,
            'trait': self.trait
        }


def run_ablation_study(
    persona_id: str,
    num_prompts: int = 10,
    seed: int = 1,
    config_path: str = "config/experiment_config.yaml"
):
    """Run ablation study for a persona."""

    print(f"\n{'='*80}")
    print(f"Ablation Study - Phase 1")
    print(f"Persona: {persona_id}")
    print(f"Seed: {seed}")
    print(f"{'='*80}\n")

    # Load config and set seed
    config = load_experiment_config(config_path)
    set_seed(seed, deterministic=config.deterministic)

    # Load persona profile
    persona_profile = load_persona_profile(persona_id)

    # Initialize steerer
    print(f"Initializing steerer...")
    steerer = Llama3ActivationSteerer(
        model_name=config.model_name,
        target_layer=config.default_layer,
        device=config.device,
    )

    # Load eval prompts
    prompts_file = Path(f"persona-opt/{persona_id}/eval_prompts.json")
    with open(prompts_file, 'r') as f:
        data = json.load(f)
        if isinstance(data, list):
            prompts = data
        elif isinstance(data, dict):
            prompts = data.get('prompts', [])

    # Sample prompts
    if len(prompts) > num_prompts:
        np.random.seed(seed)
        prompts = list(np.random.choice(prompts, num_prompts, replace=False))

    print(f"Loaded {len(prompts)} evaluation prompts\n")

    # Define ablation configurations (MINIMAL SET FOR VALIDATION)
    ablations = [
        ('proposed', None, "Proposed (SVD + CMA-ES)"),
        ('wo_svd', None, "w/o SVD (MeanDiff + CMA-ES)"),
        ('wo_cmaes', None, "w/o CMA-ES (SVD + equal weights)"),
        ('single_R2', 'R2', "Single Trait: R2 only"),
    ]

    # Run ablations
    all_results = []

    for ablation_type, trait, description in ablations:
        print(f"\n{'='*80}")
        print(f"[{len(all_results)+1}/{len(ablations)}] {description}")
        print(f"{'='*80}\n")

        # Create ablation method
        ablation_method = AblationMethod(
            steerer=steerer,
            persona_id=persona_id,
            ablation_type=ablation_type,
            layer=config.default_layer,
            alpha=config.default_alpha,
            trait=trait
        )

        # Generate responses
        responses = []
        for i, prompt in enumerate(prompts):
            steerer.remove_hooks()

            response = ablation_method.generate(
                prompt,
                **config.get_generation_kwargs()
            )
            responses.append(response)

            if (i + 1) % 5 == 0:
                print(f"  Generated {i+1}/{len(prompts)} responses")

        # Evaluate with judge
        evaluator = PersonaAwareEvaluator(
            persona_id=persona_id,
            persona_profile=persona_profile,
            judge_model=config.primary_judge,
            temperature=config.judge_temperature,
            method_name=ablation_type,
            seed=seed,
            experiment="ablation"
        )

        scores = []
        win_count = 0

        for i, (prompt, response) in enumerate(zip(prompts, responses)):
            # Generate baseline
            steerer.remove_hooks()
            baseline_response = steerer.generate(
                prompt,
                **config.get_generation_kwargs()
            )

            # Evaluate
            result = evaluator.evaluate_with_persona_judge(
                prompt=prompt,
                baseline_response=baseline_response,
                steered_response=response
            )

            score = result.get('persona_fit', 3.0)
            scores.append(score)

            if result.get('winner') == 'steered':
                win_count += 1

        mean_score = float(np.mean(scores))
        std_score = float(np.std(scores))
        win_rate = win_count / len(prompts)

        result_data = {
            'ablation_type': ablation_type,
            'description': description,
            'trait': trait,
            'layer': config.default_layer,
            'alpha': config.default_alpha,
            'mean_score': mean_score,
            'std_score': std_score,
            'win_rate': win_rate,
            'scores': scores,
            'num_prompts': len(prompts),
            'seed': seed
        }

        all_results.append(result_data)

        print(f"\n  [{ablation_type}] Mean: {mean_score:.3f} ± {std_score:.3f} | Win Rate: {win_rate:.1%}")

    return all_results


def save_results(persona_id: str, results: list, seed: int):
    """Save ablation results to JSON and Markdown."""

    # Create output directory
    output_dir = Path(f"reports/{persona_id}/phase1/ablation")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON
    json_path = output_dir / f"ablation_seed{seed}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved results: {json_path}")

    # Generate markdown table
    md_lines = [
        f"# Ablation Study Results - {persona_id}",
        "",
        f"**Seed**: {seed}",
        f"**Prompts**: {results[0]['num_prompts']}",
        "",
        "## Results Table",
        "",
        "| Configuration | Mean ± Std | Δ vs Proposed | Win Rate |",
        "|---------------|-----------|---------------|----------|"
    ]

    proposed_score = next(r['mean_score'] for r in results if r['ablation_type'] == 'proposed')

    for r in results:
        config_name = r['description']
        delta = r['mean_score'] - proposed_score
        delta_str = f"{delta:+.3f}"

        md_lines.append(
            f"| {config_name} | {r['mean_score']:.3f} ± {r['std_score']:.3f} | "
            f"{delta_str} | {r['win_rate']:.1%} |"
        )

    md_lines.extend([
        "",
        "## Interpretation",
        ""
    ])

    # Find best single trait
    single_traits = [r for r in results if r['ablation_type'].startswith('single_')]
    if single_traits:
        best_single = max(single_traits, key=lambda x: x['mean_score'])
        md_lines.append(f"- **Best single trait**: {best_single['trait']} ({best_single['mean_score']:.3f})")

    # Compare components
    wo_svd = next((r for r in results if r['ablation_type'] == 'wo_svd'), None)
    wo_cmaes = next((r for r in results if r['ablation_type'] == 'wo_cmaes'), None)

    if wo_svd:
        delta_svd = proposed_score - wo_svd['mean_score']
        md_lines.append(f"- **SVD contribution**: {delta_svd:+.3f}")

    if wo_cmaes:
        delta_cmaes = proposed_score - wo_cmaes['mean_score']
        md_lines.append(f"- **CMA-ES contribution**: {delta_cmaes:+.3f}")

    md_path = output_dir / f"ablation_seed{seed}.md"
    with open(md_path, 'w') as f:
        f.write('\n'.join(md_lines))
    print(f"✓ Saved summary: {md_path}")

    # Generate bar chart
    create_bar_chart(results, output_dir, seed)


def create_bar_chart(results: list, output_dir: Path, seed: int):
    """Create bar chart visualization."""

    fig_dir = output_dir / "figs"
    fig_dir.mkdir(exist_ok=True)

    # Prepare data
    configs = [r['description'] for r in results]
    scores = [r['mean_score'] for r in results]
    stds = [r['std_score'] for r in results]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(configs))
    bars = ax.bar(x, scores, yerr=stds, capsize=5, alpha=0.8, color='steelblue')

    # Highlight proposed
    bars[0].set_color('darkgreen')

    ax.set_xlabel('Configuration', fontsize=12)
    ax.set_ylabel('Mean Persona-Fit Score', fontsize=12)
    ax.set_title(f'Ablation Study Results (Seed {seed})', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    fig_path = fig_dir / f"ablation_bar_seed{seed}.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved figure: {fig_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--persona-id', type=str, default='episode-184019_A')
    parser.add_argument('--num-prompts', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1)

    args = parser.parse_args()

    # Run ablation study
    results = run_ablation_study(
        persona_id=args.persona_id,
        num_prompts=args.num_prompts,
        seed=args.seed
    )

    # Save results
    save_results(
        persona_id=args.persona_id,
        results=results,
        seed=args.seed
    )

    print(f"\n{'='*80}")
    print("Ablation Study Complete!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
