"""
Generate all Phase 1 figures:
1. Ablation bar chart
2. Seed variation plot (seed1-3 with error bars)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

sns.set_style('whitegrid')

def load_ablation_results(results_file: Path):
    """Load ablation study results."""
    with open(results_file, 'r') as f:
        return json.load(f)

def create_ablation_bar_chart(results, output_file: Path):
    """Create bar chart showing ablation results."""

    # Extract data
    configs = []
    scores = []
    stds = []

    # Order: Proposed, w/o SVD, w/o CMA-ES, then single traits
    order = ['proposed', 'wo_svd', 'wo_cmaes', 'single_R1', 'single_R2', 'single_R3', 'single_R4', 'single_R5']

    for ablation_type in order:
        r = next((x for x in results if x['ablation_type'] == ablation_type), None)
        if r:
            if r['trait']:
                configs.append(f"Only {r['trait']}")
            else:
                config_name = {
                    'proposed': 'Proposed\n(SVD+CMA-ES)',
                    'wo_svd': 'w/o SVD',
                    'wo_cmaes': 'w/o CMA-ES'
                }[ablation_type]
                configs.append(config_name)

            scores.append(r['mean_score'])
            stds.append(r['std_score'])

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(configs))
    colors = ['#2ecc71' if i == 0 else '#95a5a6' for i in range(len(configs))]

    bars = ax.bar(x, scores, yerr=stds, capsize=5, color=colors, edgecolor='black', linewidth=1.2)

    ax.set_ylabel('Persona-Fit Score', fontsize=12)
    ax.set_title('Ablation Study: Component Contributions', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha='right', fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars, scores)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{score:.3f}',
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Ablation bar chart saved to: {output_file}")
    plt.close()

def load_multiseed_baseline_results(results_dir: Path, seeds: list):
    """Load baseline results from multiple seeds."""
    seed_data = {}
    for seed in seeds:
        result_file = results_dir / f"baseline_comparison_seed{seed}.json"
        if result_file.exists():
            with open(result_file, 'r') as f:
                seed_data[seed] = json.load(f)
    return seed_data

def create_seed_variation_plot(seed_data, output_file: Path):
    """Create plot showing variation across seeds."""

    # Aggregate by method
    method_names = []
    means = []
    stds = []

    # Get all methods
    all_methods = set()
    for seed, data in seed_data.items():
        all_methods.update(data.keys())

    for method in sorted(all_methods):
        method_scores = []

        for seed, data in seed_data.items():
            if method in data:
                scores = data[method].get('persona_fit', [])
                if scores:
                    method_scores.append(np.mean(scores))

        if method_scores:
            method_names.append(method)
            means.append(np.mean(method_scores))
            stds.append(np.std(method_scores, ddof=1) if len(method_scores) > 1 else 0.0)

    # Sort by mean score
    sorted_indices = np.argsort(means)[::-1]
    method_names = [method_names[i] for i in sorted_indices]
    means = [means[i] for i in sorted_indices]
    stds = [stds[i] for i in sorted_indices]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(method_names))
    ax.bar(x, means, yerr=stds, capsize=5, color='skyblue', edgecolor='black', linewidth=1.2)

    ax.set_ylabel('Persona-Fit Score', fontsize=12)
    ax.set_xlabel('Method', fontsize=12)
    ax.set_title('Baseline Comparison (Seeds 1-3)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(method_names, rotation=45, ha='right', fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(i, mean + std + 0.02, f'{mean:.3f}±{std:.3f}',
                ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Seed variation plot saved to: {output_file}")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ablation-results', type=str, required=False,
                        help='Path to ablation results JSON')
    parser.add_argument('--generate-ablation', action='store_true',
                        help='Generate ablation bar chart')
    parser.add_argument('--generate-seed-variation', action='store_true',
                        help='Generate seed variation plot')

    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    figures_dir = base_dir / "reports" / "experiments" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    results_dir = base_dir / "reports" / "experiments" / "results"

    if args.generate_ablation and args.ablation_results:
        # Generate ablation bar chart
        ablation_file = Path(args.ablation_results)
        if ablation_file.exists():
            results = load_ablation_results(ablation_file)
            output_file = figures_dir / "ablation_bar_chart.png"
            create_ablation_bar_chart(results, output_file)
        else:
            print(f"Error: {ablation_file} not found!")

    if args.generate_seed_variation:
        # Generate seed variation plot
        seeds = [1, 2, 3]
        seed_data = load_multiseed_baseline_results(results_dir, seeds)

        if seed_data:
            output_file = figures_dir / "seed_variation_plot.png"
            create_seed_variation_plot(seed_data, output_file)
        else:
            print("Error: No seed data found!")

    print("\n✓ Figure generation complete!")

if __name__ == "__main__":
    main()
