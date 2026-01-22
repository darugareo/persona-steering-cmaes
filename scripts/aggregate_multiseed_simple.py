"""
Simple Multi-Seed Aggregation (no pandas dependency)
Aggregates baseline comparison results across seeds 1-3
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List


def load_seed_results(seed: int) -> Dict:
    """Load results for a single seed."""
    path = Path(f"reports/experiments/results/baseline_comparison_seed{seed}.json")
    with open(path, 'r') as f:
        return json.load(f)


def aggregate_results(seeds: List[int] = [1, 2, 3]) -> Dict:
    """Aggregate results across multiple seeds."""

    # Load all seed results
    all_results = {}
    for seed in seeds:
        all_results[seed] = load_seed_results(seed)

    # Collect scores by method
    method_scores = {}
    methods = list(all_results[seeds[0]].keys())

    for method in methods:
        scores = []
        for seed in seeds:
            if method in all_results[seed]:
                mean_score = all_results[seed][method]['metrics']['mean_score']
                scores.append(mean_score)
        method_scores[method] = scores

    # Compute aggregated statistics
    aggregated = {}
    for method, scores in method_scores.items():
        aggregated[method] = {
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'scores_by_seed': scores,
            'num_seeds': len(scores)
        }

    return aggregated


def format_markdown_table(aggregated: Dict) -> str:
    """Format results as markdown table."""

    lines = []
    lines.append("# Multi-Seed Baseline Comparison Results")
    lines.append("")
    lines.append("## Table 1: Persona-Fit Scores (Mean ± Std across Seeds 1-3)")
    lines.append("")
    lines.append("| Method | Persona-Fit Score |")
    lines.append("|--------|-------------------|")

    # Sort by mean score (descending)
    sorted_methods = sorted(aggregated.items(), key=lambda x: x[1]['mean'], reverse=True)

    for method, stats in sorted_methods:
        method_name = method.replace('_', ' ').title()
        score_str = f"{stats['mean']:.2f} ± {stats['std']:.2f}"
        lines.append(f"| {method_name} | {score_str} |")

    lines.append("")
    lines.append("## Individual Seed Results")
    lines.append("")

    for method, stats in sorted_methods:
        method_name = method.replace('_', ' ').title()
        lines.append(f"### {method_name}")
        for i, score in enumerate(stats['scores_by_seed'], 1):
            lines.append(f"- Seed {i}: {score:.2f}")
        lines.append("")

    return "\n".join(lines)


def format_latex_table(aggregated: Dict) -> str:
    """Format results as LaTeX table."""

    lines = []
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\caption{Persona-Fit Scores across Multiple Seeds}")
    lines.append("\\label{tab:multiseed_results}")
    lines.append("\\begin{tabular}{lc}")
    lines.append("\\hline")
    lines.append("Method & Persona-Fit Score \\\\")
    lines.append("\\hline")

    # Sort by mean score (descending)
    sorted_methods = sorted(aggregated.items(), key=lambda x: x[1]['mean'], reverse=True)

    for method, stats in sorted_methods:
        method_name = method.replace('_', ' ').title()
        score_str = f"${stats['mean']:.2f} \\pm {stats['std']:.2f}$"
        lines.append(f"{method_name} & {score_str} \\\\")

    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def main():
    print("Aggregating multi-seed baseline comparison results...")

    # Aggregate results
    aggregated = aggregate_results(seeds=[1, 2, 3])

    # Save JSON
    output_json = Path("reports/experiments/results/baseline_multiseed_aggregated.json")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(aggregated, f, indent=2)
    print(f"✓ Saved JSON: {output_json}")

    # Save Markdown table
    md_table = format_markdown_table(aggregated)
    output_md = Path("reports/experiments/results/table1_multiseed.md")
    with open(output_md, 'w') as f:
        f.write(md_table)
    print(f"✓ Saved Markdown: {output_md}")

    # Save LaTeX table
    latex_table = format_latex_table(aggregated)
    output_tex = Path("reports/experiments/results/table1_multiseed.tex")
    with open(output_tex, 'w') as f:
        f.write(latex_table)
    print(f"✓ Saved LaTeX: {output_tex}")

    # Print summary
    print("\n" + "="*80)
    print("MULTI-SEED SUMMARY (Seeds 1-3)")
    print("="*80 + "\n")

    sorted_methods = sorted(aggregated.items(), key=lambda x: x[1]['mean'], reverse=True)
    for method, stats in sorted_methods:
        method_name = method.replace('_', ' ').title()
        print(f"{method_name:20s}: {stats['mean']:.2f} ± {stats['std']:.2f}")

    print("\n✓ Multi-seed aggregation complete!")


if __name__ == "__main__":
    main()
