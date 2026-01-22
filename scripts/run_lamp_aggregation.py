#!/usr/bin/env python3
"""
LaMP-7 Results Aggregation - Phase 2-D

Aggregates judge comparison results and computes:
- Win rates with 95% confidence intervals (bootstrap)
- Statistical significance tests
- Comparison tables for paper
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from scipy import stats
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def bootstrap_ci(data: List[int], n_bootstrap: int = 10000, ci: float = 0.95) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for win rate.

    Args:
        data: List of 0/1 (0=loss, 1=win)
        n_bootstrap: Number of bootstrap samples
        ci: Confidence level (default: 0.95 for 95% CI)

    Returns:
        (mean, ci_lower, ci_upper)
    """
    data = np.array(data)
    n = len(data)
    bootstrap_means = []

    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_means.append(sample.mean())

    bootstrap_means = np.array(bootstrap_means)
    mean = data.mean()
    alpha = 1 - ci
    ci_lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))

    return mean, ci_lower, ci_upper


def binomial_test(wins: int, total: int, p: float = 0.5) -> float:
    """
    Test if win rate is significantly different from p (default 0.5).

    Returns p-value.
    """
    return stats.binomtest(wins, total, p, alternative='two-sided').pvalue


def compute_method_comparison(results: List[Dict], method_a: str, method_b: str) -> Dict:
    """Compute statistics for method_a vs method_b comparison."""
    # Filter results for this comparison
    comp_results = [r for r in results if r['method_a'] == method_a and r['method_b'] == method_b]

    if not comp_results:
        return {}

    # Count wins
    wins_a = [1 if r['winner'] == 'A' else 0 for r in comp_results]
    wins_b = [1 if r['winner'] == 'B' else 0 for r in comp_results]

    total = len(comp_results)
    n_wins_a = sum(wins_a)
    n_wins_b = sum(wins_b)

    # Bootstrap CI
    mean_a, ci_lower_a, ci_upper_a = bootstrap_ci(wins_a)
    mean_b, ci_lower_b, ci_upper_b = bootstrap_ci(wins_b)

    # Binomial test (null hypothesis: win rate = 0.5)
    p_value_a = binomial_test(n_wins_a, total, p=0.5)

    # Average confidence score
    avg_confidence = np.mean([r['confidence'] for r in comp_results])

    return {
        'method_a': method_a,
        'method_b': method_b,
        'total_samples': total,
        'wins_a': n_wins_a,
        'wins_b': n_wins_b,
        'win_rate_a': mean_a,
        'win_rate_a_ci_lower': ci_lower_a,
        'win_rate_a_ci_upper': ci_upper_a,
        'win_rate_b': mean_b,
        'win_rate_b_ci_lower': ci_lower_b,
        'win_rate_b_ci_upper': ci_upper_b,
        'p_value': p_value_a,
        'significant': p_value_a < 0.05,
        'avg_confidence': avg_confidence
    }


def format_table_row(stats: Dict) -> str:
    """Format comparison as table row for paper."""
    method_a = stats['method_a'].capitalize()
    method_b = stats['method_b'].capitalize()
    win_rate_a = stats['win_rate_a'] * 100
    ci_lower = stats['win_rate_a_ci_lower'] * 100
    ci_upper = stats['win_rate_a_ci_upper'] * 100
    p_value = stats['p_value']
    sig_marker = "*" if stats['significant'] else ""

    return f"{method_a:12} vs {method_b:12} | {win_rate_a:5.1f}% | [{ci_lower:5.1f}%, {ci_upper:5.1f}%] | p={p_value:.4f}{sig_marker}"


def main():
    parser = argparse.ArgumentParser(description="Aggregate LaMP-7 judge results")

    parser.add_argument("--judge-results", type=str, required=True,
                       help="Path to judge_comparisons.json")
    parser.add_argument("--output-dir", type=str, default="results/lamp7",
                       help="Output directory for aggregated results")
    parser.add_argument("--n-bootstrap", type=int, default=10000,
                       help="Number of bootstrap samples (default: 10000)")

    args = parser.parse_args()

    # Load judge results
    with open(args.judge_results) as f:
        judge_data = json.load(f)

    results = judge_data['results']
    logger.info(f"Loaded {len(results)} judge comparison results")

    # Get unique comparisons
    comparisons = set((r['method_a'], r['method_b']) for r in results)
    logger.info(f"Found {len(comparisons)} comparison types")

    # Compute statistics for each comparison
    all_stats = []
    for method_a, method_b in sorted(comparisons):
        stats = compute_method_comparison(results, method_a, method_b)
        if stats:
            all_stats.append(stats)

    # Save detailed statistics
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats_file = output_dir / "aggregated_statistics.json"
    with open(stats_file, 'w') as f:
        json.dump({
            'judge_model': judge_data['judge_model'],
            'num_samples': judge_data['num_samples'],
            'n_bootstrap': args.n_bootstrap,
            'comparisons': all_stats
        }, f, indent=2)
    logger.info(f"Saved detailed statistics to {stats_file}")

    # Generate table for paper
    table_lines = [
        "=" * 80,
        "LaMP-7 A/B Comparison Results",
        "=" * 80,
        f"Judge Model: {judge_data['judge_model']}",
        f"Samples: {judge_data['num_samples']}",
        f"Bootstrap Samples: {args.n_bootstrap}",
        "=" * 80,
        "",
        f"{'Comparison':27} | {'Win Rate':7} | {'95% CI':20} | {'Significance':15}",
        "-" * 80,
    ]

    for stats in all_stats:
        table_lines.append(format_table_row(stats))

    table_lines.extend([
        "-" * 80,
        "",
        "* = Statistically significant (p < 0.05)",
        "",
        "Win Rate: Percentage of times Method A won the comparison",
        "95% CI: 95% confidence interval (bootstrap)",
        "p-value: Binomial test against null hypothesis (win rate = 50%)",
        ""
    ])

    table_text = "\n".join(table_lines)

    # Save table
    table_file = output_dir / "comparison_table.txt"
    with open(table_file, 'w') as f:
        f.write(table_text)
    logger.info(f"Saved comparison table to {table_file}")

    # Print table
    print("\n" + table_text)

    # Generate LaTeX table format
    latex_lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{LaMP-7 Tweet Paraphrasing: A/B Comparison Results}",
        "\\label{tab:lamp7_results}",
        "\\begin{tabular}{ll|c|c|c}",
        "\\toprule",
        "Method A & Method B & Win Rate & 95\\% CI & p-value \\\\",
        "\\midrule"
    ]

    for stats in all_stats:
        method_a = stats['method_a'].capitalize()
        method_b = stats['method_b'].capitalize()
        win_rate = stats['win_rate_a'] * 100
        ci_lower = stats['win_rate_a_ci_lower'] * 100
        ci_upper = stats['win_rate_a_ci_upper'] * 100
        p_value = stats['p_value']
        sig = "$^*$" if stats['significant'] else ""

        latex_lines.append(
            f"{method_a} & {method_b} & {win_rate:.1f}\\% & [{ci_lower:.1f}, {ci_upper:.1f}] & {p_value:.4f}{sig} \\\\"
        )

    latex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\\\[0.5em]",
        "{\\small $^*$ Significant at $p < 0.05$}",
        "\\end{table}"
    ])

    latex_text = "\n".join(latex_lines)

    latex_file = output_dir / "comparison_table.tex"
    with open(latex_file, 'w') as f:
        f.write(latex_text)
    logger.info(f"Saved LaTeX table to {latex_file}")

    # Key findings summary
    logger.info("\n" + "=" * 80)
    logger.info("KEY FINDINGS")
    logger.info("=" * 80)

    # Find optimized vs equal comparison
    opt_vs_equal = next((s for s in all_stats if s['method_a'] == 'equal' and s['method_b'] == 'optimized'), None)
    if opt_vs_equal:
        win_rate = opt_vs_equal['win_rate_b'] * 100  # optimized is method_b
        ci_lower = opt_vs_equal['win_rate_b_ci_lower'] * 100
        ci_upper = opt_vs_equal['win_rate_b_ci_upper'] * 100
        is_sig = opt_vs_equal['significant']

        logger.info(f"Optimized vs Equal-weight:")
        logger.info(f"  Optimized win rate: {win_rate:.1f}% [{ci_lower:.1f}%, {ci_upper:.1f}%]")
        logger.info(f"  Statistical significance: {'YES (p < 0.05)' if is_sig else 'NO'}")

    # Find optimized vs base comparison
    opt_vs_base = next((s for s in all_stats if s['method_a'] == 'base' and s['method_b'] == 'optimized'), None)
    if opt_vs_base:
        win_rate = opt_vs_base['win_rate_b'] * 100
        ci_lower = opt_vs_base['win_rate_b_ci_lower'] * 100
        ci_upper = opt_vs_base['win_rate_b_ci_upper'] * 100

        logger.info(f"\nOptimized vs Base:")
        logger.info(f"  Optimized win rate: {win_rate:.1f}% [{ci_lower:.1f}%, {ci_upper:.1f}%]")

    logger.info("=" * 80)
    logger.info("\nAggregation complete!")


if __name__ == "__main__":
    main()
