"""
Statistical analysis for ablation study results.

Provides:
- Confidence intervals (Bootstrap)
- Statistical significance tests (Wilcoxon signed-rank test)
- Effect sizes (Cohen's d)
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats
from typing import Dict, List, Tuple


def bootstrap_ci(
    scores: List[float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval.

    Args:
        scores: List of scores
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (e.g., 0.95 for 95% CI)

    Returns:
        (mean, lower_bound, upper_bound)
    """
    if len(scores) == 0:
        return (0.0, 0.0, 0.0)

    scores = np.array(scores)
    bootstrap_means = []

    for _ in range(n_bootstrap):
        sample = np.random.choice(scores, size=len(scores), replace=True)
        bootstrap_means.append(np.mean(sample))

    bootstrap_means = np.array(bootstrap_means)
    mean = np.mean(scores)
    alpha = (1 - confidence) / 2
    lower = np.percentile(bootstrap_means, alpha * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha) * 100)

    return (mean, lower, upper)


def wilcoxon_test(
    scores_a: List[float],
    scores_b: List[float]
) -> Dict:
    """
    Perform Wilcoxon signed-rank test.

    Tests if scores_b is significantly different from scores_a.
    Paired test (same prompts evaluated by both methods).

    Args:
        scores_a: Scores from method A
        scores_b: Scores from method B

    Returns:
        {
            'statistic': float,
            'p_value': float,
            'significant': bool (p < 0.05),
            'interpretation': str
        }
    """
    if len(scores_a) != len(scores_b):
        raise ValueError("Scores must have same length for paired test")

    if len(scores_a) < 3:
        return {
            'statistic': None,
            'p_value': None,
            'significant': None,
            'interpretation': 'Sample size too small (n < 3) for statistical test'
        }

    # Wilcoxon signed-rank test (paired, non-parametric)
    statistic, p_value = stats.wilcoxon(scores_a, scores_b, alternative='two-sided')

    significant = p_value < 0.05

    if p_value < 0.001:
        interp = f"Highly significant difference (p < 0.001)"
    elif p_value < 0.01:
        interp = f"Very significant difference (p = {p_value:.3f})"
    elif p_value < 0.05:
        interp = f"Significant difference (p = {p_value:.3f})"
    else:
        interp = f"No significant difference (p = {p_value:.3f})"

    return {
        'statistic': float(statistic),
        'p_value': float(p_value),
        'significant': bool(significant),
        'interpretation': interp
    }


def cohens_d(
    scores_a: List[float],
    scores_b: List[float]
) -> Dict:
    """
    Compute Cohen's d effect size.

    Args:
        scores_a: Scores from method A
        scores_b: Scores from method B

    Returns:
        {
            'cohens_d': float,
            'magnitude': str ('negligible', 'small', 'medium', 'large')
        }
    """
    scores_a = np.array(scores_a)
    scores_b = np.array(scores_b)

    mean_a = np.mean(scores_a)
    mean_b = np.mean(scores_b)

    # Pooled standard deviation
    n_a = len(scores_a)
    n_b = len(scores_b)
    std_a = np.std(scores_a, ddof=1)
    std_b = np.std(scores_b, ddof=1)

    pooled_std = np.sqrt(((n_a - 1) * std_a**2 + (n_b - 1) * std_b**2) / (n_a + n_b - 2))

    if pooled_std == 0:
        d = 0.0
    else:
        d = (mean_b - mean_a) / pooled_std

    # Interpret magnitude
    abs_d = abs(d)
    if abs_d < 0.2:
        magnitude = 'negligible'
    elif abs_d < 0.5:
        magnitude = 'small'
    elif abs_d < 0.8:
        magnitude = 'medium'
    else:
        magnitude = 'large'

    return {
        'cohens_d': float(d),
        'magnitude': magnitude
    }


def analyze_ablation_results(results_file: Path) -> Dict:
    """
    Perform comprehensive statistical analysis on ablation results.

    Args:
        results_file: Path to ablation_seed{X}.json

    Returns:
        Statistical analysis report
    """
    with open(results_file, 'r') as f:
        results = json.load(f)

    # Extract proposed method scores
    proposed = next(r for r in results if r['ablation_type'] == 'proposed')
    proposed_scores = proposed['scores']

    analysis = {
        'sample_size': len(proposed_scores),
        'methods': {}
    }

    for result in results:
        method = result['ablation_type']
        scores = result['scores']

        # Bootstrap CI
        mean, ci_lower, ci_upper = bootstrap_ci(scores)

        method_analysis = {
            'mean': mean,
            'std': float(np.std(scores)),
            'scores': scores,
            'bootstrap_95ci': [ci_lower, ci_upper],
            'ci_width': ci_upper - ci_lower
        }

        # Compare to proposed if not proposed itself
        if method != 'proposed':
            # Wilcoxon test
            wilcoxon_result = wilcoxon_test(scores, proposed_scores)
            method_analysis['vs_proposed'] = {
                'wilcoxon_test': wilcoxon_result,
                'effect_size': cohens_d(scores, proposed_scores),
                'mean_difference': float(np.mean(proposed_scores) - np.mean(scores))
            }

        analysis['methods'][method] = method_analysis

    # Overall assessment
    analysis['assessment'] = assess_statistical_power(analysis)

    return analysis


def assess_statistical_power(analysis: Dict) -> Dict:
    """
    Assess statistical power and validity of conclusions.

    Args:
        analysis: Analysis dictionary from analyze_ablation_results

    Returns:
        Assessment with warnings and recommendations
    """
    n = analysis['sample_size']

    warnings = []
    recommendations = []

    # Sample size assessment
    if n < 10:
        warnings.append(f"Very small sample size (n={n}). Results may be unreliable.")
        recommendations.append(f"Increase to at least n=20 for robust conclusions.")
    elif n < 20:
        warnings.append(f"Small sample size (n={n}). Limited statistical power.")
        recommendations.append(f"Consider increasing to n=30-50 for publication.")

    # Check CI widths
    for method, data in analysis['methods'].items():
        ci_width = data['ci_width']
        if ci_width > 1.5:
            warnings.append(f"{method}: Wide confidence interval ({ci_width:.2f}). High uncertainty.")

    # Check for perfect scores
    for method, data in analysis['methods'].items():
        if data['std'] == 0:
            warnings.append(f"{method}: Zero variance. Possible ceiling effect or measurement issue.")

    return {
        'sample_size': n,
        'power': 'low' if n < 20 else 'moderate' if n < 50 else 'high',
        'warnings': warnings,
        'recommendations': recommendations
    }


def generate_statistical_report(
    persona_id: str,
    seed: int,
    output_dir: Path = None
):
    """
    Generate statistical analysis report for ablation study.

    Args:
        persona_id: Persona identifier
        seed: Random seed
        output_dir: Output directory (default: reports/{persona_id}/phase1/ablation/)
    """
    if output_dir is None:
        output_dir = Path(f"reports/{persona_id}/phase1/ablation")

    results_file = output_dir / f"ablation_seed{seed}.json"

    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")

    # Perform analysis
    analysis = analyze_ablation_results(results_file)

    # Save detailed JSON
    stats_file = output_dir / f"ablation_seed{seed}_statistics.json"
    with open(stats_file, 'w') as f:
        json.dump(analysis, f, indent=2)

    print(f"✓ Saved statistical analysis: {stats_file}")

    # Generate markdown report
    md_lines = [
        f"# Statistical Analysis - {persona_id}",
        "",
        f"**Sample Size**: {analysis['sample_size']} prompts per method",
        f"**Statistical Power**: {analysis['assessment']['power']}",
        "",
        "## Warnings",
        ""
    ]

    for warning in analysis['assessment']['warnings']:
        md_lines.append(f"⚠️ {warning}")

    md_lines.extend(["", "## Recommendations", ""])
    for rec in analysis['assessment']['recommendations']:
        md_lines.append(f"📋 {rec}")

    md_lines.extend(["", "## Results with Confidence Intervals", "", "| Method | Mean | 95% CI | vs Proposed |", "|--------|------|--------|-------------|"])

    for method, data in analysis['methods'].items():
        ci_lower, ci_upper = data['bootstrap_95ci']
        ci_str = f"[{ci_lower:.2f}, {ci_upper:.2f}]"

        if method == 'proposed':
            vs_str = "—"
        else:
            vs_data = data['vs_proposed']
            p_val = vs_data['wilcoxon_test']['p_value']
            if p_val is None:
                vs_str = "N/A"
            elif p_val < 0.05:
                vs_str = f"p={p_val:.3f} ✓"
            else:
                vs_str = f"p={p_val:.3f} ✗"

        md_lines.append(f"| {method} | {data['mean']:.2f} | {ci_str} | {vs_str} |")

    md_path = output_dir / f"ablation_seed{seed}_statistics.md"
    with open(md_path, 'w') as f:
        f.write('\n'.join(md_lines))

    print(f"✓ Saved statistical report: {md_path}")

    return analysis


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Statistical analysis for ablation study")
    parser.add_argument('--persona-id', required=True, help='Persona ID')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')

    args = parser.parse_args()

    analysis = generate_statistical_report(args.persona_id, args.seed)

    print("\n" + "="*80)
    print("Statistical Analysis Complete")
    print("="*80)
