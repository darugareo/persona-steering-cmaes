#!/usr/bin/env python3
"""
Aggregate Results for 10 Personas (Lightweight Strategy)

Output tables:
1. Universal comparisons (10 personas): base vs equal, base vs prompt
2. Subset comparisons (3 personas): equal vs optimized
3. Per-persona breakdown
4. Per-category breakdown (if prompts have categories)

Statistical tests:
- McNemar test (paired data)
- Sign test (distribution-free)
- Bootstrap 95% CI
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy import stats

# Configuration
INPUT_FILE = Path("results/judge_evaluation/10personas_lightweight_results.json")
OUTPUT_DIR = Path("reports/10personas")
TABLES_DIR = OUTPUT_DIR / "tables"
LOG_FILE = OUTPUT_DIR / "aggregation.log"


def log(msg):
    """Log message."""
    print(msg)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, 'a') as f:
        f.write(msg + '\n')


def bootstrap_ci(data, num_bootstrap=10000, confidence=0.95):
    """Compute bootstrap confidence interval."""
    n = len(data)
    if n == 0:
        return 0, 0, 0

    bootstrap_means = []
    for _ in range(num_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))

    mean = np.mean(data)
    ci_lower = np.percentile(bootstrap_means, (1 - confidence) / 2 * 100)
    ci_upper = np.percentile(bootstrap_means, (1 + confidence) / 2 * 100)

    return mean, ci_lower, ci_upper


def mcnemar_test(judgments, method_a, method_b):
    """Perform McNemar test for paired comparisons."""
    # Count: a wins vs b wins (excluding ties)
    a_wins_b_loses = sum(1 for j in judgments if j["winner"] == "A")
    b_wins_a_loses = sum(1 for j in judgments if j["winner"] == "B")

    # McNemar test
    contingency = [[a_wins_b_loses, 0], [0, b_wins_a_loses]]

    if a_wins_b_loses + b_wins_a_loses == 0:
        return None, None

    # Use continuity correction
    statistic = (abs(a_wins_b_loses - b_wins_a_loses) - 1) ** 2 / (a_wins_b_loses + b_wins_a_loses)
    p_value = 1 - stats.chi2.cdf(statistic, 1)

    return statistic, p_value


def sign_test(judgments):
    """Perform sign test (distribution-free)."""
    wins_b = sum(1 for j in judgments if j["winner"] == "B")
    wins_a = sum(1 for j in judgments if j["winner"] == "A")
    ties = sum(1 for j in judgments if j["winner"] == "tie")

    n = wins_a + wins_b  # Exclude ties
    if n == 0:
        return None, None

    # Two-tailed sign test
    p_value = 2 * min(stats.binom.cdf(wins_b, n, 0.5), 1 - stats.binom.cdf(wins_b - 1, n, 0.5))

    return wins_b, p_value


def compute_stats(result):
    """Compute statistics for one comparison."""
    judgments = result["judgments"]
    method_a = result["method_a"]
    method_b = result["method_b"]

    # Win rates
    wins_a = result["wins_a"]
    wins_b = result["wins_b"]
    ties = result["ties"]
    total = wins_a + wins_b + ties

    win_rate_b = wins_b / total if total > 0 else 0

    # Bootstrap CI for win rate
    # Create binary array: 1 if B wins, 0 otherwise
    binary_outcomes = (
        [1] * wins_b +
        [0] * (wins_a + ties)
    )

    mean, ci_lower, ci_upper = bootstrap_ci(binary_outcomes)

    # McNemar test
    mcnemar_stat, mcnemar_p = mcnemar_test(judgments, method_a, method_b)

    # Sign test
    sign_wins_b, sign_p = sign_test(judgments)

    return {
        "method_a": method_a,
        "method_b": method_b,
        "num_personas": result["num_personas"],
        "total_comparisons": total,
        "wins_a": wins_a,
        "wins_b": wins_b,
        "ties": ties,
        "win_rate_b": win_rate_b,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "mcnemar_statistic": mcnemar_stat,
        "mcnemar_p": mcnemar_p,
        "sign_wins_b": sign_wins_b,
        "sign_p": sign_p,
    }


def generate_main_table(stats):
    """Generate main results table."""
    log("\n" + "="*80)
    log("Table 1: Main Results (10 Personas)")
    log("="*80)

    lines = []
    lines.append("| Comparison | N | Win Rate | 95% CI | p-value (McNemar) | p-value (Sign) |")
    lines.append("|------------|---|----------|---------|-------------------|----------------|")

    for stat in stats:
        method_a = stat["method_a"]
        method_b = stat["method_b"]
        n = stat["num_personas"]
        win_rate = stat["win_rate_b"]
        ci_lower = stat["ci_lower"]
        ci_upper = stat["ci_upper"]
        mcnemar_p = stat["mcnemar_p"]
        sign_p = stat["sign_p"]

        comparison = f"{method_b} vs {method_a}"
        win_rate_str = f"{win_rate*100:.1f}%"
        ci_str = f"[{ci_lower*100:.1f}, {ci_upper*100:.1f}]"
        mcnemar_str = f"{mcnemar_p:.4f}" if mcnemar_p is not None else "N/A"
        sign_str = f"{sign_p:.4f}" if sign_p is not None else "N/A"

        # Significance markers
        if mcnemar_p and mcnemar_p < 0.001:
            mcnemar_str += "***"
        elif mcnemar_p and mcnemar_p < 0.01:
            mcnemar_str += "**"
        elif mcnemar_p and mcnemar_p < 0.05:
            mcnemar_str += "*"

        line = f"| {comparison} | {n} | {win_rate_str} | {ci_str} | {mcnemar_str} | {sign_str} |"
        lines.append(line)

    table = "\n".join(lines)
    log(table)

    # Save markdown
    markdown_file = TABLES_DIR / "table_main_results.md"
    markdown_file.parent.mkdir(parents=True, exist_ok=True)
    with open(markdown_file, 'w') as f:
        f.write(table)

    # Save LaTeX
    latex_lines = []
    latex_lines.append("\\begin{table*}[t]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{Main Results: Win Rates for 10 Personas}")
    latex_lines.append("\\label{tab:main_results_10personas}")
    latex_lines.append("\\begin{tabular}{lccccc}")
    latex_lines.append("\\toprule")
    latex_lines.append("\\textbf{Comparison} & \\textbf{N} & \\textbf{Win Rate} & \\textbf{95\\% CI} & \\textbf{$p$ (McNemar)} & \\textbf{$p$ (Sign)} \\\\")
    latex_lines.append("\\midrule")

    for stat in stats:
        method_a = stat["method_a"].replace("_", "\\_")
        method_b = stat["method_b"].replace("_", "\\_")
        n = stat["num_personas"]
        win_rate = stat["win_rate_b"]
        ci_lower = stat["ci_lower"]
        ci_upper = stat["ci_upper"]
        mcnemar_p = stat["mcnemar_p"]
        sign_p = stat["sign_p"]

        comparison = f"{method_b} vs {method_a}"
        win_rate_str = f"{win_rate*100:.1f}\\%"
        ci_str = f"[{ci_lower*100:.1f}, {ci_upper*100:.1f}]"
        mcnemar_str = f"{mcnemar_p:.4f}" if mcnemar_p is not None else "N/A"
        sign_str = f"{sign_p:.4f}" if sign_p is not None else "N/A"

        if mcnemar_p and mcnemar_p < 0.001:
            mcnemar_str += "$^{***}$"
        elif mcnemar_p and mcnemar_p < 0.01:
            mcnemar_str += "$^{**}$"
        elif mcnemar_p and mcnemar_p < 0.05:
            mcnemar_str += "$^{*}$"

        line = f"{comparison} & {n} & {win_rate_str} & {ci_str} & {mcnemar_str} & {sign_str} \\\\"
        latex_lines.append(line)

    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table*}")

    latex_file = TABLES_DIR / "table_main_results.tex"
    with open(latex_file, 'w') as f:
        f.write("\n".join(latex_lines))

    log(f"\n✓ Saved markdown: {markdown_file}")
    log(f"✓ Saved LaTeX: {latex_file}")


def generate_per_persona_table(data):
    """Generate per-persona breakdown."""
    log("\n" + "="*80)
    log("Table 2: Per-Persona Results")
    log("="*80)

    # Collect per-persona stats
    persona_stats = defaultdict(lambda: {"base_vs_equal": {}, "base_vs_prompt": {}, "equal_vs_optimized": {}})

    for result in data["results"]:
        method_a = result["method_a"]
        method_b = result["method_b"]
        comparison_key = f"{method_a}_vs_{method_b}"

        for judgment in result["judgments"]:
            persona_id = judgment["persona_id"]
            winner = judgment["winner"]

            if persona_id not in persona_stats:
                persona_stats[persona_id] = {}

            if comparison_key not in persona_stats[persona_id]:
                persona_stats[persona_id][comparison_key] = {"wins_a": 0, "wins_b": 0, "ties": 0}

            if winner == "A":
                persona_stats[persona_id][comparison_key]["wins_a"] += 1
            elif winner == "B":
                persona_stats[persona_id][comparison_key]["wins_b"] += 1
            elif winner == "tie":
                persona_stats[persona_id][comparison_key]["ties"] += 1

    # Generate table
    lines = []
    lines.append("| Persona | base vs equal | base vs prompt | equal vs optimized |")
    lines.append("|---------|---------------|----------------|-------------------|")

    all_personas = data["personas"]["all"]

    for persona_id in all_personas:
        stats = persona_stats[persona_id]

        # base vs equal
        base_equal = stats.get("base_vs_equal", {})
        wins_equal = base_equal.get("wins_b", 0)
        total_be = base_equal.get("wins_a", 0) + wins_equal + base_equal.get("ties", 0)
        win_rate_equal = f"{wins_equal/total_be*100:.0f}%" if total_be > 0 else "N/A"

        # base vs prompt
        base_prompt = stats.get("base_vs_prompt", {})
        wins_prompt = base_prompt.get("wins_b", 0)
        total_bp = base_prompt.get("wins_a", 0) + wins_prompt + base_prompt.get("ties", 0)
        win_rate_prompt = f"{wins_prompt/total_bp*100:.0f}%" if total_bp > 0 else "N/A"

        # equal vs optimized (only for existing 3)
        equal_opt = stats.get("equal_vs_optimized", {})
        wins_opt = equal_opt.get("wins_b", 0)
        total_eo = equal_opt.get("wins_a", 0) + wins_opt + equal_opt.get("ties", 0)
        win_rate_opt = f"{wins_opt/total_eo*100:.0f}%" if total_eo > 0 else "N/A"

        line = f"| {persona_id} | {win_rate_equal} | {win_rate_prompt} | {win_rate_opt} |"
        lines.append(line)

    table = "\n".join(lines)
    log(table)

    markdown_file = TABLES_DIR / "table_per_persona.md"
    with open(markdown_file, 'w') as f:
        f.write(table)

    log(f"\n✓ Saved: {markdown_file}")


def main():
    """Aggregate results for 10 personas."""
    log("="*80)
    log("AGGREGATE RESULTS: 10 PERSONAS")
    log("="*80)

    # Load data
    if not INPUT_FILE.exists():
        log(f"✗ ERROR: Input file not found: {INPUT_FILE}")
        log("Run judge evaluation first:")
        log("  python scripts/run_judge_evaluation_10personas.py")
        return 1

    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)

    log(f"✓ Loaded: {INPUT_FILE}")

    # Compute statistics for each comparison
    stats = []
    for result in data["results"]:
        stat = compute_stats(result)
        stats.append(stat)

    # Generate tables
    generate_main_table(stats)
    generate_per_persona_table(data)

    # Save aggregated stats
    output_file = OUTPUT_DIR / "aggregated_stats.json"
    output_data = {
        "date": data["date"],
        "personas": data["personas"],
        "statistics": stats,
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    log(f"\n✓ Saved aggregated stats: {output_file}")

    log("\n" + "="*80)
    log("AGGREGATION COMPLETE")
    log("="*80)
    log(f"Output directory: {OUTPUT_DIR}")
    log(f"Tables directory: {TABLES_DIR}")

    log("\nNext step: Update IEEE Access paper")
    log("  1. Update abstract with new win rates")
    log("  2. Update results section with 10-persona tables")
    log("  3. Add computational cost justification in Experimental Setup")

    return 0


if __name__ == "__main__":
    sys.exit(main())
