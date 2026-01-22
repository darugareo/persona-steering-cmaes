"""
修正① Table I の正しい Win Rate 計算

Win Rate 定義: 左側（row）の手法が勝った割合

現在の問題:
- Table I: Optimized vs Equal = 34.3% (実際は負け率を書いている)
- 正しくは: Optimized vs Equal = 65.7% (勝率)
"""

import json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest

# Configuration
INPUT_FILE = Path("results/judge_evaluation/10personas_llama3_persona_aware_results.json")
OUTPUT_TEX = Path("paper/tables/table1_corrected.tex")
OUTPUT_MD = Path("paper/tables/table1_corrected.md")


def calculate_confidence_interval(wins, total, confidence=0.95):
    """Wilson score confidence interval"""
    from scipy import stats
    if total == 0:
        return (0.0, 0.0)

    p = wins / total
    z = stats.norm.ppf((1 + confidence) / 2)
    denominator = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denominator
    offset = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator

    return (max(0, centre - offset), min(1, centre + offset))


def main():
    print("="*70)
    print("修正① Table I の正しい Win Rate 計算")
    print("="*70 + "\n")

    # Load data
    with open(INPUT_FILE) as f:
        data = json.load(f)

    print(f"Data: {INPUT_FILE.name}")
    print(f"Personas: {len(data['personas']['all'])}")
    print(f"Judge: {data['judge_models']['main']}\n")

    # Process comparisons (exclude prompt)
    results = []
    for result in data['results']:
        method_a = result['method_a']
        method_b = result['method_b']

        # Skip prompt comparisons
        if 'prompt' in method_a or 'prompt' in method_b:
            continue

        wins_a = result['wins_a']
        wins_b = result['wins_b']
        ties = result['ties']
        total = result['total_comparisons']

        # Calculate win rate for row method (method_a)
        decisive = wins_a + wins_b
        if decisive > 0:
            win_rate = wins_a / decisive * 100  # Exclude ties
            ci = calculate_confidence_interval(wins_a, decisive)
        else:
            win_rate = 0.0
            ci = (0.0, 0.0)

        # Statistical test
        if decisive > 0:
            test_result = binomtest(wins_a, decisive, 0.5, alternative='two-sided')
            p_value = test_result.pvalue
        else:
            p_value = 1.0

        results.append({
            'method_a': method_a,
            'method_b': method_b,
            'wins_a': wins_a,
            'wins_b': wins_b,
            'ties': ties,
            'total': total,
            'win_rate': win_rate,
            'ci_lower': ci[0] * 100,
            'ci_upper': ci[1] * 100,
            'p_value': p_value,
            'significant': p_value < 0.05
        })

    # Print results
    print("="*70)
    print("CORRECTED WIN RATES")
    print("="*70)
    for r in results:
        sig_marker = "***" if r['p_value'] < 0.001 else "**" if r['p_value'] < 0.01 else "*" if r['p_value'] < 0.05 else "n.s."
        print(f"\n{r['method_a'].capitalize()} vs {r['method_b'].capitalize()}:")
        print(f"  Row method ({r['method_a']}) wins: {r['wins_a']}/{r['wins_a']+r['wins_b']} decisive")
        print(f"  Win Rate: {r['win_rate']:.1f}%")
        print(f"  95% CI: [{r['ci_lower']:.1f}%, {r['ci_upper']:.1f}%]")
        print(f"  p-value: {r['p_value']:.4f} {sig_marker}")

    # Create Markdown table
    md = f"""# Table I (CORRECTED): Pairwise Win Rates Across 10 Personas

**Win Rate Definition**: Percentage of times the row method won (excluding ties)

| Comparison | Row Method | Win Rate | 95% CI | p-value | Sig. |
|------------|------------|----------|--------|---------|------|
"""

    for r in results:
        sig_str = "***" if r['p_value'] < 0.001 else "**" if r['p_value'] < 0.01 else "*" if r['p_value'] < 0.05 else "n.s."
        md += f"| {r['method_a'].capitalize()} vs {r['method_b'].capitalize()} "
        md += f"| {r['method_a'].capitalize()} "
        md += f"| {r['win_rate']:.1f}% "
        md += f"| [{r['ci_lower']:.1f}, {r['ci_upper']:.1f}] "
        md += f"| {r['p_value']:.4f} "
        md += f"| {sig_str} |\n"

    md += f"""
**Notes**:
- N = 280 comparisons per pair (10 personas × 28 prompts)
- Win rate calculated excluding ties (ties: 77-88%)
- Statistical significance: * p<0.05, ** p<0.01, *** p<0.001
"""

    with open(OUTPUT_MD, 'w') as f:
        f.write(md)
    print(f"\n✓ Markdown saved to {OUTPUT_MD}")

    # Create LaTeX table
    latex = r"""\begin{table}[t]
\caption{Pairwise Win Rates Across 10 Personas (Llama-3-8B)}
\label{tab:pairwise_winrates}
\centering
\begin{tabular}{ll|c|c|c}
\toprule
\multicolumn{2}{c|}{\textbf{Comparison}} & \textbf{Win Rate} & \textbf{95\% CI} & \textbf{$p$-value} \\
\textbf{Row Method} & \textbf{vs.} & \textbf{(\%)} & & \\
\midrule
"""

    for r in results:
        method_a_cap = r['method_a'].replace('_', '-').capitalize()
        method_b_cap = r['method_b'].replace('_', '-').capitalize()

        if r['method_a'] == 'optimized':
            method_a_cap = 'Optimized'
        elif r['method_a'] == 'equal':
            method_a_cap = 'Equal-Wt'
        elif r['method_a'] == 'base':
            method_a_cap = 'Base'

        if r['method_b'] == 'optimized':
            method_b_cap = 'Optimized'
        elif r['method_b'] == 'equal':
            method_b_cap = 'Equal-Wt'
        elif r['method_b'] == 'base':
            method_b_cap = 'Base'

        pval_str = '$<$0.001' if r['p_value'] < 0.001 else f"{r['p_value']:.3f}"

        latex += f"{method_a_cap} & {method_b_cap} "
        latex += f"& {r['win_rate']:.1f} "
        latex += f"& [{r['ci_lower']:.0f}, {r['ci_upper']:.0f}] "
        latex += f"& {pval_str} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\footnotesize
\item Win rate calculated as (row method wins) / (decisive comparisons), excluding ties.
\item N = 280 comparisons per pair. Tie rate: 77--88\%.
\end{tablenotes}
\end{table}
"""

    with open(OUTPUT_TEX, 'w') as f:
        f.write(latex)
    print(f"✓ LaTeX saved to {OUTPUT_TEX}")

    # Verification against Table II
    print("\n" + "="*70)
    print("VERIFICATION vs Table II")
    print("="*70)
    print("\nTable I (pooled across all comparisons):")
    for r in results:
        if r['method_a'] == 'optimized' and r['method_b'] == 'base':
            print(f"  Optimized vs Base: {r['win_rate']:.1f}%")
        if r['method_a'] == 'optimized' and r['method_b'] == 'equal':
            print(f"  Optimized vs Equal: {r['win_rate']:.1f}%")

    print("\nTable II (persona-level average) - from paper:")
    print("  Optimized vs Base: 62.5% (persona-level)")
    print("  Optimized vs Equal: 61.0% (persona-level)")

    print("\nDifference explanation:")
    print("  Table I: Pooled win rate (all 280 comparisons together)")
    print("  Table II: Mean of per-persona win rates (average of 10 percentages)")
    print("  → Minor numerical differences are expected")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
