"""
Aggregate baseline comparison results across multiple seeds (1, 2, 3).
Generate final Table1 with mean ± std.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

def load_seed_results(results_dir: Path, seeds: list[int]) -> dict[int, dict]:
    """Load results from all seeds."""
    seed_results = {}
    for seed in seeds:
        result_file = results_dir / f"baseline_comparison_seed{seed}.json"
        if not result_file.exists():
            print(f"Warning: {result_file} not found, skipping seed {seed}")
            continue
        with open(result_file, 'r') as f:
            seed_results[seed] = json.load(f)
    return seed_results

def aggregate_results(seed_results: dict[int, dict]) -> pd.DataFrame:
    """Aggregate results across seeds to compute mean ± std."""
    # Collect scores by method
    method_scores = {}

    for seed, results in seed_results.items():
        for method_name, method_data in results.items():
            if method_name not in method_scores:
                method_scores[method_name] = []

            # Extract mean persona_fit score
            if 'persona_fit' in method_data:
                scores = method_data['persona_fit']
                mean_score = np.mean(scores)
                method_scores[method_name].append(mean_score)

    # Compute statistics
    stats = []
    for method_name, scores in method_scores.items():
        scores = np.array(scores)
        stats.append({
            'Method': method_name,
            'Mean': np.mean(scores),
            'Std': np.std(scores, ddof=1) if len(scores) > 1 else 0.0,
            'N_Seeds': len(scores)
        })

    df = pd.DataFrame(stats)
    df = df.sort_values('Mean', ascending=False)
    return df

def save_aggregate_table(df: pd.DataFrame, tables_dir: Path):
    """Save aggregate table in multiple formats."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Markdown format
    md_path = tables_dir / "table1_multiseed_final.md"
    with open(md_path, 'w') as f:
        f.write("# Table 1: Baseline Comparison (Seeds 1-3 Aggregated)\n\n")
        f.write(f"Generated: {timestamp}\n\n")
        f.write("| Method | Mean ± Std | N_Seeds |\n")
        f.write("|--------|-----------|----------|\n")
        for _, row in df.iterrows():
            f.write(f"| {row['Method']} | {row['Mean']:.3f} ± {row['Std']:.3f} | {row['N_Seeds']} |\n")

    print(f"Saved: {md_path}")

    # CSV format
    csv_path = tables_dir / "table1_multiseed_final.csv"
    df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"Saved: {csv_path}")

    # LaTeX format
    tex_path = tables_dir / "table1_multiseed_final.tex"
    with open(tex_path, 'w') as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{Baseline Comparison (Seeds 1-3)}\n")
        f.write("\\label{tab:baseline_multiseed}\n")
        f.write("\\begin{tabular}{lcc}\n")
        f.write("\\toprule\n")
        f.write("Method & Persona-Fit & Seeds \\\\\n")
        f.write("\\midrule\n")
        for _, row in df.iterrows():
            method = row['Method'].replace('_', '\\_')
            f.write(f"{method} & ${row['Mean']:.3f} \\pm {row['Std']:.3f}$ & {row['N_Seeds']} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"Saved: {tex_path}")

def main():
    # Setup paths
    base_dir = Path(__file__).parent.parent
    results_dir = base_dir / "reports" / "experiments" / "results"
    tables_dir = base_dir / "reports" / "experiments" / "tables"

    tables_dir.mkdir(parents=True, exist_ok=True)

    # Load results from seeds 1, 2, 3
    seeds = [1, 2, 3]
    print(f"Loading results from seeds: {seeds}")
    seed_results = load_seed_results(results_dir, seeds)

    if not seed_results:
        print("Error: No seed results found!")
        return

    print(f"Found results for seeds: {list(seed_results.keys())}")

    # Aggregate
    df = aggregate_results(seed_results)

    print("\n=== Aggregated Results ===")
    print(df.to_string(index=False))

    # Save
    save_aggregate_table(df, tables_dir)

    print("\n✓ Multi-seed aggregation complete!")

if __name__ == "__main__":
    main()
