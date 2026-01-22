"""Generate Table 1 from baseline comparison results (no pandas dependency)."""

import json
import argparse
from pathlib import Path

def generate_table1(results: dict, output_dir: Path):
    """Generate Table 1: Method Comparison."""
    print("\n" + "="*80)
    print("Generating Table 1: Method Comparison")
    print("="*80)

    # Extract and sort data
    rows = []
    for method_name, result in results.items():
        if 'metrics' not in result:
            continue
        
        metrics = result['metrics']
        config = result.get('config', {})
        
        steering_type = config.get('steering', 'N/A')
        if steering_type is None:
            steering_type = 'N/A'
        
        rows.append({
            'method': method_name,
            'display_name': method_name.replace('_', ' ').title(),
            'type': steering_type.replace('_', ' ').title(),
            'mean_score': metrics['mean_score'],
            'std': metrics['std_score'],
            'min': metrics['min_score'],
            'max': metrics['max_score'],
            'win_rate': metrics['win_rate'] * 100,
            'description': config.get('description', '')[:50],
        })
    
    # Sort by mean score descending
    rows.sort(key=lambda x: x['mean_score'], reverse=True)
    
    # Save as CSV
    csv_file = output_dir / "table1_method_comparison.csv"
    with open(csv_file, 'w') as f:
        f.write("Method,Type,Mean Score,Std,Min,Max,Win Rate,Description\n")
        for row in rows:
            f.write(f"{row['display_name']},{row['type']},"
                   f"{row['mean_score']:.2f},{row['std']:.2f},"
                   f"{row['min']:.1f},{row['max']:.1f},"
                   f"{row['win_rate']:.1f}%,{row['description']}\n")
    
    print(f"✓ CSV saved: {csv_file}")
    
    # Save as Markdown
    md_file = output_dir / "table1_method_comparison.md"
    with open(md_file, 'w') as f:
        f.write("# Table 1: Method Comparison\n\n")
        f.write("| Method | Type | Mean Score | Std | Min | Max | Win Rate |\n")
        f.write("|--------|------|------------|-----|-----|-----|----------|\n")
        for row in rows:
            f.write(f"| {row['display_name']} | {row['type']} | "
                   f"{row['mean_score']:.2f} | {row['std']:.2f} | "
                   f"{row['min']:.1f} | {row['max']:.1f} | "
                   f"{row['win_rate']:.1f}% |\n")
        f.write("\n**Metrics:**\n")
        f.write("- Mean Score: Average persona fit score (1-5 scale)\n")
        f.write("- Std: Standard deviation\n")
        f.write("- Win Rate: Percentage of steered responses preferred over baseline\n")
    
    print(f"✓ Markdown saved: {md_file}")
    
    # Save as LaTeX
    tex_file = output_dir / "table1_method_comparison.tex"
    with open(tex_file, 'w') as f:
        f.write("% Table 1: Method Comparison\n")
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{Baseline Method Comparison on Persona Steering}\n")
        f.write("\\label{tab:method_comparison}\n")
        f.write("\\begin{tabular}{lllrrrr}\n")
        f.write("\\toprule\n")
        f.write("Method & Type & Mean Score & Std & Min & Max & Win Rate \\\\\n")
        f.write("\\midrule\n")
        
        for row in rows:
            f.write(f"{row['display_name']} & {row['type']} & "
                   f"{row['mean_score']:.2f} & {row['std']:.2f} & "
                   f"{row['min']:.1f} & {row['max']:.1f} & "
                   f"{row['win_rate']:.1f}\\% \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"✓ LaTeX saved: {tex_file}")
    
    # Print to console
    print("\n" + "="*80)
    print("TABLE 1: METHOD COMPARISON")
    print("="*80)
    print(f"{'Method':<20} {'Mean Score':<12} {'Win Rate':<10}")
    print("-"*80)
    for row in rows:
        print(f"{row['display_name']:<20} {row['mean_score']:.2f} ± {row['std']:.2f}    "
              f"{row['win_rate']:.1f}%")
    print("="*80)

def generate_summary(results: dict, output_dir: Path):
    """Generate summary report."""
    print("\n" + "="*80)
    print("Generating Summary Report")
    print("="*80)
    
    report_file = output_dir / "baseline_comparison_summary.md"
    
    with open(report_file, 'w') as f:
        f.write("# Baseline Comparison Summary\n\n")
        f.write("## Overview\n\n")
        f.write(f"Total methods evaluated: {len(results)}\n\n")
        
        # Rankings
        methods_with_metrics = [
            (name, result) for name, result in results.items()
            if 'metrics' in result
        ]
        
        sorted_methods = sorted(
            methods_with_metrics,
            key=lambda x: x[1]['metrics']['mean_score'],
            reverse=True
        )
        
        f.write("## Performance Rankings\n\n")
        f.write("### By Mean Score\n\n")
        for rank, (method_name, result) in enumerate(sorted_methods, 1):
            metrics = result['metrics']
            f.write(f"{rank}. **{method_name.replace('_', ' ').title()}**: "
                   f"{metrics['mean_score']:.2f} ± {metrics['std_score']:.2f}\n")
        
        # Key findings
        f.write("\n## Key Findings\n\n")
        best_method = sorted_methods[0]
        best_name = best_method[0].replace('_', ' ').title()
        best_score = best_method[1]['metrics']['mean_score']
        
        f.write(f"- **Best performing method**: {best_name} ({best_score:.2f}/5.0)\n")
        
        scores = [m['metrics']['mean_score'] for _, m in methods_with_metrics]
        f.write(f"- **Score range**: {min(scores):.2f} - {max(scores):.2f}\n")
    
    print(f"✓ Summary report saved: {report_file}")

def main():
    parser = argparse.ArgumentParser(description="Generate tables (no pandas)")
    parser.add_argument('--results-file', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default="reports/experiments/tables")
    
    args = parser.parse_args()
    
    # Load results
    with open(args.results_file, 'r') as f:
        results = json.load(f)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate outputs
    generate_table1(results, output_dir)
    generate_summary(results, output_dir)
    
    print("\n" + "="*80)
    print("ALL OUTPUTS GENERATED")
    print("="*80)
    print(f"✓ Table 1: {output_dir}/table1_method_comparison.*")
    print(f"✓ Summary: {output_dir}/baseline_comparison_summary.md")

if __name__ == "__main__":
    main()
