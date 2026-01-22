"""
Generate layer × method heatmap from cross-layer evaluation results.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
import argparse

def load_cross_layer_results(results_file: Path) -> pd.DataFrame:
    """Load cross-layer evaluation results into a DataFrame."""
    with open(results_file, 'r') as f:
        results = json.load(f)

    data = []
    for r in results:
        data.append({
            'Method': r['method'],
            'Layer': r['layer'],
            'Score': r['mean_score']
        })

    return pd.DataFrame(data)

def create_heatmap(df: pd.DataFrame, output_file: Path):
    """Create and save heatmap visualization."""

    # Pivot for heatmap
    pivot = df.pivot(index='Method', columns='Layer', values='Score')

    # Reorder methods (best to worst based on layer 22)
    if 22 in pivot.columns:
        pivot = pivot.sort_values(by=22, ascending=False)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot heatmap
    sns.heatmap(
        pivot,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        vmin=0.0,
        vmax=1.0,
        cbar_kws={'label': 'Persona-Fit Score'},
        linewidths=0.5,
        ax=ax
    )

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Method', fontsize=12)
    ax.set_title('Cross-Layer Evaluation: Persona-Fit Scores', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Heatmap saved to: {output_file}")

    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-file', type=str, required=True,
                        help='Path to cross-layer evaluation results JSON')
    parser.add_argument('--output-file', type=str, default=None,
                        help='Output file path (default: reports/experiments/figures/layer_heatmap.png)')

    args = parser.parse_args()

    # Load results
    results_file = Path(args.results_file)
    if not results_file.exists():
        print(f"Error: {results_file} not found!")
        return

    df = load_cross_layer_results(results_file)

    print(f"Loaded {len(df)} results")
    print(f"Methods: {df['Method'].unique()}")
    print(f"Layers: {sorted(df['Layer'].unique())}")

    # Setup output path
    if args.output_file:
        output_file = Path(args.output_file)
    else:
        base_dir = Path(__file__).parent.parent
        figures_dir = base_dir / "reports" / "experiments" / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        output_file = figures_dir / "layer_heatmap.png"

    # Generate heatmap
    create_heatmap(df, output_file)

    # Also save the pivot table as CSV
    pivot = df.pivot(index='Method', columns='Layer', values='Score')
    csv_file = output_file.parent / "layer_heatmap_data.csv"
    pivot.to_csv(csv_file)
    print(f"✓ Data saved to: {csv_file}")

if __name__ == "__main__":
    main()
