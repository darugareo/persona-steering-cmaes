"""
Visualization Module
Visualize trait evolution and optimization progress
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


def load_run_data(run_dir: Path) -> pd.DataFrame:
    """
    Load all generation data from a run

    Args:
        run_dir: Path to run directory

    Returns:
        DataFrame with all records
    """
    records = []
    for jsonl_file in sorted(run_dir.glob('gen*.jsonl')):
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    records.append(record)

    if not records:
        raise ValueError(f"No data found in {run_dir}")

    return pd.DataFrame(records)


def plot_winrate_trend(run_dir: Path, save: bool = True) -> None:
    """
    Plot win rate trend across generations with confidence intervals

    Args:
        run_dir: Path to run directory
        save: If True, save figure to run directory
    """
    df = load_run_data(run_dir)

    # Group by generation
    gen_stats = []
    for gen in sorted(df['generation'].unique()):
        gen_data = df[df['generation'] == gen]

        # Calculate win rate (from fitness assuming tau=0.8)
        # fitness = win_rate + tie_rate * 0.8
        # For simplicity, use fitness directly
        fitness_values = gen_data['fitness'].values
        mean_fitness = fitness_values.mean()
        std_fitness = fitness_values.std()
        n = len(fitness_values)
        ci = 1.96 * std_fitness / np.sqrt(n) if n > 1 else 0

        gen_stats.append({
            'generation': gen,
            'mean_fitness': mean_fitness,
            'std_fitness': std_fitness,
            'ci_lower': mean_fitness - ci,
            'ci_upper': mean_fitness + ci,
            'max_fitness': fitness_values.max(),
            'min_fitness': fitness_values.min()
        })

    stats_df = pd.DataFrame(gen_stats)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Mean with CI
    ax.plot(stats_df['generation'], stats_df['mean_fitness'],
            marker='o', linewidth=2, label='Mean fitness', color='steelblue')
    ax.fill_between(stats_df['generation'],
                     stats_df['ci_lower'],
                     stats_df['ci_upper'],
                     alpha=0.3, color='steelblue', label='95% CI')

    # Max/min range
    ax.fill_between(stats_df['generation'],
                     stats_df['min_fitness'],
                     stats_df['max_fitness'],
                     alpha=0.1, color='gray', label='Min-Max range')

    # Best fitness line
    ax.plot(stats_df['generation'], stats_df['max_fitness'],
            marker='s', linewidth=1.5, linestyle='--',
            label='Best fitness', color='darkgreen')

    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Fitness (Win Rate)', fontsize=12)
    ax.set_title(f'Fitness Evolution - {run_dir.name}', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save:
        save_path = run_dir / 'plot_winrate_trend.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def plot_trait_evolution(run_dir: Path, save: bool = True) -> None:
    """
    Plot trait value evolution across generations

    Args:
        run_dir: Path to run directory
        save: If True, save figure to run directory
    """
    df = load_run_data(run_dir)

    # Extract trait values
    trait_cols = ['R1', 'R2', 'R3', 'R4', 'R5']
    for col in trait_cols:
        df[col] = df['traits'].apply(lambda x: x.get(col, np.nan))

    # Group by generation
    gen_trait_stats = []
    for gen in sorted(df['generation'].unique()):
        gen_data = df[df['generation'] == gen]
        for trait in trait_cols:
            values = gen_data[trait].dropna().values
            if len(values) > 0:
                gen_trait_stats.append({
                    'generation': gen,
                    'trait': trait,
                    'mean': values.mean(),
                    'std': values.std(),
                    'median': np.median(values),
                    'q25': np.percentile(values, 25),
                    'q75': np.percentile(values, 75)
                })

    stats_df = pd.DataFrame(gen_trait_stats)

    # Plot: one subplot per trait
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    trait_names = {
        'R1': 'Directness',
        'R2': 'Emotional Valence',
        'R3': 'Social Orientation',
        'R4': 'Audience Focus',
        'R5': 'Risk Orientation'
    }

    for i, trait in enumerate(trait_cols):
        ax = axes[i]
        trait_data = stats_df[stats_df['trait'] == trait]

        if len(trait_data) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title(f'{trait}: {trait_names[trait]}')
            continue

        # Mean line with quartile bands
        ax.plot(trait_data['generation'], trait_data['mean'],
                marker='o', linewidth=2, label='Mean', color='darkblue')
        ax.fill_between(trait_data['generation'],
                        trait_data['q25'],
                        trait_data['q75'],
                        alpha=0.3, color='skyblue', label='IQR (25-75%)')

        # Zero reference line
        ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

        ax.set_xlabel('Generation', fontsize=10)
        ax.set_ylabel('Trait Value', fontsize=10)
        ax.set_title(f'{trait}: {trait_names[trait]}', fontsize=11, fontweight='bold')
        ax.set_ylim(-1.1, 1.1)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused subplot
    axes[-1].axis('off')

    plt.suptitle(f'Trait Evolution - {run_dir.name}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save:
        save_path = run_dir / 'plot_trait_evolution.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def plot_trait_scatter(run_dir: Path, method: str = 'pca', save: bool = True) -> None:
    """
    Plot trait space scatter (2D projection) colored by fitness

    Args:
        run_dir: Path to run directory
        method: Dimensionality reduction method ('pca' or 'tsne')
        save: If True, save figure to run directory
    """
    df = load_run_data(run_dir)

    # Extract trait matrix
    trait_cols = ['R1', 'R2', 'R3', 'R4', 'R5']
    trait_matrix = []
    fitness_values = []
    generations = []

    for _, row in df.iterrows():
        traits = row['traits']
        trait_vector = [traits.get(col, 0.0) for col in trait_cols]
        trait_matrix.append(trait_vector)
        fitness_values.append(row['fitness'])
        generations.append(row['generation'])

    X = np.array(trait_matrix)
    fitness = np.array(fitness_values)
    gens = np.array(generations)

    # Dimensionality reduction
    if method == 'pca':
        reducer = PCA(n_components=2)
        X_2d = reducer.fit_transform(X)
        explained_var = reducer.explained_variance_ratio_
        xlabel = f'PC1 ({explained_var[0]:.1%})'
        ylabel = f'PC2 ({explained_var[1]:.1%})'
        title = f'Trait Space (PCA) - {run_dir.name}'
    elif method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X)-1))
        X_2d = reducer.fit_transform(X)
        xlabel = 't-SNE 1'
        ylabel = 't-SNE 2'
        title = f'Trait Space (t-SNE) - {run_dir.name}'
    else:
        raise ValueError(f"Unknown method: {method}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Scatter colored by fitness
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1],
                        c=fitness, cmap='viridis',
                        s=100, alpha=0.6, edgecolors='black', linewidth=0.5)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Fitness', fontsize=11)

    # Highlight best candidate
    best_idx = fitness.argmax()
    ax.scatter(X_2d[best_idx, 0], X_2d[best_idx, 1],
              marker='*', s=500, color='red', edgecolors='black', linewidth=2,
              label='Best candidate', zorder=10)

    # Annotate generations if small dataset
    if len(X) <= 50:
        for i in range(len(X)):
            ax.annotate(f'G{gens[i]}',
                       (X_2d[i, 0], X_2d[i, 1]),
                       fontsize=7, alpha=0.6,
                       xytext=(3, 3), textcoords='offset points')

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save:
        save_path = run_dir / f'plot_trait_scatter_{method}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def plot_trait_heatmap(run_dir: Path, save: bool = True) -> None:
    """
    Plot heatmap of trait values for top candidates

    Args:
        run_dir: Path to run directory
        save: If True, save figure to run directory
    """
    df = load_run_data(run_dir)

    # Sort by fitness and take top N
    df_sorted = df.sort_values('fitness', ascending=False).head(20)

    # Extract trait matrix
    trait_cols = ['R1', 'R2', 'R3', 'R4', 'R5']
    trait_matrix = []
    labels = []

    for idx, row in df_sorted.iterrows():
        traits = row['traits']
        trait_vector = [traits.get(col, 0.0) for col in trait_cols]
        trait_matrix.append(trait_vector)
        labels.append(f"G{row['generation']}-C{row['candidate_id']}")

    X = np.array(trait_matrix)

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(8, 10))

    trait_names = ['Directness', 'Emotion', 'Social', 'Audience', 'Risk']

    sns.heatmap(X, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                vmin=-1, vmax=1, linewidths=0.5,
                xticklabels=trait_names, yticklabels=labels,
                cbar_kws={'label': 'Trait Value'}, ax=ax)

    ax.set_title(f'Top 20 Candidates - Trait Heatmap\n{run_dir.name}',
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Trait', fontsize=12)
    ax.set_ylabel('Candidate (Gen-ID)', fontsize=12)

    plt.tight_layout()

    if save:
        save_path = run_dir / 'plot_trait_heatmap.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def visualize_all(run_dir: Path, save: bool = True):
    """
    Generate all visualization plots for a run

    Args:
        run_dir: Path to run directory
        save: If True, save all figures
    """
    print(f"\n{'='*60}")
    print(f"Generating visualizations for: {run_dir.name}")
    print(f"{'='*60}\n")

    try:
        plot_winrate_trend(run_dir, save=save)
        plot_trait_evolution(run_dir, save=save)
        plot_trait_scatter(run_dir, method='pca', save=save)
        plot_trait_heatmap(run_dir, save=save)
        print(f"\n✅ All visualizations complete!")
    except Exception as e:
        print(f"❌ Error during visualization: {e}")
        raise


def main():
    """CLI interface"""
    import argparse

    parser = argparse.ArgumentParser(description="Visualize CMA-ES optimization results")
    parser.add_argument('--logdir', type=str, help='Specific run directory to visualize')
    parser.add_argument('--logs_base', type=str, default='logs', help='Base logs directory')
    parser.add_argument('--plot', type=str, choices=['winrate', 'traits', 'scatter', 'heatmap', 'all'],
                       default='all', help='Which plot to generate')
    parser.add_argument('--method', type=str, choices=['pca', 'tsne'], default='pca',
                       help='Scatter plot method')
    parser.add_argument('--no-save', action='store_true', help='Do not save figures')

    args = parser.parse_args()

    logs_path = Path(args.logs_base)

    if args.logdir:
        run_dir = Path(args.logdir)
    else:
        # Find most recent run
        run_dirs = sorted(logs_path.glob('run_*'))
        if not run_dirs:
            print(f"No run directories found in {logs_path}")
            return
        run_dir = run_dirs[-1]

    save_figs = not args.no_save

    # Generate requested plots
    if args.plot == 'all':
        visualize_all(run_dir, save=save_figs)
    elif args.plot == 'winrate':
        plot_winrate_trend(run_dir, save=save_figs)
    elif args.plot == 'traits':
        plot_trait_evolution(run_dir, save=save_figs)
    elif args.plot == 'scatter':
        plot_trait_scatter(run_dir, method=args.method, save=save_figs)
    elif args.plot == 'heatmap':
        plot_trait_heatmap(run_dir, save=save_figs)


if __name__ == "__main__":
    main()
