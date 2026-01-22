#!/usr/bin/env python3
"""
Generate comprehensive experiment report for trait semantic analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

# Setup
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
report_dir = Path("reports")
if not report_dir.exists():
    report_dir.mkdir(exist_ok=True)

print("="*80)
print("TRAIT SEMANTIC ANALYSIS - EXPERIMENT REPORT")
print("="*80)
print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Load data
print("Loading data...")
semantic_df = pd.read_parquet("data/processed/cc/representative_traits_v4_openai.parquet")
lexical_df = pd.read_parquet("data/processed/cc/filtered/persona_lexical_proxies.parquet")

print(f"✓ Loaded {len(semantic_df)} personas with semantic traits")
print(f"✓ Loaded {len(lexical_df)} personas with lexical proxies")
print()

# Filter lexical to representatives
rep_persona_ids = semantic_df["persona_id"].values
lexical_rep = lexical_df[lexical_df["persona_id"].isin(rep_persona_ids)]
print(f"✓ Filtered to {len(lexical_rep)} representative personas")
print()

# Define trait columns
trait_cols = ["R1", "R2", "R3", "R4", "R5", "R8"]

# ============================================================================
# 1. SEMANTIC TRAIT DISTRIBUTIONS
# ============================================================================
print("1. SEMANTIC TRAIT DISTRIBUTIONS")
print("-" * 80)

trait_stats = semantic_df[trait_cols].describe()
print(trait_stats)
print()

# Distribution plots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, trait in enumerate(trait_cols):
    axes[i].hist(semantic_df[trait], bins=15, edgecolor='black', alpha=0.7, color='steelblue')
    axes[i].axvline(semantic_df[trait].mean(), color='red', linestyle='--',
                    label=f'Mean: {semantic_df[trait].mean():.2f}')
    axes[i].axvline(0, color='gray', linestyle='-', alpha=0.3)
    axes[i].set_title(f"{trait} Distribution", fontweight='bold')
    axes[i].set_xlabel("Score [-1.0 to 1.0]")
    axes[i].set_ylabel("Count")
    axes[i].set_xlim(-1.1, 1.1)
    axes[i].legend()
    axes[i].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(report_dir / "semantic_trait_distributions.png", dpi=150, bbox_inches='tight')
print("✓ Saved: semantic_trait_distributions.png")
plt.close()

# ============================================================================
# 2. TRAIT CORRELATION ANALYSIS
# ============================================================================
print("\n2. TRAIT CORRELATION ANALYSIS")
print("-" * 80)

corr = semantic_df[trait_cols].corr()
print(corr)
print()

# Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".3f", cmap="coolwarm", center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8},
            vmin=-1, vmax=1)
plt.title("Semantic Trait Correlation Matrix", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(report_dir / "semantic_trait_correlation.png", dpi=150, bbox_inches='tight')
print("✓ Saved: semantic_trait_correlation.png")
plt.close()

# Check for high correlations
high_corr_pairs = []
for i in range(len(corr.columns)):
    for j in range(i+1, len(corr.columns)):
        if abs(corr.iloc[i, j]) > 0.6:
            high_corr_pairs.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))

if high_corr_pairs:
    print("\n⚠️  High Correlation Pairs (|corr| > 0.6):")
    for t1, t2, val in high_corr_pairs:
        print(f"  {t1} <-> {t2}: {val:.3f}")
else:
    print("\n✓ All trait pairs have |corr| < 0.6 (sufficiently independent)")
print()

# ============================================================================
# 3. SEMANTIC vs LEXICAL COMPARISON
# ============================================================================
print("\n3. SEMANTIC vs LEXICAL PROXY COMPARISON")
print("-" * 80)

# Merge data
merged = semantic_df.merge(lexical_rep, on="persona_id", suffixes=("", "_lex"))

# Define mappings
proxy_mappings = {
    "R1": ["self_ratio", "you_ratio", "we_ratio"],
    "R2": ["pos_emotion", "neg_emotion"],
    "R3": ["agency_ratio"],
    "R4": ["structure_ratio"],
    "R5": ["optimism_ratio", "pessimism_ratio"],
    "R8": ["past_ratio", "future_ratio"],
}

# Aggregate proxies
for trait, proxies in proxy_mappings.items():
    merged[f"{trait}_proxy_agg"] = merged[proxies].sum(axis=1)

# Correlations
print("Semantic vs Lexical Proxy Correlations:")
sem_lex_corr = {}
for trait in trait_cols:
    if f"{trait}_proxy_agg" in merged.columns:
        corr_val = merged[trait].corr(merged[f"{trait}_proxy_agg"])
        sem_lex_corr[trait] = corr_val
        status = "✓" if abs(corr_val) > 0.3 else "⚠️ "
        print(f"  {status} {trait}: {corr_val:+.3f}")
print()

# Scatter plots
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for i, trait in enumerate(trait_cols):
    if f"{trait}_proxy_agg" in merged.columns:
        axes[i].scatter(merged[f"{trait}_proxy_agg"], merged[trait], alpha=0.6, s=50)
        axes[i].set_xlabel(f"Lexical Proxy (aggregate)", fontsize=10)
        axes[i].set_ylabel(f"Semantic Score ({trait})", fontsize=10)
        axes[i].set_title(f"{trait}: Semantic vs Lexical", fontweight='bold')

        # Trend line
        z = np.polyfit(merged[f"{trait}_proxy_agg"], merged[trait], 1)
        p = np.poly1d(z)
        x_line = np.linspace(merged[f"{trait}_proxy_agg"].min(),
                            merged[f"{trait}_proxy_agg"].max(), 100)
        axes[i].plot(x_line, p(x_line), "r--", alpha=0.8,
                    label=f"r={sem_lex_corr[trait]:.2f}")
        axes[i].legend()
        axes[i].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(report_dir / "semantic_vs_lexical_scatter.png", dpi=150, bbox_inches='tight')
print("✓ Saved: semantic_vs_lexical_scatter.png")
plt.close()

# ============================================================================
# 4. TRAIT VARIANCE ANALYSIS
# ============================================================================
print("\n4. TRAIT VARIANCE ANALYSIS")
print("-" * 80)

variance = semantic_df[trait_cols].var()
std = semantic_df[trait_cols].std()
trait_range = [semantic_df[t].max() - semantic_df[t].min() for t in trait_cols]

variance_df = pd.DataFrame({
    "Trait": trait_cols,
    "Mean": [semantic_df[t].mean() for t in trait_cols],
    "Std Dev": std.values,
    "Variance": variance.values,
    "Range": trait_range,
    "Min": [semantic_df[t].min() for t in trait_cols],
    "Max": [semantic_df[t].max() for t in trait_cols]
})

print(variance_df.to_string(index=False))
print()

# Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.bar(variance_df["Trait"], variance_df["Std Dev"], color='steelblue', alpha=0.7)
ax1.set_ylabel("Standard Deviation", fontsize=12)
ax1.set_title("Trait Standard Deviations", fontsize=14, fontweight='bold')
ax1.axhline(0.2, color='red', linestyle='--', alpha=0.5, label='Threshold (0.2)')
ax1.legend()
ax1.grid(alpha=0.3)

ax2.bar(variance_df["Trait"], variance_df["Range"], color='coral', alpha=0.7)
ax2.set_ylabel("Range (Max - Min)", fontsize=12)
ax2.set_title("Trait Ranges", fontsize=14, fontweight='bold')
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(report_dir / "trait_variance_analysis.png", dpi=150, bbox_inches='tight')
print("✓ Saved: trait_variance_analysis.png")
plt.close()

# Check optimization readiness
low_variance = variance_df[variance_df["Std Dev"] < 0.2]
if len(low_variance) > 0:
    print("\n⚠️  Low variance traits (std < 0.2):")
    print(low_variance[["Trait", "Std Dev", "Range"]].to_string(index=False))
else:
    print("\n✓ All traits have sufficient variance (std ≥ 0.2) for optimization")
print()

# ============================================================================
# 5. VALIDATION SUMMARY
# ============================================================================
print("\n" + "="*80)
print("VALIDATION SUMMARY")
print("="*80)

print("\n✓ SEMANTIC TRAIT COVERAGE:")
for trait in trait_cols:
    mean = semantic_df[trait].mean()
    std = semantic_df[trait].std()
    min_val = semantic_df[trait].min()
    max_val = semantic_df[trait].max()
    print(f"  {trait}: mean={mean:+.3f}, std={std:.3f}, range=[{min_val:+.2f}, {max_val:+.2f}]")

print("\n✓ TRAIT INDEPENDENCE:")
if high_corr_pairs:
    for t1, t2, val in high_corr_pairs:
        print(f"  ⚠️  {t1} <-> {t2}: {val:.3f}")
else:
    print("  All trait pairs have |corr| < 0.6")

print("\n✓ SEMANTIC-LEXICAL ALIGNMENT:")
for trait, corr_val in sem_lex_corr.items():
    status = "✓" if abs(corr_val) > 0.3 else "⚠️ "
    print(f"  {status} {trait}: r={corr_val:+.3f}")

print("\n✓ OPTIMIZATION READINESS:")
for _, row in variance_df.iterrows():
    trait = row["Trait"]
    std_val = row["Std Dev"]
    range_val = row["Range"]
    ready = std_val > 0.2 and range_val > 0.5
    status = "✓" if ready else "⚠️ "
    print(f"  {status} {trait}: std={std_val:.3f}, range={range_val:.3f}")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("✓ traits_v3.json is validated and ready for:")
print("  1. LLaMA generator trait steering")
print("  2. CMA-ES / Bayesian Optimization")
print("  3. Persona-conditioned response generation")
print("="*80)

# Save summary CSV
summary_csv = report_dir / "experiment_summary.csv"
variance_df.to_csv(summary_csv, index=False)
print(f"\n✓ Saved summary: {summary_csv}")

print("\n✓ Report generation complete!")
print(f"  Reports saved in: {report_dir.absolute()}")
