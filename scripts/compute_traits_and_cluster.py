"""
Compute trait scores for filtered personas and perform clustering
"""
import pandas as pd
import numpy as np
import json
import re
from pathlib import Path
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 60)
print("TRAIT COMPUTATION AND CLUSTERING")
print("=" * 60)

# Paths
base_dir = Path("data/processed/cc")
filtered_dir = base_dir / "filtered"
reports_dir = Path("reports")
reports_dir.mkdir(exist_ok=True)

# Load filtered data
print("\n[1/8] Loading filtered persona data...")
persona_docs = pd.read_parquet(filtered_dir / "persona_docs_filtered.parquet")
persona_session_docs = pd.read_parquet(filtered_dir / "persona_session_docs_filtered.parquet")
print(f"  ✓ Loaded {len(persona_docs)} personas, {len(persona_session_docs)} sessions")

# Trait scoring keywords
print("\n[2/8] Setting up trait scoring dictionaries...")
hedges = {"maybe", "might", "could", "perhaps", "suggest", "consider", "possibly"}
pos_words = {"good", "great", "helpful", "clear", "effective", "robust", "happy", "glad", "excited"}
neg_words = {"bad", "unclear", "risky", "wrong", "unsafe", "sad", "upset", "angry"}
we_words = {"we", "together", "team", "collaborate"}
solo_words = {"alone", "individually", "solo", "self"}
you_words = {"you", "your"}
imp_words = {"one", "people", "users"}
risk_pos = {"bold", "ambitious", "experiment", "iterate", "adventure"}
risk_neg = {"conservative", "cautious", "fallback", "guardrail", "safe"}

def score_traits(text):
    """Calculate trait scores from text"""
    toks = re.findall(r'\b[a-z]+\b', text.lower())
    N = max(len(toks), 1)
    counter = Counter(toks)
    return {
        "directness": 1 - sum(counter[w] for w in hedges) / N,
        "emotional_valence": (sum(counter[w] for w in pos_words) - sum(counter[w] for w in neg_words)) / N,
        "social_orientation": (sum(counter[w] for w in we_words) - sum(counter[w] for w in solo_words)) / N,
        "audience_focus": (sum(counter[w] for w in you_words) - sum(counter[w] for w in imp_words)) / N,
        "risk_orientation": (sum(counter[w] for w in risk_pos) - sum(counter[w] for w in risk_neg)) / N,
    }

# Compute trait scores per session
print("\n[3/8] Computing trait scores for each session...")
trait_df = persona_session_docs.copy()
trait_scores = trait_df["session_text"].apply(score_traits).apply(pd.Series)
trait_df = pd.concat([trait_df, trait_scores], axis=1)
print(f"  ✓ Computed traits for {len(trait_df)} sessions")

# Aggregate traits per persona (mean across sessions)
print("\n[4/8] Aggregating traits per persona...")
trait_cols = ["directness", "emotional_valence", "social_orientation", "audience_focus", "risk_orientation"]
persona_traits = trait_df.groupby("persona_id")[trait_cols].mean().reset_index()
print(f"  ✓ Aggregated traits for {len(persona_traits)} personas")

# Merge with persona_docs
df = persona_docs.merge(persona_traits, on="persona_id", how="left")
print(f"  ✓ Merged with persona metadata")

print(f"\nTrait statistics:")
print(df[trait_cols].describe())

# Standardize traits
print("\n[5/8] Standardizing trait values...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[trait_cols])
print(f"  ✓ Scaled shape: {X_scaled.shape}")

# PCA dimensionality reduction
print("\n[6/8] Performing PCA (5D → 2D)...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df["pca_x"] = X_pca[:, 0]
df["pca_y"] = X_pca[:, 1]
print(f"  ✓ PCA explained variance: PC1={pca.explained_variance_ratio_[0]:.2%}, PC2={pca.explained_variance_ratio_[1]:.2%}")
print(f"  ✓ Total variance explained: {pca.explained_variance_ratio_.sum():.2%}")

# K-Means clustering
print("\n[7/8] Performing K-Means clustering...")
k = 6
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X_pca)
print(f"  ✓ Created {k} clusters")
print(f"\nCluster distribution:")
print(df["cluster"].value_counts().sort_index())

# Cluster profiles
print("\n[8/8] Analyzing cluster profiles...")
cluster_profiles = df.groupby("cluster")[trait_cols].mean()
print("\nCluster trait profiles (mean values):")
print(cluster_profiles.round(3))

# Visualization 1: Cluster profiles heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(cluster_profiles.T, annot=True, fmt=".3f", cmap="coolwarm", center=0)
plt.title("Cluster Trait Profiles (Mean Values)")
plt.xlabel("Cluster")
plt.ylabel("Trait")
plt.tight_layout()
plt.savefig(reports_dir / "cluster_trait_profiles.png", dpi=150)
print(f"\n  ✓ Saved: {reports_dir / 'cluster_trait_profiles.png'}")
plt.close()

# Select representatives (5 per cluster)
def select_representatives(df_cluster, trait_cols, n=5):
    """Select n personas closest to cluster center"""
    center = df_cluster[trait_cols].mean()
    distances = ((df_cluster[trait_cols] - center) ** 2).sum(axis=1)
    return df_cluster.loc[distances.nsmallest(n).index]

representatives = pd.concat([
    select_representatives(df[df["cluster"] == i], trait_cols, n=5)
    for i in range(k)
])
print(f"\n  ✓ Selected {len(representatives)} representatives ({k} clusters × 5)")

# Visualization 2: PCA scatter plot
plt.figure(figsize=(12, 8))

# All data points
sns.scatterplot(
    data=df,
    x="pca_x",
    y="pca_y",
    hue="cluster",
    palette="tab10",
    s=30,
    alpha=0.5,
    legend="full"
)

# Representatives
plt.scatter(
    representatives["pca_x"],
    representatives["pca_y"],
    c="red",
    s=200,
    marker="*",
    edgecolors="black",
    linewidths=1.5,
    label="Representatives",
    zorder=5
)

# Cluster centers
centers_pca = pca.transform(scaler.transform(cluster_profiles))
plt.scatter(
    centers_pca[:, 0],
    centers_pca[:, 1],
    c="black",
    s=300,
    marker="X",
    edgecolors="white",
    linewidths=2,
    label="Cluster Centers",
    zorder=10
)

plt.title(f"Trait-based Persona Clustering (PCA + KMeans, k={k})")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
plt.legend(loc="best")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(reports_dir / "trait_clustering.png", dpi=150)
print(f"  ✓ Saved: {reports_dir / 'trait_clustering.png'}")
plt.close()

# Save results
print("\n" + "=" * 60)
print("SAVING RESULTS")
print("=" * 60)

# Save representative personas
representatives.to_parquet(base_dir / "representative_personas.parquet", index=False)
print(f"  ✓ {base_dir / 'representative_personas.parquet'}")

# Save CSV summary
representatives_summary = representatives[["persona_id", "cluster", "relationship"] + trait_cols].copy()
representatives_summary.to_csv(reports_dir / "selected_representatives.csv", index=False)
print(f"  ✓ {reports_dir / 'selected_representatives.csv'}")

# Save full persona data with traits and clusters
df.to_parquet(filtered_dir / "persona_docs_with_traits.parquet", index=False)
print(f"  ✓ {filtered_dir / 'persona_docs_with_traits.parquet'}")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Total personas: {len(df)}")
print(f"  Clusters: {k}")
print(f"  Representatives: {len(representatives)} (5 per cluster)")
print(f"  Traits analyzed: {', '.join(trait_cols)}")
print(f"  PCA variance explained: {pca.explained_variance_ratio_.sum():.2%}")
print("\n✓ All done!")
