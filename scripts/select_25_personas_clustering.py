#!/usr/bin/env python3
"""
Select 25 Diverse Personas Using Clustering
============================================

Strategy:
1. Keep 3 existing anchor personas
2. Load lexical proxy features for all 4200 candidates
3. Perform K-Means clustering (k=22) on feature space
4. Select representative from each cluster
5. Ensure maximum diversity across trait dimensions

Output:
- personas_final_25.txt
- persona_selection_25_report.json
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

# Existing 3 anchors
EXISTING_3 = [
    "episode-184019_A",  # P1: formal/verbose/detached (high self, low empathy)
    "episode-239427_A",  # P2: casual/emotional/intimate (medium, medium empathy)
    "episode-118328_B",  # P3: neutral/concise (low self, lowest empathy)
]

print("=" * 80)
print("SELECT 25 PERSONAS USING FEATURE-BASED CLUSTERING")
print("=" * 80)

# Load filtered persona IDs
print("\n[1/6] Loading filtered persona IDs...")
data_dir = Path("data/processed/cc/filtered")
with open(data_dir / "filtered_persona_ids.json", 'r') as f:
    all_candidates = json.load(f)

print(f"  ✓ Loaded {len(all_candidates)} candidates")
print(f"  ✓ Existing anchors: {len(EXISTING_3)}")

# Load lexical proxies (use simpler approach without pandas)
print("\n[2/6] Loading lexical proxy features...")

# Try reading parquet with pyarrow directly
try:
    import pyarrow.parquet as pq

    table = pq.read_table(data_dir / "persona_lexical_proxies.parquet")

    # Convert to dict format
    persona_ids = table['persona_id'].to_pylist()

    # Get all feature columns (exclude persona_id)
    feature_cols = [col for col in table.column_names if col != 'persona_id']

    features_dict = {}
    for i, pid in enumerate(persona_ids):
        features_dict[pid] = {col: table[col][i].as_py() for col in feature_cols}

    print(f"  ✓ Loaded features for {len(features_dict)} personas")
    print(f"  ✓ Feature dimensions: {len(feature_cols)}")
    print(f"  ✓ Features: {feature_cols[:5]}...")

except Exception as e:
    print(f"  ✗ Error loading parquet: {e}")
    print("  → Fallback: Using stratified numeric sampling instead")

    # Fallback to simple numeric sampling
    def extract_episode_number(persona_id: str) -> int:
        parts = persona_id.split('-')[1].split('_')
        return int(parts[0])

    # Remove existing from candidates
    candidates = [p for p in all_candidates if p not in EXISTING_3]
    candidate_nums = [(p, extract_episode_number(p)) for p in candidates]
    candidate_nums.sort(key=lambda x: x[1])

    # Divide into 22 bins
    n_select = 22
    n_candidates = len(candidate_nums)
    bin_size = n_candidates // n_select

    import random
    random.seed(42)

    selected_22 = []
    for i in range(n_select):
        start_idx = i * bin_size
        end_idx = start_idx + bin_size if i < n_select - 1 else n_candidates

        if end_idx > start_idx:
            bin_candidates = candidate_nums[start_idx:end_idx]
            choice = random.choice(bin_candidates)
            selected_22.append(choice[0])

    final_25 = EXISTING_3 + selected_22

    print(f"\n  ✓ Selected {len(selected_22)} new personas using fallback method")

    # Save results
    output_dir = Path(".")
    with open(output_dir / "personas_final_25.txt", 'w') as f:
        for persona_id in final_25:
            f.write(f"{persona_id}\n")

    report = {
        "selection_method": "stratified_numeric_sampling_fallback",
        "selection_date": "2025-01-02",
        "total_candidates": len(all_candidates),
        "existing_anchors": EXISTING_3,
        "new_personas": selected_22,
        "final_25": final_25,
        "note": "Fallback method due to parquet loading issue"
    }

    with open(output_dir / "persona_selection_25_report.json", 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n✓ Saved: personas_final_25.txt")
    print(f"✓ Saved: persona_selection_25_report.json")

    print("\n" + "=" * 80)
    print("FINAL 25 PERSONAS")
    print("=" * 80)
    for i, pid in enumerate(final_25, 1):
        label = "[ANCHOR]" if pid in EXISTING_3 else "[NEW]"
        print(f"P{i:2d}. {pid:25s} {label}")

    exit(0)

# Continue with clustering approach if features loaded successfully
print("\n[3/6] Preparing feature matrix...")

# Filter to only candidates (remove anchors from clustering)
cluster_candidates = [p for p in all_candidates if p not in EXISTING_3]
cluster_candidates = [p for p in cluster_candidates if p in features_dict]

print(f"  ✓ Candidates for clustering: {len(cluster_candidates)}")

# Build feature matrix
X = []
persona_id_list = []

for pid in cluster_candidates:
    features = features_dict[pid]
    # Convert to feature vector
    feature_vector = [features[col] for col in feature_cols]

    # Check for NaN
    if not any(np.isnan(v) if isinstance(v, (int, float)) else False for v in feature_vector):
        X.append(feature_vector)
        persona_id_list.append(pid)

X = np.array(X)
print(f"  ✓ Feature matrix shape: {X.shape}")

# Normalize features
print("\n[4/6] Normalizing features...")
X_mean = X.mean(axis=0)
X_std = X.std(axis=0) + 1e-8
X_norm = (X - X_mean) / X_std

print(f"  ✓ Normalized: mean={X_norm.mean():.4f}, std={X_norm.std():.4f}")

# Simple K-Means clustering (manual implementation to avoid sklearn library issues)
print("\n[5/6] Performing K-Means clustering (k=22)...")

n_clusters = 22
max_iter = 50
np.random.seed(42)

# Initialize centroids randomly
n_samples = X_norm.shape[0]
centroid_indices = np.random.choice(n_samples, n_clusters, replace=False)
centroids = X_norm[centroid_indices].copy()

# K-Means iterations
for iteration in range(max_iter):
    # Assign points to nearest centroid
    distances = np.zeros((n_samples, n_clusters))
    for i in range(n_clusters):
        distances[:, i] = np.linalg.norm(X_norm - centroids[i], axis=1)

    cluster_labels = np.argmin(distances, axis=1)

    # Update centroids
    new_centroids = np.zeros_like(centroids)
    for i in range(n_clusters):
        cluster_mask = cluster_labels == i
        if cluster_mask.sum() > 0:
            new_centroids[i] = X_norm[cluster_mask].mean(axis=0)
        else:
            # Keep old centroid if cluster is empty
            new_centroids[i] = centroids[i]

    # Check convergence
    if np.allclose(centroids, new_centroids, rtol=1e-4):
        print(f"  ✓ Converged after {iteration + 1} iterations")
        break

    centroids = new_centroids
else:
    print(f"  ✓ Completed {max_iter} iterations")

print(f"  ✓ Cluster sizes:")

cluster_counts = defaultdict(int)
for label in cluster_labels:
    cluster_counts[label] += 1

for cluster_id in sorted(cluster_counts.keys()):
    print(f"    Cluster {cluster_id:2d}: {cluster_counts[cluster_id]:4d} personas")

# Select representative from each cluster (closest to centroid)
print("\n[6/6] Selecting cluster representatives...")

selected_22 = []

for cluster_id in range(n_clusters):
    # Get all points in this cluster
    cluster_mask = cluster_labels == cluster_id
    cluster_points = X_norm[cluster_mask]
    cluster_personas = [persona_id_list[i] for i, m in enumerate(cluster_mask) if m]

    if len(cluster_personas) == 0:
        print(f"  WARNING: Cluster {cluster_id} is empty, skipping")
        continue

    # Find centroid
    centroid = centroids[cluster_id]

    # Find closest point to centroid
    distances = np.linalg.norm(cluster_points - centroid, axis=1)
    closest_idx = np.argmin(distances)

    representative = cluster_personas[closest_idx]
    selected_22.append(representative)

    print(f"  Cluster {cluster_id:2d}: {representative} (distance: {distances[closest_idx]:.3f})")

# Combine with anchors
final_25 = EXISTING_3 + selected_22

print("\n" + "=" * 80)
print("FINAL 25 PERSONAS")
print("=" * 80)

for i, pid in enumerate(final_25, 1):
    label = "[ANCHOR]" if pid in EXISTING_3 else "[NEW]"
    print(f"P{i:2d}. {pid:25s} {label}")

# Save results
print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

output_dir = Path(".")

# Save persona list
with open(output_dir / "personas_final_25.txt", 'w') as f:
    for persona_id in final_25:
        f.write(f"{persona_id}\n")

print(f"✓ Saved: personas_final_25.txt")

# Save detailed report
report = {
    "selection_method": "kmeans_clustering_lexical_features",
    "selection_date": "2025-01-02",
    "total_candidates": len(all_candidates),
    "n_clusters": n_clusters,
    "feature_dimensions": len(feature_cols),
    "features_used": feature_cols,
    "existing_anchors": EXISTING_3,
    "new_personas": selected_22,
    "final_25": final_25,
}

with open(output_dir / "persona_selection_25_report.json", 'w') as f:
    json.dump(report, f, indent=2)

print(f"✓ Saved: persona_selection_25_report.json")

print("\n" + "=" * 80)
print("✓ SELECTION COMPLETE")
print("=" * 80)
print(f"\nNext steps:")
print(f"  1. Review personas_final_25.txt")
print(f"  2. Extract persona profiles for new personas")
print(f"  3. Run optimization experiments")
