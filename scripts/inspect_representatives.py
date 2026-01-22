"""
代表Personaの発話分析スクリプト
"""
import pandas as pd
import numpy as np
import re
from pathlib import Path
from collections import Counter

print("="*80)
print("REPRESENTATIVE PERSONA UTTERANCE ANALYSIS")
print("="*80)

# Paths
base_dir = Path("data/processed/cc")
filtered_dir = base_dir / "filtered"
reports_dir = Path("reports")

# Load data
print("\n[1/5] Loading data...")
representatives = pd.read_parquet(base_dir / "representative_personas.parquet")
session_docs = pd.read_parquet(filtered_dir / "persona_session_docs_filtered.parquet")
print(f"  ✓ {len(representatives)} representative personas")
print(f"  ✓ {len(session_docs)} total sessions")

# Extract representative sessions
representative_ids = representatives["persona_id"].unique().tolist()
rep_sessions = session_docs[session_docs["persona_id"].isin(representative_ids)].copy()

# Merge trait values
trait_cols = ["directness", "emotional_valence", "social_orientation", "audience_focus", "risk_orientation"]
merge_cols = ["persona_id", "cluster"] + trait_cols

rep_sessions = rep_sessions.merge(
    representatives[merge_cols],
    on="persona_id",
    how="left",
    suffixes=('', '_rep')
)

rep_sessions = rep_sessions.sort_values(["cluster", "persona_id", "session_idx"]).reset_index(drop=True)

print(f"\n  ✓ Extracted {len(rep_sessions)} sessions from {len(representative_ids)} representatives")
print(f"  ✓ Average sessions per persona: {len(rep_sessions) / len(representative_ids):.1f}")

# Trait keywords
print("\n[2/5] Defining trait keywords...")
TRAIT_KEYWORDS = {
    "social_orientation": {
        "positive": ["we", "us", "our", "let's", "together", "team"],
        "negative": ["i", "me", "my", "myself", "alone"]
    },
    "emotional_valence": {
        "positive": ["happy", "excited", "great", "wonderful", "fun", "love", "glad"],
        "negative": ["sad", "upset", "angry", "mad", "frustrated", "worried", "bad"]
    },
    "audience_focus": {
        "positive": ["you", "your", "yours"],
        "negative": ["one", "someone", "anyone", "people"]
    },
    "directness": {
        "positive": ["definitely", "absolutely", "certainly", "for sure"],
        "negative": ["maybe", "might", "perhaps", "i guess", "kind of"]
    }
}

# Count keyword function
def count_keywords(text, keywords_dict):
    if not isinstance(text, str):
        return 0, 0
    text_lower = text.lower()
    pos_count = sum(len(re.findall(r'\b' + re.escape(kw) + r'\b', text_lower)) for kw in keywords_dict["positive"])
    neg_count = sum(len(re.findall(r'\b' + re.escape(kw) + r'\b', text_lower)) for kw in keywords_dict["negative"])
    return pos_count, neg_count

# Compute keyword counts
print("\n[3/5] Computing keyword occurrences...")
for trait_name, keywords in TRAIT_KEYWORDS.items():
    pos_counts = []
    neg_counts = []

    for _, row in rep_sessions.iterrows():
        pos, neg = count_keywords(row["session_text"], keywords)
        pos_counts.append(pos)
        neg_counts.append(neg)

    rep_sessions[f"{trait_name}_pos_kw"] = pos_counts
    rep_sessions[f"{trait_name}_neg_kw"] = neg_counts

print("  ✓ Keyword counts computed")

# Overall keyword statistics
print("\n[4/5] Computing statistics...")
keyword_stats = []

for trait_name, keywords in TRAIT_KEYWORDS.items():
    pos_col = f"{trait_name}_pos_kw"
    neg_col = f"{trait_name}_neg_kw"

    keyword_stats.append({
        "trait": trait_name,
        "avg_pos_keywords": rep_sessions[pos_col].mean(),
        "avg_neg_keywords": rep_sessions[neg_col].mean(),
        "pos_sessions_pct": (rep_sessions[pos_col] > 0).mean() * 100,
        "neg_sessions_pct": (rep_sessions[neg_col] > 0).mean() * 100
    })

keyword_stats_df = pd.DataFrame(keyword_stats)

print("\nKeyword occurrence statistics:")
print(keyword_stats_df.to_string(index=False))

# Cluster-level statistics
print("\n[5/5] Analyzing clusters...")
cluster_stats = []

for cluster_id in sorted(rep_sessions["cluster"].unique()):
    cluster_data = rep_sessions[rep_sessions["cluster"] == cluster_id]

    stats = {"cluster": cluster_id, "n_personas": cluster_data["persona_id"].nunique()}

    for trait_name in TRAIT_KEYWORDS.keys():
        stats[f"{trait_name}_pos"] = cluster_data[f"{trait_name}_pos_kw"].mean()
        stats[f"{trait_name}_neg"] = cluster_data[f"{trait_name}_neg_kw"].mean()
        stats[f"{trait_name}_score"] = cluster_data[trait_name].mean()

    cluster_stats.append(stats)

cluster_stats_df = pd.DataFrame(cluster_stats)

print("\nKeyword occurrence by cluster:")
print(cluster_stats_df[["cluster", "n_personas", "social_orientation_pos", "social_orientation_neg",
                         "emotional_valence_pos", "emotional_valence_neg",
                         "audience_focus_pos", "audience_focus_neg"]].to_string(index=False))

# Extract representative examples
print("\n" + "="*80)
print("EXTRACTING REPRESENTATIVE EXAMPLES")
print("="*80)

examples = []

for cluster_id in sorted(rep_sessions["cluster"].unique()):
    cluster_data = rep_sessions[rep_sessions["cluster"] == cluster_id]

    print(f"\nCluster {cluster_id}:")

    # For each main trait, find the session with highest score
    for trait in ["emotional_valence", "social_orientation", "audience_focus"]:
        top_session = cluster_data.nlargest(1, trait).iloc[0]

        excerpt = top_session["session_text"][:300]
        if len(top_session["session_text"]) > 300:
            excerpt += "..."

        examples.append({
            "cluster": cluster_id,
            "persona_id": top_session["persona_id"],
            "session_idx": top_session["session_idx"],
            "relationship": top_session["relationship"],
            "trait_name": trait,
            "trait_value": top_session[trait],
            "pos_keywords": top_session[f"{trait}_pos_kw"],
            "neg_keywords": top_session[f"{trait}_neg_kw"],
            "excerpt": excerpt
        })

        print(f"  {trait:20s}: {top_session[trait]:.4f} (pos={top_session[f'{trait}_pos_kw']}, neg={top_session[f'{trait}_neg_kw']})")

examples_df = pd.DataFrame(examples)

# Save results
print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

# CSV files
examples_df.to_csv(reports_dir / "representative_utterance_examples.csv", index=False)
print(f"  ✓ {reports_dir / 'representative_utterance_examples.csv'}")

keyword_stats_df.to_csv(reports_dir / "keyword_statistics.csv", index=False)
print(f"  ✓ {reports_dir / 'keyword_statistics.csv'}")

cluster_stats_df.to_csv(reports_dir / "cluster_keyword_statistics.csv", index=False)
print(f"  ✓ {reports_dir / 'cluster_keyword_statistics.csv'}")

# Markdown report
print("\nGenerating Markdown report...")
report_lines = []
report_lines.append("# 代表Persona発話分析レポート\n\n")
report_lines.append(f"**分析日時**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
report_lines.append(f"**代表persona数**: {len(representative_ids)}\n\n")
report_lines.append(f"**総セッション数**: {len(rep_sessions)}\n\n")
report_lines.append("---\n\n")

report_lines.append("## 1. Traitキーワード出現統計\n\n")
report_lines.append("```\n")
report_lines.append(keyword_stats_df.to_string(index=False))
report_lines.append("\n```\n")
report_lines.append("\n\n---\n\n")

report_lines.append("## 2. Clusterごとのキーワード傾向\n\n")
report_lines.append("```\n")
report_lines.append(cluster_stats_df[["cluster", "social_orientation_pos", "social_orientation_neg",
                                       "emotional_valence_pos", "emotional_valence_neg",
                                       "audience_focus_pos", "audience_focus_neg"]].to_string(index=False))
report_lines.append("\n```\n")
report_lines.append("\n\n---\n\n")

report_lines.append("## 3. 代表発話例\n\n")
for cluster_id in sorted(examples_df["cluster"].unique()):
    cluster_examples = examples_df[examples_df["cluster"] == cluster_id]
    report_lines.append(f"### Cluster {cluster_id}\n\n")

    for _, row in cluster_examples.iterrows():
        report_lines.append(f"**{row['trait_name']}** (value: {row['trait_value']:.3f}, pos_kw: {row['pos_keywords']}, neg_kw: {row['neg_keywords']})\\n\n")
        report_lines.append(f"- Persona: {row['persona_id']} | Session: {row['session_idx']} | Relationship: {row['relationship']}\n")
        report_lines.append(f"- Excerpt: _{row['excerpt']}_\n\n")

report_path = reports_dir / "representative_persona_analysis.md"
with open(report_path, "w", encoding="utf-8") as f:
    f.writelines(report_lines)

print(f"  ✓ {report_path}")

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"  Total representative personas: {len(representative_ids)}")
print(f"  Total sessions analyzed: {len(rep_sessions)}")
print(f"  Clusters: {rep_sessions['cluster'].nunique()}")
print(f"  Examples extracted: {len(examples_df)}")
print("\n✓ Analysis complete!")

# Display sample examples
print("\n" + "="*80)
print("SAMPLE EXAMPLES (First 3)")
print("="*80)
for i, row in examples_df.head(3).iterrows():
    print(f"\nCluster {row['cluster']} | {row['trait_name']} = {row['trait_value']:.3f}")
    print(f"Persona: {row['persona_id']} | Relationship: {row['relationship']}")
    print(f"Keywords: +{row['pos_keywords']} / -{row['neg_keywords']}")
    print(f"Excerpt: {row['excerpt'][:200]}...")
    print("-" * 80)
