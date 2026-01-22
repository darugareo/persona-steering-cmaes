"""
Compute Lexical Proxies for Trait Analysis

This script extracts lexical proxy indicators for R1-R8 trait candidates
from persona conversation texts. The proxies are simple frequency-based
metrics that serve as a first-pass screening for trait presence and variance.

Usage:
    python scripts/compute_lexical_proxies.py

Input:
    data/processed/cc/filtered/persona_docs_filtered.parquet

Output:
    data/processed/cc/filtered/persona_lexical_proxies.parquet
    reports/trait_proxy_summary.csv
    reports/trait_proxy_corr_heatmap.png
    reports/trait_proxy_distribution.png
"""

import pandas as pd
import re
from pathlib import Path
from collections import Counter
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ==== Configuration ====
BASE_DIR = Path("data/processed/cc/filtered")
INPUT = BASE_DIR / "persona_docs_filtered.parquet"
OUTPUT = BASE_DIR / "persona_lexical_proxies.parquet"
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)

# ==== Load data ====
print("Loading persona docs...")
df = pd.read_parquet(INPUT)
print(f"Loaded {len(df)} personas")

def lexical_proxies(text: str) -> dict:
    """
    Compute lexical proxy indicators for trait candidates

    Args:
        text: Full conversation text for a persona

    Returns:
        Dictionary of proxy metrics
    """
    t = text.lower()
    toks = re.findall(r"[a-z']+", t)
    N = max(len(toks), 1)
    c = Counter(toks)

    def freq(words):
        """Calculate frequency of word list"""
        return sum(c[w] for w in words) / N

    # Proxy indicators for each trait candidate
    return {
        # R1: Self vs Other focus
        "self_ratio":  freq(["i", "me", "my", "myself"]),
        "you_ratio":   freq(["you", "your", "yours"]),
        "we_ratio":    freq(["we", "us", "our"]),

        # R2: Emotional openness
        "pos_emotion": freq(["happy", "glad", "excited", "love", "good", "great"]),
        "neg_emotion": freq(["sad", "angry", "upset", "worried", "anxious", "hate"]),

        # R3: Agency / Proactive
        "agency_ratio": freq(["decide", "decided", "will", "going", "plan", "try"]),

        # R4: Structure / Planning
        "structure_ratio": freq(["first", "second", "finally", "step", "goal", "plan"]),

        # R5: Positivity / Outlook
        "optimism_ratio": freq(["hope", "might", "maybe", "can", "possible", "positive"]),
        "pessimism_ratio": freq(["no", "never", "nothing", "useless", "pointless"]),

        # R6: Vulnerability / self-disclosure
        "vulnerability_ratio": freq([
            "afraid", "scared", "worried", "insecure", "embarrassed"
        ]),

        # R7: Conflict vs Harmony
        "conflict_ratio": freq([
            "angry", "fight", "argue", "annoyed", "frustrated"
        ]),
        "harmony_ratio": freq([
            "understand", "okay", "i see", "i get", "makes sense"
        ]),

        # R8: Time orientation
        "past_ratio": freq(["used", "remember", "when", "ago"]),
        "future_ratio": freq(["will", "going", "future", "plan", "hope"]),
    }


print("Computing proxies...")
rows = []
for _, row in df.iterrows():
    text = row["all_text"]
    proxies = lexical_proxies(text)
    proxies["persona_id"] = row["persona_id"]
    rows.append(proxies)

proxy_df = pd.DataFrame(rows)
proxy_df.to_parquet(OUTPUT, index=False)

print(f"✓ Saved proxies to: {OUTPUT}")

# ==== Generate statistics and visualizations ====
print("Generating summary and heatmap...")

# Summary statistics
summary = proxy_df.describe()
summary.to_csv(REPORT_DIR / "trait_proxy_summary.csv")
print(f"✓ Saved summary to: {REPORT_DIR / 'trait_proxy_summary.csv'}")

# Correlation heatmap
corr = proxy_df.drop(columns=["persona_id"]).corr()
plt.figure(figsize=(16, 14))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            square=True, linewidths=0.5)
plt.title("Trait Proxy Correlation Heatmap", fontsize=16)
plt.tight_layout()
plt.savefig(REPORT_DIR / "trait_proxy_corr_heatmap.png", dpi=150)
print(f"✓ Saved heatmap to: {REPORT_DIR / 'trait_proxy_corr_heatmap.png'}")

# Distribution plot
plt.figure(figsize=(16, 14))
proxy_df.drop(columns=["persona_id"]).hist(bins=40, figsize=(16, 14))
plt.suptitle("Trait Proxy Distributions", fontsize=16, y=1.0)
plt.tight_layout()
plt.savefig(REPORT_DIR / "trait_proxy_distribution.png", dpi=150)
print(f"✓ Saved distributions to: {REPORT_DIR / 'trait_proxy_distribution.png'}")

# Print screening insights
print("\n" + "="*80)
print("SCREENING INSIGHTS")
print("="*80)

# Check for low variance
print("\nLow Variance Proxies (std < 0.001):")
low_var = summary.loc["std"][summary.loc["std"] < 0.001]
if len(low_var) > 0:
    for col in low_var.index:
        print(f"  - {col}: std={low_var[col]:.6f}")
else:
    print("  None found (all proxies have sufficient variance)")

# Check for high correlation
print("\nHigh Correlation Pairs (|corr| > 0.80):")
high_corr_pairs = []
for i in range(len(corr.columns)):
    for j in range(i+1, len(corr.columns)):
        if abs(corr.iloc[i, j]) > 0.80:
            high_corr_pairs.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))

if len(high_corr_pairs) > 0:
    for col1, col2, corr_val in high_corr_pairs:
        print(f"  - {col1} <-> {col2}: corr={corr_val:.3f}")
else:
    print("  None found (no highly correlated pairs)")

# Check for extreme distributions
print("\nExtreme Distributions (skew > 2.0 or < -2.0):")
from scipy import stats
skew_vals = proxy_df.drop(columns=["persona_id"]).apply(stats.skew)
extreme_skew = skew_vals[(skew_vals > 2.0) | (skew_vals < -2.0)]
if len(extreme_skew) > 0:
    for col in extreme_skew.index:
        print(f"  - {col}: skew={extreme_skew[col]:.3f}")
else:
    print("  None found (all distributions reasonably balanced)")

print("\n" + "="*80)
print("Done. Review reports/ for visualizations.")
print("Next step: Run notebooks/trait_proxy_analysis.ipynb for trait selection")
print("="*80)
