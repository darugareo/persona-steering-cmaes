#!/usr/bin/env python3
"""
Trait-v4 Validation Script

目的:
  - semantic trait_v4 (R1,R2,R3,R4,R5,R8) がしっかり機能しているか検証
  - 分布/分散/相関/lexical対応/trait間構造をチェック
  - 可視化 & markdown レポート生成
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

BASE = Path("data/processed/cc")
SEM = BASE / "representative_traits_v4_openai.parquet"
LEX = BASE / "filtered/persona_lexical_proxies.parquet"
REP = BASE / "representative_personas.parquet"

OUT_DIR = Path("reports/trait_v4_validation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAITS = ["R1","R2","R3","R4","R5","R8"]

def main():
    print("=== VALIDATING TRAIT V4 ===")

    sem = pd.read_parquet(SEM)
    lex = pd.read_parquet(LEX)

    # merge for persona-level matching
    df = sem.merge(lex, on="persona_id", how="left")

    # Save semantic summary
    sem[TRAITS].describe().to_csv(OUT_DIR/"semantic_summary.csv")
    print("✓ Saved semantic_summary.csv")

    # Distribution plots
    plt.figure(figsize=(12,8))
    sem[TRAITS].hist(bins=12, figsize=(12,8))
    plt.tight_layout()
    plt.savefig(OUT_DIR/"semantic_distribution.png")
    plt.close()
    print("✓ Saved semantic_distribution.png")

    # Correlation heatmap (semantic only)
    plt.figure(figsize=(8,6))
    sns.heatmap(sem[TRAITS].corr(), annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Semantic Trait Correlation")
    plt.tight_layout()
    plt.savefig(OUT_DIR/"pairwise_correlation.png")
    plt.close()
    print("✓ Saved pairwise_correlation.png")

    # Semantic vs lexical alignment (scatter)
    if "expressiveness_ratio" in df.columns:
        plt.figure(figsize=(6,4))
        sns.scatterplot(x=df["R2"], y=df["expressiveness_ratio"])
        plt.xlabel("Semantic Expressiveness (R2)")
        plt.ylabel("Lexical Expressiveness Ratio")
        plt.tight_layout()
        plt.savefig(OUT_DIR/"semantic_vs_lexical_expressiveness.png")
        plt.close()

    if "assertive_ratio" in df.columns:
        plt.figure(figsize=(6,4))
        sns.scatterplot(x=df["R3"], y=df["assertive_ratio"])
        plt.xlabel("Semantic Assertiveness (R3)")
        plt.ylabel("Lexical Assertiveness Ratio")
        plt.tight_layout()
        plt.savefig(OUT_DIR/"semantic_vs_lexical_assertiveness.png")
        plt.close()

    # Markdown report
    report = OUT_DIR/"trait_v4_validation_report.md"
    with report.open("w", encoding="utf-8") as f:
        f.write("# Trait-v4 Validation Report\n\n")
        f.write("## Summary\n")
        f.write("```\n")
        f.write(sem[TRAITS].describe().to_string())
        f.write("\n```\n\n")
        f.write("## Notes\n")
        f.write("- Check std > 0.2 for strong traits\n")
        f.write("- Check correlation < 0.7 for independence (semantic)\n")
        f.write("- Compare semantic vs lexical scatter plots\n")

    print(f"✓ Saved report: {report}")

if __name__ == "__main__":
    main()
