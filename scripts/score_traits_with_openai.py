#!/usr/bin/env python3
"""
Score Traits with OpenAI API (GPT-4o-mini)

This script uses GPT-4o-mini to score 30 representative personas
on the 6 core trait dimensions (R1-R5, R8) defined in traits_v3.json.

The scoring is based on semantic understanding of the full conversation history,
using a reliable instruction-following model with JSON mode.

Usage:
    export OPENAI_API_KEY=your_key_here
    python scripts/score_traits_with_openai.py

Input:
    - data/processed/cc/representative_personas.parquet
    - persona_opt/trait_scorer_prompt_v4.txt

Output:
    - data/processed/cc/representative_traits_v4_openai.parquet
"""

import pandas as pd
from pathlib import Path
import json
from tqdm import tqdm
import os
import sys
from openai import OpenAI

# Check for API key
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("ERROR: OPENAI_API_KEY environment variable not set")
    print("Please set it with: export OPENAI_API_KEY=your_key_here")
    exit(1)

client = OpenAI(api_key=api_key)

# ==== Configuration ====
BASE = Path("data/processed/cc")
REP = BASE / "representative_personas.parquet"
PROMPT_FILE = Path("persona_opt/trait_scorer_prompt_v4.txt")
OUT = BASE / "representative_traits_v4_openai.parquet"

# Check if representative personas exist
if not REP.exists():
    print(f"ERROR: {REP} not found!")
    print("Please run persona selection script first")
    exit(1)

print("Loading representative personas...")
df = pd.read_parquet(REP)
print(f"✓ Loaded {len(df)} personas")

print("Loading trait scorer prompt...")
with open(PROMPT_FILE, "r", encoding="utf-8") as f:
    trait_prompt = f.read().strip()

print(f"✓ Using GPT-4o-mini with JSON mode")
print(f"✓ Estimated cost: ~${len(df) * 0.001:.3f}")


def score_persona(text: str, max_length: int = 16000) -> dict:
    """
    Score a persona's conversation text on 6 trait dimensions using OpenAI API

    Args:
        text: Full conversation history
        max_length: Maximum characters to use (GPT-4o-mini has large context)

    Returns:
        Dictionary with R1-R5, R8 scores and comment
    """
    # Truncate text if too long
    if len(text) > max_length:
        text = text[:max_length] + "\n[... truncated for length ...]"

    messages = [
        {"role": "system", "content": trait_prompt},
        {"role": "user", "content": text},
    ]

    try:
        # Call OpenAI API with JSON mode
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=500,
        )

        # Extract JSON from response
        content = response.choices[0].message.content
        data = json.loads(content)

        # Validate required fields
        required = ["R1", "R2", "R3", "R4", "R5", "R8", "comment"]
        for field in required:
            if field not in data:
                raise ValueError(f"Missing field: {field}")

        # Add raw output for debugging
        data["raw_output"] = content

        return data

    except json.JSONDecodeError as e:
        print(f"\n⚠️  JSON parse failed: {e}")
        print(f"Response: {content[:200]}")
        return {
            "R1": 0.0, "R2": 0.0, "R3": 0.0, "R4": 0.0, "R5": 0.0, "R8": 0.0,
            "comment": f"ERROR: JSON parse failed - {e}",
            "raw_output": content[:500]
        }

    except Exception as e:
        print(f"\n⚠️  API call failed: {e}")
        return {
            "R1": 0.0, "R2": 0.0, "R3": 0.0, "R4": 0.0, "R5": 0.0, "R8": 0.0,
            "comment": f"ERROR: API call failed - {e}",
            "raw_output": ""
        }


# ==== Score all personas ====
rows = []
print(f"\nScoring {len(df)} personas with GPT-4o-mini...")
print("This should take 2-3 minutes...\n")

for _, row in tqdm(df.iterrows(), total=len(df), desc="Scoring"):
    pid = row["persona_id"]
    text = row["all_text"]

    try:
        result = score_persona(text)
        result["persona_id"] = pid

        # Add cluster label if available
        if "cluster_label" in row:
            result["cluster_label"] = row["cluster_label"]

        rows.append(result)

    except Exception as e:
        print(f"\n⚠️  Error scoring persona {pid}: {e}")
        rows.append({
            "persona_id": pid,
            "R1": 0.0, "R2": 0.0, "R3": 0.0, "R4": 0.0, "R5": 0.0, "R8": 0.0,
            "comment": f"ERROR: {str(e)}",
            "cluster_label": row.get("cluster_label", None),
            "raw_output": ""
        })

# ==== Save results ====
out_df = pd.DataFrame(rows)
out_df.to_parquet(OUT, index=False)

print(f"\n✓ Saved trait scores to: {OUT}")
print(f"\nScore summary:")
print(out_df[["R1", "R2", "R3", "R4", "R5", "R8"]].describe())

# Check for any failed parses
failed = out_df[out_df["comment"].str.contains("ERROR", na=False)]
if len(failed) > 0:
    print(f"\n⚠️  {len(failed)} personas had scoring errors")
    print("Review the output file for details")
else:
    print("\n✓ All personas scored successfully!")

# Check trait variance
print("\n" + "="*80)
print("TRAIT VARIANCE CHECK:")
print("="*80)
for trait in ["R1", "R2", "R3", "R4", "R5", "R8"]:
    std = out_df[trait].std()
    range_val = out_df[trait].max() - out_df[trait].min()
    status = "✓" if std > 0.2 and range_val > 0.5 else "⚠️ "
    print(f"{status} {trait}: std={std:.3f}, range={range_val:.3f}")

# Check correlation
print("\n" + "="*80)
print("HIGH CORRELATION CHECK:")
print("="*80)
trait_cols = ["R1", "R2", "R3", "R4", "R5", "R8"]
corr = out_df[trait_cols].corr()
high_corr_count = 0
for i in range(len(trait_cols)):
    for j in range(i+1, len(trait_cols)):
        if abs(corr.iloc[i, j]) > 0.7:
            print(f"⚠️  {trait_cols[i]} <-> {trait_cols[j]}: {corr.iloc[i, j]:.3f}")
            high_corr_count += 1

if high_corr_count == 0:
    print("✓ No high correlations (|r| > 0.7) detected")

print("\n" + "="*80)
print("SCORING COMPLETE!")
print("="*80)
