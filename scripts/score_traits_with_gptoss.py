"""
Score Traits with GPT-OSS

This script uses the gpt-oss-20b model to score 30 representative personas
on the 6 core trait dimensions (R1-R5, R8) defined in traits_v3.json.

The scoring is based on semantic understanding of the full conversation history,
going beyond lexical proxies to capture deeper personality patterns.

Usage:
    python scripts/score_traits_with_gptoss.py

Input:
    - data/processed/cc/representative_personas.parquet
    - persona_opt/trait_scorer_prompt_v3.txt

Output:
    - data/processed/cc/representative_traits_v3.parquet
"""

import pandas as pd
from pathlib import Path
import json
from tqdm import tqdm
import re
import sys

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from persona_opt.gpt_oss_wrapper import generate_with_gpt_oss

# ==== Configuration ====
BASE = Path("data/processed/cc")
REP = BASE / "representative_personas.parquet"
PROMPT_FILE = Path("persona_opt/trait_scorer_prompt_v3.txt")
OUT = BASE / "representative_traits_v3.parquet"

# Check if representative personas exist
if not REP.exists():
    print(f"ERROR: {REP} not found!")
    print("Please run persona selection script first to create representative_personas.parquet")
    exit(1)

print("Loading representative personas...")
df = pd.read_parquet(REP)
print(f"Loaded {len(df)} personas")

print("Loading trait scorer prompt...")
with open(PROMPT_FILE, "r", encoding="utf-8") as f:
    trait_prompt = f.read().strip()

print("Using gpt-oss via Python 3.12 wrapper...")
print("Note: This will call the conda environment for each persona")


def score_persona(text: str, max_length: int = 8000) -> dict:
    """
    Score a persona's conversation text on 6 trait dimensions using gpt-oss

    Args:
        text: Full conversation history
        max_length: Maximum characters to use (for context window limits)

    Returns:
        Dictionary with R1-R5, R8 scores and comment
    """
    # Truncate text if too long (keep from beginning as that's often most characteristic)
    if len(text) > max_length:
        text = text[:max_length] + "\n[... truncated for length ...]"

    messages = [
        {"role": "system", "content": trait_prompt},
        {"role": "user", "content": text},
    ]

    # Generate using gpt-oss wrapper (calls Python 3.12 environment)
    decoded = generate_with_gpt_oss(
        messages=messages,
        max_new_tokens=512,
        temperature=0.0,  # Greedy decoding for consistency
    )

    # Try to extract JSON from response
    try:
        # First try direct JSON parse
        data = json.loads(decoded)
        return data
    except json.JSONDecodeError:
        # Try to find JSON object in the text
        json_match = re.search(r'\{[^{}]*"R1"[^{}]*\}', decoded, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                return data
            except json.JSONDecodeError:
                pass

        # Fallback: return neutral scores with error comment
        print(f"⚠️  JSON parse failed. Response preview: {decoded[:200]}")
        return {
            "R1": 0.0,
            "R2": 0.0,
            "R3": 0.0,
            "R4": 0.0,
            "R5": 0.0,
            "R8": 0.0,
            "comment": f"ERROR: Failed to parse JSON from model output",
            "raw_output": decoded[:500]
        }


# ==== Score all personas ====
rows = []
print(f"\nScoring {len(df)} personas with gpt-oss...")
print("This may take several minutes...\n")

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
        # Add default scores for failed persona
        rows.append({
            "persona_id": pid,
            "R1": 0.0,
            "R2": 0.0,
            "R3": 0.0,
            "R4": 0.0,
            "R5": 0.0,
            "R8": 0.0,
            "comment": f"ERROR: {str(e)}",
            "cluster_label": row.get("cluster_label", None)
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
    print("\n✓ All personas scored successfully")
