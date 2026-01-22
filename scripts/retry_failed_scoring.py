"""
Retry Failed Trait Scoring

This script retries scoring for personas that failed in the initial run,
using the updated timeout settings.

Usage:
    python scripts/retry_failed_scoring.py
"""

import pandas as pd
from pathlib import Path
import json
from tqdm import tqdm
import re
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from persona_opt.gpt_oss_wrapper import generate_with_gpt_oss

# ==== Configuration ====
BASE = Path("data/processed/cc")
REP = BASE / "representative_personas.parquet"
PROMPT_FILE = Path("persona_opt/trait_scorer_prompt_v3.txt")
EXISTING_SCORES = BASE / "representative_traits_v3.parquet"
OUT = BASE / "representative_traits_v3.parquet"

print("Loading existing scores...")
existing_df = pd.read_parquet(EXISTING_SCORES)

# Find personas with errors
failed_personas = existing_df[existing_df["comment"].str.contains("ERROR", na=False)]
print(f"Found {len(failed_personas)} personas with errors")

if len(failed_personas) == 0:
    print("No failed personas to retry. Exiting.")
    exit(0)

print(f"\nFailed personas:")
for pid in failed_personas["persona_id"].values:
    print(f"  - {pid}")

# Load full persona data
print("\nLoading full persona data...")
full_df = pd.read_parquet(REP)

# Load trait scorer prompt
print("Loading trait scorer prompt...")
with open(PROMPT_FILE, "r", encoding="utf-8") as f:
    trait_prompt = f.read().strip()


def score_persona(text: str, max_length: int = 8000) -> dict:
    """Score a persona with extended timeout"""
    # Truncate text if too long
    if len(text) > max_length:
        text = text[:max_length] + "\n[... truncated for length ...]"

    messages = [
        {"role": "system", "content": trait_prompt},
        {"role": "user", "content": text},
    ]

    # Generate using gpt-oss wrapper (now with 600s timeout)
    decoded = generate_with_gpt_oss(
        messages=messages,
        max_new_tokens=512,
        temperature=0.0,
    )

    # Try to extract JSON
    try:
        data = json.loads(decoded)
        return data
    except json.JSONDecodeError:
        # Try to find JSON in text
        json_match = re.search(r'\{[^{}]*"R1"[^{}]*\}', decoded, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                return data
            except json.JSONDecodeError:
                pass

        # Fallback
        print(f"⚠️  JSON parse failed. Response: {decoded[:200]}")
        return {
            "R1": 0.0,
            "R2": 0.0,
            "R3": 0.0,
            "R4": 0.0,
            "R5": 0.0,
            "R8": 0.0,
            "comment": f"ERROR: Failed to parse JSON",
            "raw_output": decoded[:500]
        }


# Retry failed personas
print(f"\nRetrying {len(failed_personas)} personas with extended timeout (600s)...")
retry_results = []

for _, failed_row in tqdm(failed_personas.iterrows(), total=len(failed_personas), desc="Retrying"):
    pid = failed_row["persona_id"]

    # Get full text
    persona_row = full_df[full_df["persona_id"] == pid]
    if len(persona_row) == 0:
        print(f"\n⚠️  Persona {pid} not found in full data, skipping")
        continue

    text = persona_row.iloc[0]["all_text"]

    try:
        result = score_persona(text)
        result["persona_id"] = pid

        # Add cluster label if available
        if "cluster_label" in persona_row.columns:
            result["cluster_label"] = persona_row.iloc[0]["cluster_label"]

        retry_results.append(result)

    except Exception as e:
        print(f"\n⚠️  Error retrying persona {pid}: {e}")
        # Keep the error result
        retry_results.append({
            "persona_id": pid,
            "R1": 0.0,
            "R2": 0.0,
            "R3": 0.0,
            "R4": 0.0,
            "R5": 0.0,
            "R8": 0.0,
            "comment": f"ERROR (retry): {str(e)}",
            "cluster_label": persona_row.iloc[0].get("cluster_label", None)
        })

# Merge retry results with existing scores
print("\nMerging retry results with existing scores...")
retry_df = pd.DataFrame(retry_results)

# Remove failed personas from existing scores
success_df = existing_df[~existing_df["comment"].str.contains("ERROR", na=False)]

# Combine
final_df = pd.concat([success_df, retry_df], ignore_index=True)

# Save
final_df.to_parquet(OUT, index=False)
print(f"\n✓ Saved updated scores to: {OUT}")

# Check for remaining failures
remaining_failed = final_df[final_df["comment"].str.contains("ERROR", na=False)]
if len(remaining_failed) > 0:
    print(f"\n⚠️  {len(remaining_failed)} personas still have errors after retry")
    print("Remaining failed personas:")
    for pid in remaining_failed["persona_id"].values:
        print(f"  - {pid}")
else:
    print("\n✓ All personas scored successfully!")

print(f"\nFinal statistics:")
print(final_df[["R1", "R2", "R3", "R4", "R5", "R8"]].describe())
