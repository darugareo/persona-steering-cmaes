#!/usr/bin/env python3
"""
Test the strengthened trait scorer prompt (v4) on a single persona
"""

import pandas as pd
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from persona_opt.gpt_oss_wrapper import generate_with_gpt_oss

# Load prompt
PROMPT_FILE = Path("persona_opt/trait_scorer_prompt_v4.txt")
with open(PROMPT_FILE, "r", encoding="utf-8") as f:
    prompt = f.read().strip()

# Load one representative persona
REP = Path("data/processed/cc/representative_personas.parquet")
df = pd.read_parquet(REP)

# Test on the first persona
persona = df.iloc[0]
persona_id = persona["persona_id"]
text = persona["all_text"][:8000]  # Truncate for testing

print("="*80)
print(f"Testing trait_scorer_prompt_v4 on: {persona_id}")
print("="*80)
print(f"\nInput text preview (first 300 chars):")
print(text[:300])
print("...")
print()

# Build messages
messages = [
    {"role": "system", "content": prompt},
    {"role": "user", "content": text},
]

print("Calling gpt-oss with strengthened prompt...")
print("(This may take 30-60 seconds)\n")

try:
    result = generate_with_gpt_oss(
        messages=messages,
        max_new_tokens=512,
        temperature=0.0,
    )

    print("="*80)
    print("RAW OUTPUT:")
    print("="*80)
    print(result)
    print()

    # Try to parse JSON
    print("="*80)
    print("PARSING ATTEMPT:")
    print("="*80)

    try:
        # Direct parse
        data = json.loads(result)
        print("✓ SUCCESS: Valid JSON parsed directly")
        print(json.dumps(data, indent=2))

    except json.JSONDecodeError as e:
        print(f"✗ Direct parse failed: {e}")
        print("\nAttempting to extract JSON from text...")

        # Try to find JSON in output
        import re
        json_match = re.search(r'\{[^{}]*"R1"[^{}]*\}', result, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                print("✓ SUCCESS: JSON extracted from text")
                print(json.dumps(data, indent=2))
            except json.JSONDecodeError:
                print("✗ Extracted text is not valid JSON")
                print(f"Extracted: {json_match.group(0)[:200]}")
        else:
            print("✗ No JSON pattern found in output")

            # Check if output starts with analysis/reasoning
            if result.strip().startswith(("analysis", "Analysis", "let me", "Let me", "I will")):
                print("\n⚠️  Model is still producing reasoning preamble")
                print("    This suggests the prompt is not strong enough for this model")

except Exception as e:
    print(f"✗ ERROR: {e}")

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
