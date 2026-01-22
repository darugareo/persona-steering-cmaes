#!/usr/bin/env python3
"""Test OpenAI scoring on a single persona"""

import pandas as pd
from pathlib import Path
import json
import os
from openai import OpenAI

# Check API key
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("ERROR: OPENAI_API_KEY not set")
    print("Set with: export OPENAI_API_KEY=sk-...")
    exit(1)

client = OpenAI(api_key=api_key)

# Load data
REP = Path("data/processed/cc/representative_personas.parquet")
df = pd.read_parquet(REP)
persona = df.iloc[0]

# Load prompt
PROMPT_FILE = Path("persona_opt/trait_scorer_prompt_v4.txt")
with open(PROMPT_FILE, "r") as f:
    prompt = f.read().strip()

# Test
text = persona["all_text"][:8000]
print(f"Testing persona: {persona['persona_id']}")
print(f"Text length: {len(text)} chars")
print("\nCalling GPT-4o-mini...")

messages = [
    {"role": "system", "content": prompt},
    {"role": "user", "content": text},
]

try:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0.0,
        max_tokens=500,
    )

    content = response.choices[0].message.content
    print("\n" + "="*80)
    print("RAW OUTPUT:")
    print("="*80)
    print(content)
    print()

    data = json.loads(content)
    print("="*80)
    print("PARSED JSON:")
    print("="*80)
    print(json.dumps(data, indent=2))

    print("\n✓ SUCCESS: Valid JSON output received")

except Exception as e:
    print(f"\n✗ ERROR: {e}")
