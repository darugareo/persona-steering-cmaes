#!/usr/bin/env python3
"""
Score Trait-v4 (R1,R2,R3,R4,R5,R8) using OpenAI GPT-4o-mini.

This script:
  - Loads representative personas (30人)
  - Loads the new trait_scorer_prompt_v4.txt
  - Calls GPT-4o-mini (JSON mode) for each persona
  - Saves results to representative_traits_v4_openai.parquet
"""

import os
import json
import time
import pandas as pd
from pathlib import Path
from openai import OpenAI

BASE = Path("data/processed/cc")
REP_PATH = BASE / "representative_personas.parquet"
PROMPT_PATH = Path("persona_opt/trait_scorer_prompt_v4.txt")
OUT_PATH = BASE / "representative_traits_v4_openai.parquet"

def load_scorer_prompt():
    return PROMPT_PATH.read_text()

def build_prompt(persona_text, scorer_template):
    return scorer_template.replace("{{PERSONA_TEXT}}", persona_text)

def call_openai(prompt):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are a strict JSON personality evaluator."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    return resp.choices[0].message.content

def main():
    print("=== TRAIT-V4 Semantic Scoring ===")

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY")

    print("Loading personas...")
    df = pd.read_parquet(REP_PATH)

    scorer_template = load_scorer_prompt()
    results = []

    for i, row in df.iterrows():
        pid = row["persona_id"]
        text = row["all_text"]
        print(f"\nScoring: {pid} ({len(text)} chars)")

        prompt = build_prompt(text, scorer_template)
        raw = call_openai(prompt)

        try:
            j = json.loads(raw)
        except json.JSONDecodeError:
            print("JSON parse failed; saving raw output.")
            j = { "R1": 0, "R2": 0, "R3": 0, "R4": 0, "R5": 0, "R8": 0,
                  "comment": f"FAILED_PARSE: {raw[:100]}" }

        j["persona_id"] = pid
        j["raw_output"] = raw
        results.append(j)

        time.sleep(0.2)  # rate control

    out_df = pd.DataFrame(results)
    out_df.to_parquet(OUT_PATH, index=False)
    print(f"\n✓ Saved: {OUT_PATH}")

if __name__ == "__main__":
    main()
