#!/usr/bin/env python3
"""
Generate USER_PERSONA_PREFERENCES summary
from semantic trait_v4 (R1-R6) using GPT-4o-mini.

Output:
  data/processed/cc/preference_summary.txt
"""

import json
import pandas as pd
from pathlib import Path
from openai import OpenAI
import os
import time

# Paths
BASE = Path("data/processed/cc")
TRAITS_PATH = BASE / "representative_traits_v4_openai.parquet"
OUT_PATH = BASE / "preference_summary.txt"

# Load semantic traits
print("Loading semantic traits (trait_v4)...")
df = pd.read_parquet(TRAITS_PATH)

# Average traits for the 30 representatives
avg_traits = df[["R1","R2","R3","R4","R5","R8"]].mean().to_dict()

print("Average trait values:")
print(avg_traits)


# Build prompt
def build_prompt(traits):
    return f"""
あなたのタスクは、以下の性格スコア（-1〜+1）を読み取り、
この人物が「どのような回答スタイルを好むか」を自然言語で説明することです。

必ず以下の6軸について、それぞれ1〜2文の「好みの説明」に変換してください：

R1 (Self-Other Focus):
  -1 = 他者中心、相手への配慮が強い
  +1 = 自己中心、自分語りや主張が多い

R2 (Expressiveness):
  -1 = 簡潔、最低限の説明
  +1 = 詳細、説明的、感情や補足が多い

R3 (Assertiveness):
  -1 = 控えめ、提案も弱い
  +1 = 強め、断言、ハッキリ言う

R4 (Structure / Planning):
  -1 = 自由形式
  +1 = 構造化・ステップ形式

R5 (Outlook Valence):
  -1 = ネガティブ寄り
  +1 = ポジティブ寄り

R8 (Time Orientation):
  -1 = 過去に焦点
  +1 = 未来に焦点

与えられたtrait値:
{json.dumps(traits, indent=2)}

出力形式：

USER_PERSONA_PREFERENCES:
- self_other_focus: ...
- expressiveness: ...
- assertiveness: ...
- structure: ...
- outlook: ...
- time_orientation: ...

文章は簡潔にし、日本語で書くこと。
"""


# Run OpenAI
def call_openai(prompt):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert personality interpreter."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    return resp.choices[0].message.content


# Generate summary
prompt = build_prompt(avg_traits)
print("\nCalling GPT-4o-mini to generate preference summary...")
summary = call_openai(prompt)

print("\nGenerated summary:")
print(summary)

# Save
OUT_PATH.write_text(summary, encoding="utf-8")
print(f"\nSaved to: {OUT_PATH}")
