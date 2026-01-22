#!/usr/bin/env python3
"""
より多くのターンでテスト（10ターン）

Usage: python scripts/test_more_turns.py episode-184019_A
"""

import json
import os
import sys
from pathlib import Path
from openai import OpenAI
import time

if len(sys.argv) < 2:
    persona_id = "episode-184019_A"
else:
    persona_id = sys.argv[1]

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("❌ Error: OPENAI_API_KEY not set")
    sys.exit(1)

client = OpenAI(api_key=api_key)

persona_dir = Path(f"personas_cc/{persona_id}")
test_file = persona_dir / "test_turns.json"
if not test_file.exists():
    test_file = persona_dir / "test_turns_selected.json"

with open(test_file) as f:
    data = json.load(f)

turns = data.get("turns", data if isinstance(data, list) else [])
test_turns = turns[:10]  # 最初の10ターン

print(f"Testing {len(test_turns)} turns for {persona_id}...\n")

profile = {
    "speaker_role": "Husband",
    "relationship": "Marriage"
}

selected_count = 0
rejected_count = 0

for i, turn in enumerate(test_turns, 1):
    user_input = turn.get('user', turn.get('user_message', turn.get('input', '')))
    assistant_response = turn.get('assistant', turn.get('assistant_message', turn.get('ground_truth', '')))

    prompt = f"""あなたは会話データの品質を評価するエキスパートです。

## タスク
以下の会話ターンが「ペルソナ特有」かどうか判定してください。

## ペルソナ情報
- 役割: {profile["speaker_role"]}
- 関係性: {profile["relationship"]}

## 会話ターン
Partner's Input:
{user_input}

Persona's Response:
{assistant_response}

## 判定基準

「ペルソナ特有」とは：
1. この役割（{profile["speaker_role"]}）でなければ自然に答えられない
2. 第三者が同じ応答をしたら明らかに不自然
3. 関係性や共有の記憶への言及がある
4. 役割特有の視点や感情が含まれる

「ペルソナ特有ではない」とは：
1. 誰でも同じように答えられる汎用的な応答
2. 一般的な事実や意見
3. ペルソナの視点が不要

## 出力形式（JSON）
{{
    "persona_specific": true/false,
    "confidence": 1-5,
    "reason": "判定理由（日本語、50文字以内）",
    "criteria_met": ["関係性特有", "共有記憶", "役割視点"]
}}

JSONのみを出力してください。"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=300
        )

        result_text = response.choices[0].message.content.strip()
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0]
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0]

        result = json.loads(result_text.strip())

        if result['persona_specific'] and result['confidence'] >= 3:
            selected_count += 1
            print(f"{i:2d}. ✅ Selected (conf={result['confidence']}) - {result['reason'][:40]}...")
        else:
            rejected_count += 1
            print(f"{i:2d}. ❌ Rejected (conf={result['confidence']}) - {result['reason'][:40]}...")

        time.sleep(0.5)

    except Exception as e:
        print(f"{i:2d}. ⚠️  Error: {e}")
        rejected_count += 1

print(f"\n{'='*80}")
print(f"Results:")
print(f"  Selected: {selected_count}/{len(test_turns)} ({selected_count/len(test_turns)*100:.0f}%)")
print(f"  Rejected: {rejected_count}/{len(test_turns)} ({rejected_count/len(test_turns)*100:.0f}%)")
print(f"{'='*80}")

if selected_count == 0:
    print("\n⚠️ Warning: 選定されたターンが0です")
    print("   このペルソナのターンは汎用的すぎる可能性があります")
    print("   または判定基準が厳しすぎる可能性があります")
elif selected_count < 3:
    print(f"\n⚠️ Warning: 選定率が低い ({selected_count/len(test_turns)*100:.0f}%)")
    print("   判定基準を調整する必要があるかもしれません")
else:
    print(f"\n✅ 選定率は適切です ({selected_count/len(test_turns)*100:.0f}%)")
    print("   全ペルソナで実行しても良いでしょう")
