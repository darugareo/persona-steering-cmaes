#!/usr/bin/env python3
"""
ペルソナ特有ターン選定のテスト（1ペルソナのみ）

使用方法:
    export OPENAI_API_KEY='your-api-key'
    python scripts/test_persona_specific_selection.py episode-184019_A
"""

import json
import os
import sys
from pathlib import Path
from openai import OpenAI
import time

if len(sys.argv) < 2:
    print("Usage: python scripts/test_persona_specific_selection.py <persona_id>")
    print("Example: python scripts/test_persona_specific_selection.py episode-184019_A")
    sys.exit(1)

persona_id = sys.argv[1]

# OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("❌ Error: OPENAI_API_KEY environment variable not set")
    print("\nPlease set it:")
    print("  export OPENAI_API_KEY='your-api-key-here'")
    sys.exit(1)

print(f"✅ API Key found: {api_key[:20]}...")

client = OpenAI(api_key=api_key)

# ペルソナディレクトリ確認
persona_dir = Path(f"personas_cc/{persona_id}")
if not persona_dir.exists():
    print(f"❌ Error: {persona_dir} not found")
    sys.exit(1)

print(f"\n{'='*80}")
print(f"テスト: ペルソナ特有ターン選定")
print(f"{'='*80}")
print(f"Persona: {persona_id}")
print(f"Directory: {persona_dir}")
print(f"{'='*80}\n")

# Profile読み込み（簡易版）
profile = {
    "speaker_role": "Husband" if "husband" in persona_id.lower() else "Unknown",
    "relationship": "Marriage"
}

# Test turnsを読み込み
test_file = persona_dir / "test_turns.json"
if not test_file.exists():
    test_file = persona_dir / "test_turns_selected.json"

if not test_file.exists():
    print(f"❌ Error: test_turns.json not found in {persona_dir}")
    sys.exit(1)

with open(test_file) as f:
    data = json.load(f)

turns = data.get("turns", data if isinstance(data, list) else [])
print(f"✅ Loaded {len(turns)} turns from {test_file.name}\n")

# 最初の3ターンのみテスト
test_turns = turns[:3]

for i, turn in enumerate(test_turns, 1):
    print(f"{'─'*80}")
    print(f"Turn {i}/{len(test_turns)}")
    print(f"{'─'*80}")

    # ターンデータの表示
    user_input = turn.get('user', turn.get('user_message', turn.get('input', '')))
    assistant_response = turn.get('assistant', turn.get('assistant_message', turn.get('ground_truth', '')))

    print(f"\n👤 User: {user_input[:100]}...")
    print(f"🤖 Assistant: {assistant_response[:100]}...")

    # GPT-4oによる判定
    speaker_role = profile.get("speaker_role", "Unknown")
    relationship = profile.get("relationship", "Unknown")

    prompt = f"""あなたは会話データの品質を評価するエキスパートです。

## タスク
以下の会話ターンが「ペルソナ特有」かどうか判定してください。

## ペルソナ情報
- 役割: {speaker_role}
- 関係性: {relationship}

## 会話ターン
Partner's Input:
{user_input}

Persona's Response:
{assistant_response}

## 判定基準

「ペルソナ特有」とは：
1. この役割（{speaker_role}）でなければ自然に答えられない
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

    print(f"\n🔄 Querying GPT-4o...", end=" ", flush=True)

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=300
        )

        result_text = response.choices[0].message.content.strip()

        # JSON部分を抽出
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0]
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0]

        result = json.loads(result_text.strip())

        print("✅ Done")

        # 結果表示
        print(f"\n📊 判定結果:")
        print(f"  Persona Specific: {'✅ YES' if result['persona_specific'] else '❌ NO'}")
        print(f"  Confidence: {result['confidence']}/5")
        print(f"  Reason: {result['reason']}")
        print(f"  Criteria Met: {', '.join(result.get('criteria_met', []))}")

        if result['persona_specific'] and result['confidence'] >= 3:
            print(f"\n  → ✅ このターンは選定される")
        else:
            print(f"\n  → ❌ このターンは選定されない")

    except Exception as e:
        print(f"❌ Error: {e}")

    print()
    time.sleep(1)  # Rate limiting

print(f"{'='*80}")
print(f"✅ テスト完了")
print(f"{'='*80}")
print(f"\n次のステップ:")
print(f"  1. 判定結果が適切であることを確認")
print(f"  2. 問題なければ全ペルソナで実行:")
print(f"     python scripts/select_persona_specific_turns.py")
